from submission_paths import path as _submission_path
import argparse
import csv
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
WORKSPACE_DIR = Path(str(_submission_path('project', '')))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
from code_strategy3 import Strategy3Config, Strategy3LLMClient, load_strategy3_runtime_config
from code_strategy3.decision import attach_task2_risk_decision
from code_strategy3.prompts import build_task2_messages
from code_strategy3.prompts.task2 import TASK2_PROMPT_VERSION
from code_strategy3.prompts.task2 import resolve_task2_prompt_version
from code_strategy3.python_settings import DEFAULT_STRATEGY3_CONFIG
from code_strategy3.schemas import survival_schema
from code_strategy3.tasks import build_task2_sample_summary, load_task2_labels, load_task2_processed_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Batch evaluation for Strategy 3 Task 2 survival risk.')
    parser.add_argument('--max_samples', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--sample_ids', type=str, default=None)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--use_env', action='store_true')
    parser.add_argument('--profile', type=str, default=None)
    parser.add_argument('--no_cache', action='store_true')
    parser.add_argument('--batch_backend', action='store_true')
    parser.add_argument('--batch_mode', type=str, default=None, choices=['sync', 'concurrent', 'bailian_async'])
    parser.add_argument('--batch_workers', type=int, default=6)
    parser.add_argument('--batch_poll_interval', type=int, default=15)
    parser.add_argument('--batch_completion_window', type=str, default='24h')
    parser.add_argument('--prompt_version', type=str, default=TASK2_PROMPT_VERSION, choices=['v2.1', 'v2.2.1'])
    return parser.parse_args()

def load_runtime_config(use_env: bool, profile_name: str | None=None) -> Strategy3Config:
    if profile_name:
        return load_strategy3_runtime_config(profile_name=profile_name, use_env=use_env)
    if use_env:
        return Strategy3Config.from_env()
    return Strategy3Config.from_mapping(DEFAULT_STRATEGY3_CONFIG)

def resolve_batch_mode(args) -> str:
    if args.batch_mode:
        return str(args.batch_mode)
    if args.batch_backend:
        return 'concurrent'
    return 'sync'

def main():
    args = parse_args()
    config = load_runtime_config(args.use_env, args.profile)
    prompt_version = resolve_task2_prompt_version(args.prompt_version)
    output_subdir = f"task2_prompt_{prompt_version.replace('.', '').replace('v', 'v')}_full"
    output_dir = Path(args.output_dir) if args.output_dir else config.results_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_task2_processed_dataset()
    labels = load_task2_labels()
    sample_ids = resolve_sample_ids(labels, limit=args.max_samples, sample_ids_arg=args.sample_ids)
    schema = survival_schema()
    raw_output_path = output_dir / f'task2_n{len(sample_ids)}_raw.jsonl'
    prediction_path = output_dir / f'task2_n{len(sample_ids)}_predictions.csv'
    summary_path = output_dir / f'task2_n{len(sample_ids)}_summary.json'
    progress_path = output_dir / f'task2_n{len(sample_ids)}_progress.json'
    if args.dry_run:
        sample_summary = build_task2_sample_summary(df, sample_id=sample_ids[0])
        system_prompt, user_prompt = build_task2_messages(sample_summary, prompt_version=prompt_version)
        payload = {'task_name': 'task2', 'prompt_version': prompt_version, 'system_prompt': system_prompt, 'user_prompt': user_prompt, 'schema': schema, 'sample_summary': sample_summary, 'sample_ids': sample_ids}
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return
    client = Strategy3LLMClient(config)
    batch_mode = resolve_batch_mode(args)
    rows = []
    started_at = time.time()
    prepared_requests = build_prepared_requests(df=df, sample_ids=sample_ids, schema=schema, prompt_version=prompt_version)
    initialized = initialize_output_files(raw_output_path=raw_output_path, prediction_path=prediction_path, progress_path=progress_path, total_samples=len(sample_ids), batch_mode=batch_mode, prompt_version=prompt_version)
    with raw_output_path.open('a', encoding='utf-8') as raw_handle, prediction_path.open('a', encoding='utf-8', newline='') as prediction_handle:
        writer = build_prediction_writer(prediction_handle)
        if initialized['needs_prediction_header']:
            writer.writeheader()
            prediction_handle.flush()
        if batch_mode == 'sync':
            run_sync_mode(client=client, prepared_requests=prepared_requests, schema=schema, labels=labels, raw_handle=raw_handle, prediction_writer=writer, prediction_handle=prediction_handle, rows=rows, progress_path=progress_path, batch_mode=batch_mode, batch_workers=args.batch_workers, started_at=started_at, use_cache=not args.no_cache, prompt_version=prompt_version)
        elif batch_mode == 'concurrent':
            run_concurrent_mode(client=client, prepared_requests=prepared_requests, schema=schema, labels=labels, raw_handle=raw_handle, prediction_writer=writer, prediction_handle=prediction_handle, rows=rows, progress_path=progress_path, batch_mode=batch_mode, batch_workers=args.batch_workers, started_at=started_at, use_cache=not args.no_cache, prompt_version=prompt_version)
        else:
            run_bailian_async_mode(client=client, prepared_requests=prepared_requests, labels=labels, raw_handle=raw_handle, prediction_writer=writer, prediction_handle=prediction_handle, rows=rows, progress_path=progress_path, batch_mode=batch_mode, batch_workers=args.batch_workers, batch_poll_interval=args.batch_poll_interval, batch_completion_window=args.batch_completion_window, started_at=started_at, use_cache=not args.no_cache, prompt_version=prompt_version)
    summary = build_summary(sample_ids=sample_ids, rows=rows, predictions_path=prediction_path, raw_path=raw_output_path, summary_path=summary_path, progress_path=progress_path, batch_mode=batch_mode, batch_workers=args.batch_workers, prompt_version=prompt_version)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_progress_snapshot(progress_path=progress_path, sample_ids=sample_ids, rows=rows, batch_mode=batch_mode, batch_workers=args.batch_workers, current_sample_id=None, started_at=started_at, completed=True, phase='completed', prompt_version=prompt_version)

def resolve_sample_ids(labels: pd.DataFrame, limit: int, sample_ids_arg: str | None) -> list[str]:
    if sample_ids_arg:
        return [sample_id.strip() for sample_id in sample_ids_arg.split(',') if sample_id.strip()]
    limit = min(limit, int(labels.shape[0]))
    early_cutoff = float(labels['duration'].quantile(0.33))
    late_cutoff = float(labels['duration'].quantile(0.67))
    early_event = labels[(labels['event'] == 1) & (labels['duration'] <= early_cutoff)].index.tolist()
    late_censored = labels[(labels['event'] == 0) & (labels['duration'] >= late_cutoff)].index.tolist()
    remaining = [str(sample_id) for sample_id in labels.index.tolist() if sample_id not in set(early_event).union(late_censored)]
    early_target = limit // 3
    late_target = limit // 3
    middle_target = limit - early_target - late_target
    selected = early_event[:early_target] + late_censored[:late_target] + remaining[:middle_target]
    if len(selected) < limit:
        seen = set(selected)
        for sample_id in labels.index.tolist():
            sample_id = str(sample_id)
            if sample_id not in seen:
                selected.append(sample_id)
                seen.add(sample_id)
            if len(selected) >= limit:
                break
    return [str(sample_id) for sample_id in selected[:limit]]

def build_prepared_requests(*, df: pd.DataFrame, sample_ids: list[str], schema: dict, prompt_version: str) -> list[dict]:
    prepared_requests = []
    for sample_id in tqdm(sample_ids, desc='Building Task2 prompts', unit='sample'):
        sample_summary = build_task2_sample_summary(df, sample_id=sample_id)
        system_prompt, user_prompt = build_task2_messages(sample_summary, prompt_version=prompt_version)
        prepared_requests.append({'sample_id': str(sample_id), 'system_prompt': system_prompt, 'user_prompt': user_prompt, 'schema': schema, 'prompt_version': prompt_version})
    return prepared_requests

def initialize_output_files(*, raw_output_path: Path, prediction_path: Path, progress_path: Path, total_samples: int, batch_mode: str, prompt_version: str) -> dict:
    raw_output_path.write_text('', encoding='utf-8')
    prediction_path.write_text('', encoding='utf-8')
    progress_path.write_text(json.dumps({'task_name': 'task2', 'prompt_version': prompt_version, 'completed_samples': 0, 'total_samples': total_samples, 'progress_fraction': 0.0, 'batch_mode': batch_mode, 'completed': False, 'phase': 'initialized'}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'needs_prediction_header': True}

def build_prediction_writer(handle) -> csv.DictWriter:
    return csv.DictWriter(handle, fieldnames=['sample_id', 'duration', 'event', 'risk_group', 'risk_score', 'confidence', 'total_tokens', 'rationale'])

def append_task2_result(*, raw_handle, prediction_writer: csv.DictWriter, prediction_handle, sample_id: str, result: dict, labels: pd.DataFrame, prompt_version: str) -> dict:
    result = attach_task2_risk_decision(result)
    raw_handle.write(json.dumps({'task_name': 'task2', 'prompt_version': prompt_version, 'sample_id': sample_id, 'result': result}, ensure_ascii=False) + '\n')
    raw_handle.flush()
    parsed = result.get('parsed_response', {})
    derived = result.get('derived_decision', {})
    row = {'sample_id': sample_id, 'duration': float(labels.loc[sample_id, 'duration']), 'event': int(labels.loc[sample_id, 'event']), 'risk_group': str(derived.get('risk_group', parsed.get('risk_group', 'intermediate'))), 'risk_score': round(float(derived.get('risk_score', parsed.get('risk_score', 0.5))), 4), 'confidence': round(float(parsed.get('confidence', 0.5)), 4), 'rationale': parsed.get('rationale', []), 'total_tokens': result.get('raw_response', {}).get('usage', {}).get('total_tokens')}
    prediction_writer.writerow({**row, 'rationale': json.dumps(row['rationale'], ensure_ascii=False)})
    prediction_handle.flush()
    return row

def run_sync_mode(*, client: Strategy3LLMClient, prepared_requests: list[dict], schema: dict, labels: pd.DataFrame, raw_handle, prediction_writer: csv.DictWriter, prediction_handle, rows: list[dict], progress_path: Path, batch_mode: str, batch_workers: int, started_at: float, use_cache: bool, prompt_version: str) -> None:
    progress_bar = tqdm(total=len(prepared_requests), desc='Running Task2', unit='sample')
    try:
        for idx, request in enumerate(prepared_requests, start=1):
            result = client.generate_json(system_prompt=request['system_prompt'], user_prompt=request['user_prompt'], schema=schema, use_cache=use_cache)
            row = append_task2_result(raw_handle=raw_handle, prediction_writer=prediction_writer, prediction_handle=prediction_handle, sample_id=str(request['sample_id']), result=result, labels=labels, prompt_version=prompt_version)
            rows.append(row)
            progress_bar.update(1)
            progress_bar.set_postfix(sample=row['sample_id'], risk=row['risk_group'])
            print_task2_progress(idx=idx, total=len(prepared_requests), row=row, started_at=started_at)
            write_progress_snapshot(progress_path=progress_path, sample_ids=[str(item['sample_id']) for item in prepared_requests], rows=rows, batch_mode=batch_mode, batch_workers=batch_workers, current_sample_id=str(request['sample_id']), started_at=started_at, completed=False, phase='running', prompt_version=prompt_version)
    finally:
        progress_bar.close()

def run_concurrent_mode(*, client: Strategy3LLMClient, prepared_requests: list[dict], schema: dict, labels: pd.DataFrame, raw_handle, prediction_writer: csv.DictWriter, prediction_handle, rows: list[dict], progress_path: Path, batch_mode: str, batch_workers: int, started_at: float, use_cache: bool, prompt_version: str) -> None:
    worker_count = max(1, min(int(batch_workers), len(prepared_requests)))
    progress_bar = tqdm(total=len(prepared_requests), desc='Running Task2', unit='sample')
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_request = {executor.submit(client.generate_json, system_prompt=request['system_prompt'], user_prompt=request['user_prompt'], schema=schema, use_cache=use_cache): request for request in prepared_requests}
            for idx, future in enumerate(as_completed(future_to_request), start=1):
                request = future_to_request[future]
                result = future.result()
                row = append_task2_result(raw_handle=raw_handle, prediction_writer=prediction_writer, prediction_handle=prediction_handle, sample_id=str(request['sample_id']), result=result, labels=labels, prompt_version=prompt_version)
                rows.append(row)
                progress_bar.update(1)
                progress_bar.set_postfix(sample=row['sample_id'], risk=row['risk_group'])
                print_task2_progress(idx=idx, total=len(prepared_requests), row=row, started_at=started_at)
                write_progress_snapshot(progress_path=progress_path, sample_ids=[str(item['sample_id']) for item in prepared_requests], rows=rows, batch_mode=batch_mode, batch_workers=batch_workers, current_sample_id=str(request['sample_id']), started_at=started_at, completed=False, phase='running', prompt_version=prompt_version)
    finally:
        progress_bar.close()

def run_bailian_async_mode(*, client: Strategy3LLMClient, prepared_requests: list[dict], labels: pd.DataFrame, raw_handle, prediction_writer: csv.DictWriter, prediction_handle, rows: list[dict], progress_path: Path, batch_mode: str, batch_workers: int, batch_poll_interval: int, batch_completion_window: str, started_at: float, use_cache: bool, prompt_version: str) -> None:
    print('Submitting Bailian async batch. Per-sample progress will update after the remote batch completes.')
    write_progress_snapshot(progress_path=progress_path, sample_ids=[str(item['sample_id']) for item in prepared_requests], rows=rows, batch_mode=batch_mode, batch_workers=batch_workers, current_sample_id=None, started_at=started_at, completed=False, phase='submitted_remote_batch', prompt_version=prompt_version)
    results = client.batch_generate_json(requests=prepared_requests, use_cache=use_cache, max_workers=batch_workers, batch_mode=batch_mode, poll_interval_seconds=batch_poll_interval, completion_window=batch_completion_window)
    progress_bar = tqdm(total=len(prepared_requests), desc='Writing Task2 results', unit='sample')
    try:
        for idx, request in enumerate(prepared_requests, start=1):
            row = append_task2_result(raw_handle=raw_handle, prediction_writer=prediction_writer, prediction_handle=prediction_handle, sample_id=str(request['sample_id']), result=results[idx - 1], labels=labels, prompt_version=prompt_version)
            rows.append(row)
            progress_bar.update(1)
            progress_bar.set_postfix(sample=row['sample_id'], risk=row['risk_group'])
            print_task2_progress(idx=idx, total=len(prepared_requests), row=row, started_at=started_at)
            write_progress_snapshot(progress_path=progress_path, sample_ids=[str(item['sample_id']) for item in prepared_requests], rows=rows, batch_mode=batch_mode, batch_workers=batch_workers, current_sample_id=str(request['sample_id']), started_at=started_at, completed=False, phase='writing_completed_batch', prompt_version=prompt_version)
    finally:
        progress_bar.close()

def print_task2_progress(*, idx: int, total: int, row: dict, started_at: float) -> None:
    progress_pct = 100.0 * idx / total if total else 100.0
    elapsed_seconds = max(0.0, time.time() - started_at)
    avg_seconds = elapsed_seconds / idx if idx else 0.0
    remaining_seconds = max(0.0, avg_seconds * (total - idx))
    print(f"[task2 {idx}/{total} {progress_pct:5.1f}%] {row['sample_id']} duration={row['duration']:.1f} event={row['event']} risk={row['risk_group']} score={row['risk_score']:.4f} conf={row['confidence']:.4f} elapsed={elapsed_seconds / 60:.1f}m eta={remaining_seconds / 60:.1f}m")

def write_progress_snapshot(*, progress_path: Path, sample_ids: list[str], rows: list[dict], batch_mode: str, batch_workers: int, current_sample_id: str | None, started_at: float, completed: bool, phase: str, prompt_version: str) -> None:
    completed_count = len(rows)
    total_count = len(sample_ids)
    elapsed_seconds = max(0.0, time.time() - started_at)
    avg_seconds = elapsed_seconds / completed_count if completed_count else None
    eta_seconds = 0.0 if completed else (avg_seconds or 0.0) * max(0, total_count - completed_count)
    snapshot = {'task_name': 'task2', 'prompt_version': prompt_version, 'completed_samples': completed_count, 'total_samples': total_count, 'progress_fraction': round(completed_count / total_count, 4) if total_count else 1.0, 'current_sample_id': current_sample_id, 'completed_sample_ids_tail': [str(row['sample_id']) for row in rows[-10:]], 'elapsed_seconds': round(elapsed_seconds, 2), 'estimated_remaining_seconds': round(eta_seconds, 2), 'batch_mode': batch_mode, 'batch_backend': batch_mode != 'sync', 'batch_workers': int(batch_workers) if batch_mode == 'concurrent' else 1, 'phase': phase, 'completed': bool(completed)}
    progress_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def build_summary(*, sample_ids: list[str], rows: list[dict], predictions_path: Path, raw_path: Path, summary_path: Path, progress_path: Path, batch_mode: str, batch_workers: int, prompt_version: str) -> dict:
    durations = [float(row['duration']) for row in rows]
    events = [int(row['event']) for row in rows]
    risk_scores = [float(row['risk_score']) for row in rows]
    confidence_values = [float(row['confidence']) for row in rows]
    token_values = [int(row['total_tokens']) for row in rows if row['total_tokens'] is not None]
    risk_group_distribution: dict[str, int] = {}
    for row in rows:
        risk_group_distribution[row['risk_group']] = risk_group_distribution.get(row['risk_group'], 0) + 1
    mean_risk_event_1 = [row['risk_score'] for row in rows if row['event'] == 1]
    mean_risk_event_0 = [row['risk_score'] for row in rows if row['event'] == 0]
    c_index = concordance_index_survival(durations, risk_scores, events)
    return {'task_name': 'task2', 'prompt_version': prompt_version, 'n_samples': len(rows), 'sample_ids': sample_ids, 'risk_group_distribution': risk_group_distribution, 'c_index': round(float(c_index), 4) if not math.isnan(c_index) else None, 'mean_risk_score': round(sum(risk_scores) / len(risk_scores), 4) if risk_scores else None, 'mean_confidence': round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else None, 'mean_duration': round(sum(durations) / len(durations), 4) if durations else None, 'event_rate': round(sum(events) / len(events), 4) if events else None, 'mean_risk_score_event1': round(sum(mean_risk_event_1) / len(mean_risk_event_1), 4) if mean_risk_event_1 else None, 'mean_risk_score_event0': round(sum(mean_risk_event_0) / len(mean_risk_event_0), 4) if mean_risk_event_0 else None, 'mean_total_tokens': round(sum(token_values) / len(token_values), 2) if token_values else None, 'batch_mode': batch_mode, 'batch_backend': batch_mode != 'sync', 'batch_workers': int(batch_workers) if batch_mode == 'concurrent' else 1, 'output_files': {'raw_jsonl': str(raw_path), 'predictions_csv': str(predictions_path), 'summary_json': str(summary_path), 'progress_json': str(progress_path)}}

def concordance_index_survival(durations: list[float], predictions: list[float], events: list[int]) -> float:
    concordant = 0.0
    permissible = 0.0
    ties = 0.0
    n_samples = len(durations)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            ti, tj = (durations[i], durations[j])
            pi, pj = (predictions[i], predictions[j])
            ei, ej = (events[i], events[j])
            if ti == tj and ei == 0 and (ej == 0):
                continue
            if ti < tj and ei == 1:
                permissible += 1
                if pi > pj:
                    concordant += 1
                elif pi == pj:
                    ties += 1
            elif tj < ti and ej == 1:
                permissible += 1
                if pj > pi:
                    concordant += 1
                elif pi == pj:
                    ties += 1
    if permissible == 0:
        return math.nan
    return (concordant + 0.5 * ties) / permissible
if __name__ == '__main__':
    main()
