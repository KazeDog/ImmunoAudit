from submission_paths import path as _submission_path
import argparse
import json
import math
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
WORKSPACE_DIR = Path(str(_submission_path('project', '')))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
from code_strategy3 import Strategy3Config, Strategy3LLMClient, load_strategy3_runtime_config
from code_strategy3.decision import attach_task5_risk_decision
from code_strategy3.prompts import build_task5_messages
from code_strategy3.prompts.task5 import TASK5_PROMPT_VERSION
from code_strategy3.python_settings import DEFAULT_STRATEGY3_CONFIG
from code_strategy3.schemas import survival_schema
from code_strategy3.tasks import build_task5_sample_summary, load_task5_labels, load_task5_processed_dataset
from code_strategy3.tasks.task5 import DEFAULT_TASK5_DATASET, TASK5_DATASET_SOURCES
TASK5_OUTPUT_SUBDIR = 'task5_survival_batch'
ALL_TASK5_DATASETS = list(TASK5_DATASET_SOURCES.keys())

def parse_args():
    parser = argparse.ArgumentParser(description='Batch evaluation for Task 5 survival strategy.')
    parser.add_argument('--dataset', type=str, default=DEFAULT_TASK5_DATASET)
    parser.add_argument('--datasets', type=str, default=None)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--sample_ids', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_prefix', type=str, default=None)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--use_env', action='store_true')
    parser.add_argument('--profile', type=str, default=None)
    parser.add_argument('--no_cache', action='store_true')
    parser.add_argument('--batch_backend', action='store_true')
    parser.add_argument('--batch_mode', type=str, default=None, choices=['sync', 'concurrent', 'bailian_async'])
    parser.add_argument('--batch_workers', type=int, default=6)
    parser.add_argument('--batch_poll_interval', type=int, default=15)
    parser.add_argument('--batch_completion_window', type=str, default='24h')
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
    output_dir = Path(args.output_dir) if args.output_dir else config.results_dir / TASK5_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = resolve_datasets(args.dataset, args.datasets)
    aggregate_prefix = args.output_prefix or build_multi_dataset_prefix(datasets, args.limit)
    if args.dry_run:
        manifest = build_dry_run_manifest(datasets=datasets, limit=args.limit, sample_ids_arg=args.sample_ids, output_dir=output_dir, output_prefix=args.output_prefix)
        aggregate_summary_path = output_dir / f'{aggregate_prefix}_aggregate_summary.json'
        report_csv_path = output_dir / f'{aggregate_prefix}_report_table.csv'
        manifest['report_csv_path'] = str(report_csv_path)
        aggregate_summary_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    client = Strategy3LLMClient(config)
    schema = survival_schema()
    batch_mode = resolve_batch_mode(args)
    dataset_summaries = []
    all_records = []
    for dataset in datasets:
        run_result = run_single_dataset_batch(dataset=dataset, limit=args.limit, sample_ids_arg=args.sample_ids, output_dir=output_dir, output_prefix=args.output_prefix, client=client, schema=schema, no_cache=args.no_cache, batch_mode=batch_mode, batch_workers=args.batch_workers, batch_poll_interval=args.batch_poll_interval, batch_completion_window=args.batch_completion_window)
        dataset_summaries.append(run_result['summary'])
        all_records.extend(run_result['records'])
    aggregate_summary = build_aggregate_summary(datasets=datasets, dataset_summaries=dataset_summaries, all_records=all_records)
    aggregate_summary_path = output_dir / f'{aggregate_prefix}_aggregate_summary.json'
    report_csv_path = output_dir / f'{aggregate_prefix}_report_table.csv'
    aggregate_summary['aggregate_summary_path'] = str(aggregate_summary_path)
    aggregate_summary['report_csv_path'] = str(report_csv_path)
    aggregate_summary_path.write_text(json.dumps(aggregate_summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report_df = pd.DataFrame(build_report_rows(dataset_summaries, aggregate_summary))
    report_df.to_csv(report_csv_path, index=False)
    print(json.dumps(aggregate_summary, ensure_ascii=False, indent=2))

def run_single_dataset_batch(*, dataset: str, limit: int, sample_ids_arg: str | None, output_dir: Path, output_prefix: str | None, client: Strategy3LLMClient, schema: dict, no_cache: bool, batch_mode: str, batch_workers: int, batch_poll_interval: int, batch_completion_window: str) -> dict:
    X = load_task5_processed_dataset(dataset)
    y = load_task5_labels(dataset)
    sample_ids = resolve_sample_ids(y, limit=limit, sample_ids_arg=sample_ids_arg)
    dataset_prefix = f'{dataset}_n{len(sample_ids)}'
    if output_prefix:
        dataset_prefix = f'{output_prefix}_{dataset}_n{len(sample_ids)}'
    predictions_path = output_dir / f'{dataset_prefix}_predictions.csv'
    summary_path = output_dir / f'{dataset_prefix}_summary.json'
    raw_path = output_dir / f'{dataset_prefix}_raw.jsonl'
    records = []
    raw_path.write_text('', encoding='utf-8')
    prepared_requests = []
    for sample_id in tqdm(sample_ids, desc=f'Building {dataset}', unit='sample', leave=False):
        sample_summary = build_task5_sample_summary(X, sample_id=sample_id)
        system_prompt, user_prompt = build_task5_messages(sample_summary)
        prepared_requests.append({'sample_id': sample_id, 'sample_summary': sample_summary, 'system_prompt': system_prompt, 'user_prompt': user_prompt, 'schema': schema})
    if batch_mode != 'sync':
        results = client.batch_generate_json(requests=prepared_requests, use_cache=not no_cache, max_workers=batch_workers, batch_mode=batch_mode, poll_interval_seconds=batch_poll_interval, completion_window=batch_completion_window)
    else:
        results = []
        for request in tqdm(prepared_requests, desc=f'Running {dataset}', unit='sample'):
            result = client.generate_json(system_prompt=request['system_prompt'], user_prompt=request['user_prompt'], schema=schema, use_cache=not no_cache)
            results.append(result)
    for idx, request in enumerate(tqdm(prepared_requests, desc=f'Writing {dataset}', unit='sample'), start=1):
        sample_id = str(request['sample_id'])
        result = results[idx - 1]
        result = attach_task5_risk_decision(result, sample_summary=request['sample_summary'])
        record = build_record(dataset=dataset, sample_id=sample_id, duration=float(y.loc[sample_id, 'duration']), event=int(y.loc[sample_id, 'event']), result=result)
        records.append(record)
        with raw_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'task_name': 'task5', 'prompt_version': TASK5_PROMPT_VERSION, 'dataset': dataset, 'sample_id': sample_id, 'result': result}, ensure_ascii=False) + '\n')
        print(f"[{dataset} {idx}/{len(sample_ids)}] {sample_id} duration={record['duration']:.1f} event={record['event']} risk={record['risk_group']} score={record['risk_score']:.4f} conf={record['confidence']:.4f}")
    predictions_df = pd.DataFrame(records)
    predictions_df.to_csv(predictions_path, index=False)
    summary = build_summary(dataset=dataset, sample_ids=sample_ids, records=records, predictions_path=predictions_path, raw_path=raw_path, batch_mode=batch_mode, batch_workers=batch_workers)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'records': records, 'summary': summary}

def build_dry_run_manifest(*, datasets: list[str], limit: int, sample_ids_arg: str | None, output_dir: Path, output_prefix: str | None) -> dict:
    dataset_runs = []
    for dataset in datasets:
        y = load_task5_labels(dataset)
        sample_ids = resolve_sample_ids(y, limit=limit, sample_ids_arg=sample_ids_arg)
        dataset_prefix = f'{dataset}_n{len(sample_ids)}'
        if output_prefix:
            dataset_prefix = f'{output_prefix}_{dataset}_n{len(sample_ids)}'
        dataset_runs.append({'task_name': 'task5', 'prompt_version': TASK5_PROMPT_VERSION, 'dataset': dataset, 'sample_count': len(sample_ids), 'sample_ids': sample_ids, 'predictions_path': str(output_dir / f'{dataset_prefix}_predictions.csv'), 'summary_path': str(output_dir / f'{dataset_prefix}_summary.json'), 'raw_path': str(output_dir / f'{dataset_prefix}_raw.jsonl')})
    aggregate_prefix = output_prefix or build_multi_dataset_prefix(datasets, limit)
    return {'task_name': 'task5', 'prompt_version': TASK5_PROMPT_VERSION, 'datasets': datasets, 'dataset_runs': dataset_runs, 'aggregate_prefix': aggregate_prefix, 'aggregate_summary_path': str(output_dir / f'{aggregate_prefix}_aggregate_summary.json'), 'report_csv_path': str(output_dir / f'{aggregate_prefix}_report_table.csv')}

def resolve_datasets(dataset: str, datasets_arg: str | None) -> list[str]:
    if datasets_arg:
        return [item.strip() for item in datasets_arg.split(',') if item.strip()]
    if dataset and dataset.lower() not in {'all', 'task5_all'}:
        return [dataset]
    return ALL_TASK5_DATASETS

def build_multi_dataset_prefix(datasets: list[str], limit: int) -> str:
    if len(datasets) == 1:
        return f'{datasets[0]}_n{limit}'
    return f'task5_multi_{len(datasets)}datasets_n{limit}'

def resolve_sample_ids(labels: pd.DataFrame, limit: int, sample_ids_arg: str | None) -> list[str]:
    if sample_ids_arg:
        selected = [sample_id.strip() for sample_id in sample_ids_arg.split(',') if sample_id.strip()]
    else:
        selected = labels.index.astype(str).tolist()
    return [str(sample_id) for sample_id in selected[:limit]]

def build_record(*, dataset: str, sample_id: str, duration: float, event: int, result: dict) -> dict:
    parsed = result.get('parsed_response', {})
    derived = result.get('derived_decision', {})
    return {'dataset': dataset, 'sample_id': sample_id, 'duration': round(float(duration), 4), 'event': int(event), 'risk_group': str(derived.get('risk_group', parsed.get('risk_group', 'intermediate'))), 'risk_score': round(float(derived.get('risk_score', parsed.get('risk_score', 0.5))), 4), 'confidence': round(float(parsed.get('confidence', derived.get('confidence', 0.5))), 4), 'total_tokens': result.get('raw_response', {}).get('usage', {}).get('total_tokens'), 'rationale': parsed.get('rationale', [])}

def build_summary(*, dataset: str, sample_ids: list[str], records: list[dict], predictions_path: Path, raw_path: Path, batch_mode: str, batch_workers: int) -> dict:
    durations = [float(row['duration']) for row in records]
    events = [int(row['event']) for row in records]
    risk_scores = [float(row['risk_score']) for row in records]
    confidence_values = [float(row['confidence']) for row in records]
    token_values = [int(row['total_tokens']) for row in records if row['total_tokens'] is not None]
    risk_group_distribution: dict[str, int] = {}
    for row in records:
        risk_group_distribution[row['risk_group']] = risk_group_distribution.get(row['risk_group'], 0) + 1
    return {'task_name': 'task5', 'prompt_version': TASK5_PROMPT_VERSION, 'dataset': dataset, 'n_samples': len(records), 'sample_ids': sample_ids, 'risk_group_distribution': risk_group_distribution, 'c_index': round(concordance_index_survival(durations, risk_scores, events), 4) if records else math.nan, 'event_rate': round(sum(events) / len(events), 4) if events else math.nan, 'mean_duration': round(sum(durations) / len(durations), 4) if durations else math.nan, 'mean_risk_score': round(sum(risk_scores) / len(risk_scores), 4) if risk_scores else math.nan, 'mean_risk_score_event1': round(sum((score for score, event in zip(risk_scores, events) if event == 1)) / max(1, sum((1 for event in events if event == 1))), 4) if records else math.nan, 'mean_risk_score_event0': round(sum((score for score, event in zip(risk_scores, events) if event == 0)) / max(1, sum((1 for event in events if event == 0))), 4) if records else math.nan, 'mean_confidence': round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else math.nan, 'mean_total_tokens': round(sum(token_values) / len(token_values), 2) if token_values else None, 'batch_mode': batch_mode, 'batch_backend': batch_mode != 'sync', 'batch_workers': int(batch_workers) if batch_mode == 'concurrent' else 1, 'output_files': {'raw_jsonl': str(raw_path), 'predictions_csv': str(predictions_path)}}

def concordance_index_survival(durations: list[float], predictions: list[float], events: list[int]) -> float:
    concordant = 0.0
    permissible = 0.0
    ties = 0.0
    for i in range(len(durations)):
        for j in range(i + 1, len(durations)):
            if durations[i] == durations[j] and events[i] == 0 and (events[j] == 0):
                continue
            if durations[i] < durations[j] and events[i] == 1:
                permissible += 1
                if predictions[i] > predictions[j]:
                    concordant += 1
                elif predictions[i] == predictions[j]:
                    ties += 1
            elif durations[j] < durations[i] and events[j] == 1:
                permissible += 1
                if predictions[j] > predictions[i]:
                    concordant += 1
                elif predictions[i] == predictions[j]:
                    ties += 1
    if permissible == 0:
        return math.nan
    return (concordant + 0.5 * ties) / permissible

def build_aggregate_summary(*, datasets: list[str], dataset_summaries: list[dict], all_records: list[dict]) -> dict:
    mean_confidence = sum((row['confidence'] for row in all_records)) / len(all_records) if all_records else math.nan
    mean_total_tokens = sum((int(row['total_tokens']) for row in all_records if row['total_tokens'] is not None)) / max(1, sum((1 for row in all_records if row['total_tokens'] is not None))) if all_records else math.nan
    risk_distribution: dict[str, int] = {}
    for row in all_records:
        risk_distribution[row['risk_group']] = risk_distribution.get(row['risk_group'], 0) + 1
    valid_c_indices = [(float(summary['c_index']), int(summary['n_samples'])) for summary in dataset_summaries if summary.get('c_index') is not None and (not math.isnan(float(summary['c_index'])))]
    mean_c_index = sum((value for value, _ in valid_c_indices)) / len(valid_c_indices) if valid_c_indices else math.nan
    weighted_mean_c_index = sum((value * n for value, n in valid_c_indices)) / sum((n for _, n in valid_c_indices)) if valid_c_indices else math.nan
    events = [int(row['event']) for row in all_records]
    risk_scores = [float(row['risk_score']) for row in all_records]
    return {'task_name': 'task5', 'prompt_version': TASK5_PROMPT_VERSION, 'datasets': datasets, 'dataset_count': len(datasets), 'dataset_summaries': dataset_summaries, 'total_sample_count': len(all_records), 'risk_group_distribution': risk_distribution, 'mean_dataset_c_index': round(mean_c_index, 4) if not math.isnan(mean_c_index) else math.nan, 'weighted_mean_dataset_c_index': round(weighted_mean_c_index, 4) if not math.isnan(weighted_mean_c_index) else math.nan, 'overall_event_rate': round(sum(events) / len(events), 4) if events else math.nan, 'mean_risk_score': round(sum(risk_scores) / len(risk_scores), 4) if risk_scores else math.nan, 'mean_risk_score_event1': round(sum((score for score, event in zip(risk_scores, events) if event == 1)) / max(1, sum((1 for event in events if event == 1))), 4) if all_records else math.nan, 'mean_risk_score_event0': round(sum((score for score, event in zip(risk_scores, events) if event == 0)) / max(1, sum((1 for event in events if event == 0))), 4) if all_records else math.nan, 'mean_confidence': round(mean_confidence, 4) if not math.isnan(mean_confidence) else math.nan, 'mean_total_tokens': round(mean_total_tokens, 4) if not math.isnan(mean_total_tokens) else math.nan}

def build_report_rows(dataset_summaries: list[dict], aggregate_summary: dict) -> list[dict]:
    rows = []
    for summary in dataset_summaries:
        distribution = summary['risk_group_distribution']
        rows.append({'dataset': summary['dataset'], 'n_samples': summary['n_samples'], 'c_index': summary['c_index'], 'event_rate': summary['event_rate'], 'mean_risk_score': summary['mean_risk_score'], 'mean_risk_score_event1': summary['mean_risk_score_event1'], 'mean_risk_score_event0': summary['mean_risk_score_event0'], 'mean_confidence': summary['mean_confidence'], 'high_risk_count': distribution.get('high', 0), 'intermediate_risk_count': distribution.get('intermediate', 0), 'low_risk_count': distribution.get('low', 0)})
    aggregate_distribution = aggregate_summary['risk_group_distribution']
    rows.append({'dataset': 'ALL_DATASETS', 'n_samples': aggregate_summary['total_sample_count'], 'c_index': aggregate_summary['weighted_mean_dataset_c_index'], 'event_rate': aggregate_summary['overall_event_rate'], 'mean_risk_score': aggregate_summary['mean_risk_score'], 'mean_risk_score_event1': aggregate_summary['mean_risk_score_event1'], 'mean_risk_score_event0': aggregate_summary['mean_risk_score_event0'], 'mean_confidence': aggregate_summary['mean_confidence'], 'high_risk_count': aggregate_distribution.get('high', 0), 'intermediate_risk_count': aggregate_distribution.get('intermediate', 0), 'low_risk_count': aggregate_distribution.get('low', 0)})
    return rows
if __name__ == '__main__':
    main()
