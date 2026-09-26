from submission_paths import path as _submission_path
import argparse
import csv
import json
from pathlib import Path
import sys
WORKSPACE_DIR = Path(str(_submission_path('project', '')))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
from code_strategy3 import Strategy3Config, Strategy3LLMClient, load_strategy3_runtime_config
from code_strategy3.decision import TASK1_POSITIVE_MARGIN_THRESHOLD, attach_task1_dual_score_decision
from code_strategy3.prompts import build_task1_messages
from code_strategy3.python_settings import DEFAULT_STRATEGY3_CONFIG
from code_strategy3.schemas import task1_dual_score_schema
from code_strategy3.tasks import build_task1_sample_summary, load_task1_labels, load_task1_processed_dataset
TASK1_OUTPUT_SUBDIR = 'task1_batch_eval'

def parse_args():
    parser = argparse.ArgumentParser(description='Batch evaluation for Strategy 3 Task 1.')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--use_env', action='store_true')
    parser.add_argument('--profile', type=str, default=None)
    parser.add_argument('--margin_threshold', type=float, default=TASK1_POSITIVE_MARGIN_THRESHOLD)
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

def label_to_name(value: int) -> str:
    return 'positive' if int(value) == 1 else 'negative'

def compute_metrics(rows: list[dict]) -> dict:
    tp = sum((1 for row in rows if row['true_label'] == 1 and row['pred_label'] == 1))
    tn = sum((1 for row in rows if row['true_label'] == 0 and row['pred_label'] == 0))
    fp = sum((1 for row in rows if row['true_label'] == 0 and row['pred_label'] == 1))
    fn = sum((1 for row in rows if row['true_label'] == 1 and row['pred_label'] == 0))
    total = len(rows)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'n_samples': total, 'accuracy': round(accuracy, 4), 'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

def main():
    args = parse_args()
    config = load_runtime_config(args.use_env, args.profile)
    output_dir = Path(args.output_dir) if args.output_dir else config.results_dir / TASK1_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_task1_processed_dataset()
    labels = load_task1_labels()
    sample_ids = [sample_id for sample_id in labels.index.tolist() if sample_id in df.index]
    if args.max_samples is not None:
        sample_ids = sample_ids[:args.max_samples]
    schema = task1_dual_score_schema()
    rows = []
    raw_output_path = output_dir / f'task1_n{len(sample_ids)}_raw.jsonl'
    prediction_path = output_dir / f'task1_n{len(sample_ids)}_predictions.csv'
    summary_path = output_dir / f'task1_n{len(sample_ids)}_summary.json'
    if args.dry_run:
        sample_summary = build_task1_sample_summary(df, sample_id=sample_ids[0])
        system_prompt, user_prompt = build_task1_messages(sample_summary)
        payload = {'system_prompt': system_prompt, 'user_prompt': user_prompt, 'schema': schema, 'sample_summary': sample_summary}
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return
    client = Strategy3LLMClient(config)
    batch_mode = resolve_batch_mode(args)
    prepared_requests = []
    for sample_id in sample_ids:
        sample_summary = build_task1_sample_summary(df, sample_id=sample_id)
        system_prompt, user_prompt = build_task1_messages(sample_summary)
        prepared_requests.append({'sample_id': sample_id, 'system_prompt': system_prompt, 'user_prompt': user_prompt, 'schema': schema})
    if batch_mode != 'sync':
        results = client.batch_generate_json(requests=prepared_requests, use_cache=True, max_workers=args.batch_workers, batch_mode=batch_mode, poll_interval_seconds=args.batch_poll_interval, completion_window=args.batch_completion_window)
    else:
        results = [client.generate_json(system_prompt=request['system_prompt'], user_prompt=request['user_prompt'], schema=schema) for request in prepared_requests]
    with raw_output_path.open('w', encoding='utf-8') as raw_handle:
        for idx, request in enumerate(prepared_requests, start=1):
            sample_id = str(request['sample_id'])
            result = results[idx - 1]
            result = attach_task1_dual_score_decision(result, margin_threshold=args.margin_threshold)
            raw_handle.write(json.dumps({'task_name': 'task1', 'sample_id': sample_id, 'result': result}, ensure_ascii=False) + '\n')
            parsed = result.get('parsed_response', {})
            derived = result.get('derived_decision', {})
            prediction = str(derived.get('prediction', parsed.get('prediction', 'negative'))).lower()
            pred_label = 1 if prediction == 'positive' else 0
            confidence = float(derived.get('confidence', parsed.get('confidence', 0.5)))
            rows.append({'sample_id': sample_id, 'true_label': int(labels.loc[sample_id]), 'pred_label': pred_label, 'true_name': label_to_name(int(labels.loc[sample_id])), 'pred_name': prediction, 'confidence': round(confidence, 4), 'response_support_score': round(float(parsed.get('response_support_score', 0.5)), 4), 'resistance_support_score': round(float(parsed.get('resistance_support_score', 0.5)), 4), 'score_margin': round(float(derived.get('score_margin', 0.0)), 4), 'rationale': parsed.get('rationale', []), 'total_tokens': result.get('raw_response', {}).get('usage', {}).get('total_tokens')})
    with prediction_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['sample_id', 'true_label', 'pred_label', 'true_name', 'pred_name', 'confidence', 'response_support_score', 'resistance_support_score', 'score_margin', 'total_tokens', 'rationale'])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'rationale': json.dumps(row['rationale'], ensure_ascii=False)})
    metrics = compute_metrics(rows)
    confidence_values = [float(row['confidence']) for row in rows]
    token_values = [int(row['total_tokens']) for row in rows if row['total_tokens'] is not None]
    summary = {'task_name': 'task1', 'n_samples': len(rows), 'margin_threshold': round(args.margin_threshold, 4), 'metrics': metrics, 'mean_confidence': round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else None, 'mean_score_margin': round(sum((float(row['score_margin']) for row in rows)) / len(rows), 4) if rows else None, 'mean_total_tokens': round(sum(token_values) / len(token_values), 2) if token_values else None, 'batch_mode': batch_mode, 'batch_backend': batch_mode != 'sync', 'batch_workers': int(args.batch_workers) if batch_mode == 'concurrent' else 1, 'output_files': {'raw_jsonl': str(raw_output_path), 'predictions_csv': str(prediction_path), 'summary_json': str(summary_path)}}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
if __name__ == '__main__':
    main()
