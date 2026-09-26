from submission_paths import path as _submission_path
import argparse
import json
import math
from pathlib import Path
import sys
import pandas as pd
WORKSPACE_DIR = Path(str(_submission_path('project', '')))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
from code_strategy3 import Strategy3Config, Strategy3LLMClient, load_strategy3_runtime_config
from code_strategy3.decision import attach_task3_dual_score_decision
from code_strategy3.prompts import build_task3_messages
from code_strategy3.python_settings import DEFAULT_STRATEGY3_CONFIG
from code_strategy3.schemas import task3_dual_score_schema
from code_strategy3.tasks import build_task3_sample_summary, load_task3_processed_dataset
from code_strategy3.tasks.task3 import TASK3_DATASET_SOURCES
TASK3_OUTPUT_SUBDIR = 'task3_dual_score_batch'
ALL_TASK3_DATASETS = list(TASK3_DATASET_SOURCES.keys())

def parse_args():
    parser = argparse.ArgumentParser(description='Batch evaluation for Task 3 dual-score strategy.')
    parser.add_argument('--dataset', type=str, default=None)
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
    output_dir = Path(args.output_dir) if args.output_dir else config.results_dir / TASK3_OUTPUT_SUBDIR
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
    schema = task3_dual_score_schema()
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
    X = load_task3_processed_dataset(dataset)
    X.attrs['dataset_name'] = dataset
    y = load_task3_labels(dataset)
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
    for sample_id in sample_ids:
        sample_summary = build_task3_sample_summary(X, sample_id=sample_id)
        system_prompt, user_prompt = build_task3_messages(sample_summary)
        prepared_requests.append({'sample_id': sample_id, 'system_prompt': system_prompt, 'user_prompt': user_prompt, 'schema': schema})
    if batch_mode != 'sync':
        results = client.batch_generate_json(requests=prepared_requests, use_cache=not no_cache, max_workers=batch_workers, batch_mode=batch_mode, poll_interval_seconds=batch_poll_interval, completion_window=batch_completion_window)
    else:
        results = [client.generate_json(system_prompt=request['system_prompt'], user_prompt=request['user_prompt'], schema=schema, use_cache=not no_cache) for request in prepared_requests]
    for idx, request in enumerate(prepared_requests, start=1):
        sample_id = str(request['sample_id'])
        result = results[idx - 1]
        result = attach_task3_dual_score_decision(result)
        record = build_record(sample_id=sample_id, true_label=int(y.loc[sample_id]), result=result)
        record['dataset'] = dataset
        records.append(record)
        with raw_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'dataset': dataset, 'sample_id': sample_id, 'result': result}, ensure_ascii=False) + '\n')
        print(f"[{dataset} {idx}/{len(sample_ids)}] {sample_id} true={record['true_label']} pred={record['prediction']} margin={record['score_margin']:.4f} conf={record['confidence']:.4f}")
    predictions_df = pd.DataFrame(records)
    predictions_df.to_csv(predictions_path, index=False)
    summary = build_summary(dataset=dataset, sample_ids=sample_ids, records=records, predictions_path=predictions_path, raw_path=raw_path, batch_mode=batch_mode, batch_workers=batch_workers)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'records': records, 'summary': summary}

def build_dry_run_manifest(*, datasets: list[str], limit: int, sample_ids_arg: str | None, output_dir: Path, output_prefix: str | None) -> dict:
    dataset_runs = []
    for dataset in datasets:
        y = load_task3_labels(dataset)
        sample_ids = resolve_sample_ids(y, limit=limit, sample_ids_arg=sample_ids_arg)
        dataset_prefix = f'{dataset}_n{len(sample_ids)}'
        if output_prefix:
            dataset_prefix = f'{output_prefix}_{dataset}_n{len(sample_ids)}'
        dataset_runs.append({'dataset': dataset, 'sample_count': len(sample_ids), 'sample_ids': sample_ids, 'predictions_path': str(output_dir / f'{dataset_prefix}_predictions.csv'), 'summary_path': str(output_dir / f'{dataset_prefix}_summary.json'), 'raw_path': str(output_dir / f'{dataset_prefix}_raw.jsonl')})
    aggregate_prefix = output_prefix or build_multi_dataset_prefix(datasets, limit)
    return {'datasets': datasets, 'dataset_runs': dataset_runs, 'aggregate_prefix': aggregate_prefix, 'aggregate_summary_path': str(output_dir / f'{aggregate_prefix}_aggregate_summary.json'), 'report_csv_path': str(output_dir / f'{aggregate_prefix}_report_table.csv')}

def resolve_datasets(dataset: str, datasets_arg: str | None) -> list[str]:
    if datasets_arg:
        return [item.strip() for item in datasets_arg.split(',') if item.strip()]
    if dataset and dataset.lower() not in {'all', 'task3_all'}:
        return [dataset]
    return ALL_TASK3_DATASETS

def build_multi_dataset_prefix(datasets: list[str], limit: int) -> str:
    if len(datasets) == 1:
        return f'{datasets[0]}_n{limit}'
    return f'task3_multi_{len(datasets)}datasets_n{limit}'

def load_task3_labels(dataset: str) -> pd.Series:
    label_path = WORKSPACE_DIR / 'processed_data_strategy1' / f'{dataset}_y_final.csv'
    if not label_path.exists():
        raise FileNotFoundError(f'Task 3 label file not found: {label_path}')
    label_df = pd.read_csv(label_path, index_col=0)
    return label_df.iloc[:, 0]

def resolve_sample_ids(y: pd.Series, limit: int, sample_ids_arg: str | None) -> list[str]:
    if sample_ids_arg:
        return [sample_id.strip() for sample_id in sample_ids_arg.split(',') if sample_id.strip()]
    limit = min(limit, int(y.shape[0]))
    negative_count = limit // 2
    positive_count = limit - negative_count
    negatives = y[y == 0].head(negative_count).index.tolist()
    positives = y[y == 1].head(positive_count).index.tolist()
    selected = negatives + positives
    if len(selected) < limit:
        seen = set(selected)
        for sample_id in y.index.tolist():
            if sample_id not in seen:
                selected.append(sample_id)
                seen.add(sample_id)
            if len(selected) >= limit:
                break
    return [str(sample_id) for sample_id in selected[:limit]]

def build_record(sample_id: str, true_label: int, result: dict) -> dict:
    parsed = result['parsed_response']
    usage = result['raw_response'].get('usage', {})
    return {'sample_id': sample_id, 'true_label': true_label, 'prediction': parsed['prediction'], 'pred_binary': 1 if parsed['prediction'] == 'positive' else 0, 'confidence': float(parsed['confidence']), 'response_support_score': float(parsed['response_support_score']), 'resistance_support_score': float(parsed['resistance_support_score']), 'score_margin': float(parsed['score_margin']), 'dominant_side': parsed['dominant_side'], 'prompt_tokens': usage.get('prompt_tokens'), 'completion_tokens': usage.get('completion_tokens'), 'total_tokens': usage.get('total_tokens'), 'rationale': ' | '.join(parsed.get('rationale', []))}

def build_summary(dataset: str, sample_ids: list[str], records: list[dict], predictions_path: Path, raw_path: Path, batch_mode: str, batch_workers: int) -> dict:
    tp = sum((1 for row in records if row['true_label'] == 1 and row['pred_binary'] == 1))
    tn = sum((1 for row in records if row['true_label'] == 0 and row['pred_binary'] == 0))
    fp = sum((1 for row in records if row['true_label'] == 0 and row['pred_binary'] == 1))
    fn = sum((1 for row in records if row['true_label'] == 1 and row['pred_binary'] == 0))
    accuracy = (tp + tn) / len(records) if records else math.nan
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_confidence = sum((row['confidence'] for row in records)) / len(records) if records else math.nan
    mean_total_tokens = sum((row['total_tokens'] for row in records if row['total_tokens'] is not None)) / max(1, sum((1 for row in records if row['total_tokens'] is not None)))
    prediction_distribution = {}
    for row in records:
        prediction_distribution[row['prediction']] = prediction_distribution.get(row['prediction'], 0) + 1
    return {'dataset': dataset, 'sample_count': len(sample_ids), 'sample_ids': sample_ids, 'prediction_distribution': prediction_distribution, 'accuracy': round(accuracy, 4), 'precision': round(precision, 4), 'recall': round(recall, 4), 'specificity': round(specificity, 4), 'f1': round(f1, 4), 'mean_confidence': round(mean_confidence, 4), 'mean_total_tokens': round(mean_total_tokens, 4), 'predictions_path': str(predictions_path), 'raw_path': str(raw_path), 'batch_mode': batch_mode, 'batch_backend': batch_mode != 'sync', 'batch_workers': int(batch_workers) if batch_mode == 'concurrent' else 1}

def build_aggregate_summary(*, datasets: list[str], dataset_summaries: list[dict], all_records: list[dict]) -> dict:
    mean_confidence = sum((row['confidence'] for row in all_records)) / len(all_records) if all_records else math.nan
    mean_total_tokens = sum((row['total_tokens'] for row in all_records if row['total_tokens'] is not None)) / max(1, sum((1 for row in all_records if row['total_tokens'] is not None)))
    prediction_distribution = {}
    for row in all_records:
        prediction_distribution[row['prediction']] = prediction_distribution.get(row['prediction'], 0) + 1
    overall_accuracy = sum((int(row['true_label'] == row['pred_binary']) for row in all_records)) / len(all_records) if all_records else math.nan
    return {'datasets': datasets, 'dataset_count': len(datasets), 'dataset_summaries': dataset_summaries, 'total_sample_count': len(all_records), 'prediction_distribution': prediction_distribution, 'overall_accuracy': round(overall_accuracy, 4), 'mean_confidence': round(mean_confidence, 4), 'mean_total_tokens': round(mean_total_tokens, 4)}

def build_report_rows(dataset_summaries: list[dict], aggregate_summary: dict) -> list[dict]:
    rows = []
    for summary in dataset_summaries:
        distribution = summary.get('prediction_distribution', {})
        rows.append({'dataset': summary['dataset'], 'sample_count': summary['sample_count'], 'positive_predictions': distribution.get('positive', 0), 'negative_predictions': distribution.get('negative', 0), 'accuracy': summary['accuracy'], 'precision': summary['precision'], 'recall': summary['recall'], 'specificity': summary['specificity'], 'f1': summary['f1'], 'mean_confidence': summary['mean_confidence'], 'mean_total_tokens': summary['mean_total_tokens'], 'predictions_path': summary['predictions_path']})
    aggregate_distribution = aggregate_summary.get('prediction_distribution', {})
    rows.append({'dataset': 'ALL_DATASETS', 'sample_count': aggregate_summary['total_sample_count'], 'positive_predictions': aggregate_distribution.get('positive', 0), 'negative_predictions': aggregate_distribution.get('negative', 0), 'accuracy': aggregate_summary['overall_accuracy'], 'precision': '', 'recall': '', 'specificity': '', 'f1': '', 'mean_confidence': aggregate_summary['mean_confidence'], 'mean_total_tokens': aggregate_summary['mean_total_tokens'], 'predictions_path': ''})
    return rows
if __name__ == '__main__':
    main()
