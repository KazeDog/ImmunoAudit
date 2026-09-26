from submission_paths import path as _submission_path
import argparse
import csv
import json
from pathlib import Path
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
WORKSPACE_DIR = Path(str(_submission_path('project', '')))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
from code_strategy3 import Strategy3Config, Strategy3LLMClient, load_strategy3_runtime_config
from code_strategy3.decision import TASK4_POSITIVE_MARGIN_THRESHOLD, attach_task4_llm_decision, attach_task4_rule_decision
from code_strategy3.prompts.task4 import TASK4_PROMPT_VERSION, build_task4_messages
from code_strategy3.python_settings import DEFAULT_STRATEGY3_CONFIG
from code_strategy3.schemas import task4_dual_score_schema
from code_strategy3.tasks.task4 import build_task4_sample_summary, compute_task4_v3_rule_profiles, load_task4_labels, load_task4_processed_dataset
TASK4_OUTPUT_SUBDIR = 'task4_dual_score_batch'

def parse_args():
    parser = argparse.ArgumentParser(description='Batch evaluation for Strategy 3 Task 4.')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--use_env', action='store_true')
    parser.add_argument('--profile', type=str, default=None)
    parser.add_argument('--execution_mode', type=str, default='rule_only', choices=['rule_only', 'llm', 'hybrid'])
    parser.add_argument('--margin_threshold', type=float, default=TASK4_POSITIVE_MARGIN_THRESHOLD)
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
    if not rows:
        return {}
    y_true = [int(row['true_label']) for row in rows]
    y_pred = [int(row['pred_label']) for row in rows]
    y_score = [float(row['fused_toxicity_index']) for row in rows]
    try:
        auroc = round(float(roc_auc_score(y_true, y_score)), 4)
    except ValueError:
        auroc = None
    try:
        auprc = round(float(average_precision_score(y_true, y_score)), 4)
    except ValueError:
        auprc = None
    return {'n_samples': len(rows), 'accuracy': round(float(accuracy_score(y_true, y_pred)), 4), 'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4), 'recall': round(float(recall_score(y_true, y_pred, zero_division=0)), 4), 'f1': round(float(f1_score(y_true, y_pred, zero_division=0)), 4), 'auroc': auroc, 'auprc': auprc, 'tp': sum((1 for row in rows if row['true_label'] == 1 and row['pred_label'] == 1)), 'tn': sum((1 for row in rows if row['true_label'] == 0 and row['pred_label'] == 0)), 'fp': sum((1 for row in rows if row['true_label'] == 0 and row['pred_label'] == 1)), 'fn': sum((1 for row in rows if row['true_label'] == 1 and row['pred_label'] == 0))}

def main():
    args = parse_args()
    config = load_runtime_config(args.use_env, args.profile)
    output_dir = Path(args.output_dir) if args.output_dir else config.results_dir / TASK4_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_task4_processed_dataset()
    labels = load_task4_labels()
    sample_ids = [sample_id for sample_id in labels.index.tolist() if sample_id in df.index]
    if args.max_samples is not None:
        positives = [sample_id for sample_id in sample_ids if int(labels.loc[sample_id]) == 1]
        negatives = [sample_id for sample_id in sample_ids if int(labels.loc[sample_id]) == 0]
        interleaved = []
        max_len = max(len(positives), len(negatives))
        for idx in range(max_len):
            if idx < len(positives):
                interleaved.append(positives[idx])
            if idx < len(negatives):
                interleaved.append(negatives[idx])
        sample_ids = interleaved[:args.max_samples]
    schema = task4_dual_score_schema()
    rows = []
    raw_output_path = output_dir / f'task4_n{len(sample_ids)}_raw.jsonl'
    prediction_path = output_dir / f'task4_n{len(sample_ids)}_predictions.csv'
    summary_path = output_dir / f'task4_n{len(sample_ids)}_summary.json'
    if args.dry_run:
        sample_summary = build_task4_sample_summary(df, sample_id=sample_ids[0])
        all_sample_summaries = [build_task4_sample_summary(df, sample_id=str(sample_id)) for sample_id in sample_ids]
        rule_profiles = compute_task4_v3_rule_profiles(all_sample_summaries)
        system_prompt, user_prompt = build_task4_messages(sample_summary)
        payload = {'execution_mode': args.execution_mode, 'system_prompt': system_prompt, 'user_prompt': user_prompt, 'schema': schema, 'sample_summary': sample_summary, 'rule_profile': rule_profiles[str(sample_summary['sample_id'])]}
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return
    batch_mode = resolve_batch_mode(args)
    sample_summaries = []
    for sample_id in sample_ids:
        sample_summary = build_task4_sample_summary(df, sample_id=sample_id)
        sample_summaries.append(sample_summary)
    rule_profiles = compute_task4_v3_rule_profiles(sample_summaries)
    llm_results_by_sample_id: dict[str, dict] = {}
    effective_batch_mode = batch_mode
    if args.execution_mode != 'rule_only':
        client = Strategy3LLMClient(config)
        prepared_requests = []
        for sample_summary in sample_summaries:
            system_prompt, user_prompt = build_task4_messages(sample_summary)
            prepared_requests.append({'sample_id': str(sample_summary['sample_id']), 'system_prompt': system_prompt, 'user_prompt': user_prompt, 'schema': schema})
        if args.execution_mode == 'llm' and batch_mode != 'sync':
            llm_results = client.batch_generate_json(requests=prepared_requests, use_cache=True, max_workers=args.batch_workers, batch_mode=batch_mode, poll_interval_seconds=args.batch_poll_interval, completion_window=args.batch_completion_window)
            llm_results_by_sample_id = {str(request['sample_id']): result for request, result in zip(prepared_requests, llm_results, strict=False)}
        else:
            if args.execution_mode == 'hybrid':
                effective_batch_mode = 'sync'
            for request in prepared_requests:
                sample_id = str(request['sample_id'])
                try:
                    llm_result = client.generate_json(system_prompt=request['system_prompt'], user_prompt=request['user_prompt'], schema=schema)
                    llm_results_by_sample_id[sample_id] = llm_result
                except Exception:
                    if args.execution_mode != 'hybrid':
                        raise
    with raw_output_path.open('w', encoding='utf-8') as raw_handle:
        for sample_summary in sample_summaries:
            sample_id = str(sample_summary['sample_id'])
            rule_profile = rule_profiles[sample_id]
            llm_result = llm_results_by_sample_id.get(sample_id)
            if llm_result is not None:
                result = attach_task4_llm_decision(llm_result, margin_threshold=args.margin_threshold)
                result['mode'] = 'llm'
            else:
                result = attach_task4_rule_decision(rule_profile, margin_threshold=args.margin_threshold)
                if args.execution_mode == 'hybrid':
                    result['mode'] = 'hybrid_rule_fallback'
            raw_handle.write(json.dumps({'task_name': 'task4', 'sample_id': sample_id, 'result': result}, ensure_ascii=False) + '\n')
            parsed = result.get('parsed_response', {})
            derived = result.get('derived_decision', {})
            prediction = str(derived.get('prediction', parsed.get('prediction', 'negative'))).lower()
            pred_label = 1 if prediction == 'positive' else 0
            confidence = float(derived.get('confidence', parsed.get('confidence', 0.5)))
            rows.append({'sample_id': sample_id, 'true_label': int(labels.loc[sample_id]), 'pred_label': pred_label, 'true_name': label_to_name(int(labels.loc[sample_id])), 'pred_name': prediction, 'confidence': round(confidence, 4), 'toxicity_support_score': round(float(parsed.get('toxicity_support_score', 0.5)), 4), 'tolerance_support_score': round(float(parsed.get('tolerance_support_score', 0.5)), 4), 'score_margin': round(float(derived.get('score_margin', 0.0)), 4), 'calibrated_margin': round(float(derived.get('calibrated_margin', 0.0)), 4), 'summary_toxicity_index': round(float(derived.get('summary_toxicity_index', 0.0)), 4), 'fused_toxicity_index': round(float(derived.get('fused_toxicity_index', 0.0)), 4), 'dominant_side': str(derived.get('dominant_side', parsed.get('dominant_side', 'mixed'))), 'rationale': parsed.get('rationale', []), 'total_tokens': result.get('raw_response', {}).get('usage', {}).get('total_tokens')})
    with prediction_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['sample_id', 'true_label', 'pred_label', 'true_name', 'pred_name', 'confidence', 'toxicity_support_score', 'tolerance_support_score', 'score_margin', 'calibrated_margin', 'summary_toxicity_index', 'fused_toxicity_index', 'dominant_side', 'total_tokens', 'rationale'])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'rationale': json.dumps(row['rationale'], ensure_ascii=False)})
    metrics = compute_metrics(rows)
    confidence_values = [float(row['confidence']) for row in rows]
    token_values = [int(row['total_tokens']) for row in rows if row['total_tokens'] is not None]
    summary = {'task_name': 'task4', 'prompt_version': TASK4_PROMPT_VERSION, 'n_samples': len(rows), 'margin_threshold': round(args.margin_threshold, 4), 'metrics': metrics, 'positive_rate': round(float(pd.Series([row['true_label'] for row in rows]).mean()), 4) if rows else None, 'predicted_positive_rate': round(float(pd.Series([row['pred_label'] for row in rows]).mean()), 4) if rows else None, 'mean_confidence': round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else None, 'mean_score_margin': round(sum((float(row['score_margin']) for row in rows)) / len(rows), 4) if rows else None, 'mean_calibrated_margin': round(sum((float(row['calibrated_margin']) for row in rows)) / len(rows), 4) if rows else None, 'mean_summary_toxicity_index': round(sum((float(row['summary_toxicity_index']) for row in rows)) / len(rows), 4) if rows else None, 'mean_fused_toxicity_index': round(sum((float(row['fused_toxicity_index']) for row in rows)) / len(rows), 4) if rows else None, 'mean_total_tokens': round(sum(token_values) / len(token_values), 2) if token_values else None, 'execution_mode': args.execution_mode, 'batch_mode': effective_batch_mode if args.execution_mode != 'rule_only' else 'rule_only', 'batch_backend': args.execution_mode != 'rule_only' and effective_batch_mode != 'sync', 'batch_workers': int(args.batch_workers) if effective_batch_mode == 'concurrent' else 1, 'output_files': {'raw_jsonl': str(raw_output_path), 'predictions_csv': str(prediction_path)}}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
if __name__ == '__main__':
    main()
