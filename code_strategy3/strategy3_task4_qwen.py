from submission_paths import path as _submission_path
import argparse
import json
from pathlib import Path
import sys
WORKSPACE_DIR = Path(str(_submission_path('project', '')))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
from code_strategy3 import Strategy3Config, Strategy3LLMClient, load_strategy3_runtime_config
from code_strategy3.decision import TASK4_POSITIVE_MARGIN_THRESHOLD, attach_task4_llm_decision, attach_task4_rule_decision
from code_strategy3.prompts.task4 import build_task4_messages
from code_strategy3.python_settings import DEFAULT_STRATEGY3_CONFIG
from code_strategy3.schemas import task4_dual_score_schema
from code_strategy3.tasks.task4 import build_task4_sample_summary, compute_task4_v3_rule_profiles, load_task4_processed_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Task 4 Qwen implementation for Strategy 3.')
    parser.add_argument('--sample_id', type=str, default=None)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--output_file', type=str, default='./out.txt')
    parser.add_argument('--use_env', action='store_true')
    parser.add_argument('--profile', type=str, default=None)
    parser.add_argument('--execution_mode', type=str, default='rule_only', choices=['rule_only', 'llm', 'hybrid'])
    parser.add_argument('--margin_threshold', type=float, default=TASK4_POSITIVE_MARGIN_THRESHOLD)
    return parser.parse_args()

def load_runtime_config(use_env: bool, profile_name: str | None=None) -> Strategy3Config:
    if profile_name:
        return load_strategy3_runtime_config(profile_name=profile_name, use_env=use_env)
    if use_env:
        return Strategy3Config.from_env()
    return Strategy3Config.from_mapping(DEFAULT_STRATEGY3_CONFIG)

def attach_task4_dual_score_decision(rule_profile: dict, margin_threshold: float=TASK4_POSITIVE_MARGIN_THRESHOLD) -> dict:
    return attach_task4_rule_decision(rule_profile, margin_threshold=margin_threshold)

def main():
    args = parse_args()
    df = load_task4_processed_dataset()
    sample_summary = build_task4_sample_summary(df, sample_id=args.sample_id)
    all_sample_summaries = [build_task4_sample_summary(df, sample_id=str(sample_id)) for sample_id in df.index.tolist()]
    rule_profiles = compute_task4_v3_rule_profiles(all_sample_summaries)
    rule_profile = rule_profiles[str(sample_summary['sample_id'])]
    system_prompt, user_prompt = build_task4_messages(sample_summary)
    schema = task4_dual_score_schema()
    payload = {'system_prompt': system_prompt, 'user_prompt': user_prompt, 'schema': schema, 'sample_summary': sample_summary, 'rule_profile': rule_profile}
    if args.dry_run:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output_file:
            Path(args.output_file).write_text(text + '\n', encoding='utf-8')
        else:
            print(text)
        return
    if args.execution_mode == 'rule_only':
        result = attach_task4_rule_decision(rule_profile, margin_threshold=args.margin_threshold)
    else:
        config = load_runtime_config(args.use_env, args.profile)
        client = Strategy3LLMClient(config)
        try:
            llm_result = client.generate_json(system_prompt=system_prompt, user_prompt=user_prompt, schema=schema)
            result = attach_task4_llm_decision(llm_result, margin_threshold=args.margin_threshold)
            result['mode'] = 'llm'
        except Exception:
            if args.execution_mode != 'hybrid':
                raise
            result = attach_task4_rule_decision(rule_profile, margin_threshold=args.margin_threshold)
            result['mode'] = 'hybrid_rule_fallback'
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_file:
        Path(args.output_file).write_text(text + '\n', encoding='utf-8')
    else:
        print(text)
if __name__ == '__main__':
    main()
