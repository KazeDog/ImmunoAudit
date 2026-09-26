from submission_paths import path as _submission_path
import argparse
import json
from pathlib import Path
import sys
WORKSPACE_DIR = Path(str(_submission_path('project', '')))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
from code_strategy3 import Strategy3Config, Strategy3LLMClient, load_strategy3_runtime_config
from code_strategy3.decision import attach_task5_risk_decision
from code_strategy3.prompts import build_task5_messages
from code_strategy3.python_settings import DEFAULT_STRATEGY3_CONFIG
from code_strategy3.schemas import survival_schema
from code_strategy3.tasks import build_task5_sample_summary, load_task5_processed_dataset
from code_strategy3.tasks.task5 import DEFAULT_TASK5_DATASET

def parse_args():
    parser = argparse.ArgumentParser(description='Task 5 Qwen API skeleton for Strategy 3.')
    parser.add_argument('--dataset', type=str, default=DEFAULT_TASK5_DATASET)
    parser.add_argument('--sample_id', type=str, default=None)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--output_file', type=str, default='./out.txt')
    parser.add_argument('--use_env', action='store_true')
    parser.add_argument('--profile', type=str, default=None)
    return parser.parse_args()

def load_runtime_config(use_env: bool, profile_name: str | None=None) -> Strategy3Config:
    if profile_name:
        return load_strategy3_runtime_config(profile_name=profile_name, use_env=use_env)
    if use_env:
        return Strategy3Config.from_env()
    return Strategy3Config.from_mapping(DEFAULT_STRATEGY3_CONFIG)

def main():
    args = parse_args()
    df = load_task5_processed_dataset(args.dataset)
    sample_summary = build_task5_sample_summary(df, sample_id=args.sample_id)
    system_prompt, user_prompt = build_task5_messages(sample_summary)
    schema = survival_schema()
    payload = {'system_prompt': system_prompt, 'user_prompt': user_prompt, 'schema': schema, 'sample_summary': sample_summary}
    if args.dry_run:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output_file:
            Path(args.output_file).write_text(text + '\n', encoding='utf-8')
        else:
            print(text)
        return
    config = load_runtime_config(args.use_env, args.profile)
    client = Strategy3LLMClient(config)
    result = client.generate_json(system_prompt=system_prompt, user_prompt=user_prompt, schema=schema)
    result = attach_task5_risk_decision(result, sample_summary=sample_summary)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_file:
        Path(args.output_file).write_text(text + '\n', encoding='utf-8')
    else:
        print(text)
if __name__ == '__main__':
    main()
