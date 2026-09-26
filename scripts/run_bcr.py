"""Run the current BCR implementation in a fresh parameter-selected output directory."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from submission_paths import PACKAGE_ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--mode',choices=['smoke','fixed','repeats','all'],default='smoke')
    args=parser.parse_args()
    if not 1<=args.workers<=8: parser.error('Use 1--8 workers, each fit may use 8 CPU threads.')
    dest=args.output_dir.expanduser().resolve()
    if dest.exists(): parser.error('--output-dir must be new to protect existing results.')
    base=PACKAGE_ROOT/'analysis/bcr'
    dest.mkdir(parents=True)
    for name in ['repair_core.py','run_repair.py','bcr_inputs.py','task4_grouped_core.py','test_repair.py','protocol.md']:
        shutil.copy2(base/name,dest/name)
    modes=['smoke','fixed','repeats'] if args.mode=='all' else [args.mode]
    for mode in modes:
        subprocess.run([sys.executable,str(dest/'run_repair.py'),'--mode',mode,'--workers',str(args.workers)],check=True)
    if 'repeats' in modes:
        import pandas as pd
        blocks=[]
        for seed in range(42,62):
            frame=pd.read_csv(dest/f'repeats/checkpoints/all_samples_case_post_ici_incident__{seed}/sample_predictions.csv')
            blocks.append(frame[frame.model_key.eq('strategy1|BCR_3mer|XGBoost')])
        pd.concat(blocks,ignore_index=True).to_csv(dest/'bcr_case_input.csv',index=False)
    print('Completed current BCR analysis; no LLM calls or new embeddings.')


if __name__=='__main__':main()
