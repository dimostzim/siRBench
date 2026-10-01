"""Run matched GNN extent or original-schedule checks using frozen main wrappers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pandas as pd

from run_gnn_matrix import verify_predictions
from subset_gnn_features import subset_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--series',choices=['extent','original_schedule'],required=True)
    parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--prepared-standardized',type=Path,required=True)
    parser.add_argument('--prepared-original',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    args.output.resolve().relative_to(repo)
    competitors = repo/'benchmark/competitors'
    args.output.mkdir(parents=True,exist_ok=True)
    records = []
    variants = ['standardized','original'] if args.series == 'extent' else ['original_schedule']
    for variant in variants:
        folder = args.cohort/variant/'fold_0' if args.series == 'extent' else args.cohort/'grouped/fold_0'
        inputs = {part:folder/f'{part}.csv' for part in ['train','val','test']}
        inputs['leftout'] = folder/'hela_full.csv' if args.series == 'extent' else args.cohort/'hela_full.csv'
        source = args.prepared_original if variant == 'original' else args.prepared_standardized
        if source is None or not (source/'all.csv').is_file():
            raise ValueError('Canonical prepared source unavailable')
        original_schedule = args.series == 'original_schedule'
        for seed in [0,1,2]:
            name = f'{variant}-fold0-seed{seed}'
            run = args.output/name
            initialization = ['python3',str(competitors/'scripts/prepare_run.py'),
                '--repo-root',str(repo),'--run-dir',str(run),'--tool','gnn4sirna',
                '--seed',str(seed),'--original',str(int(original_schedule)),'--deterministic','0']
            for part,path in inputs.items(): initialization.extend(['--'+part,str(path)])
            subprocess.run(initialization,check=True)
            data = run/'data/gnn4sirna'
            marker = data/'subsets_completed.json'
            if not marker.exists():
                partitions = {part:pd.read_csv(path) for part,path in inputs.items()}
                partitions['trainval'] = pd.concat([partitions['train'],partitions['val']],ignore_index=True)
                for part,frame in partitions.items(): subset_features(source,frame,data,part)
                partitions['trainval'].to_csv(data/'trainval_input.csv',index=False)
                manifests = sorted(data.glob('*_feature_manifest.json'))
                marker.write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in manifests},indent=2)+'\n')
            for manifest in data.glob('*_feature_manifest.json'):
                contents = json.loads(manifest.read_text())
                if str(source/'all.csv') not in contents['source_sha256']:
                    raise ValueError('Canonical feature source changed')
                for filename,digest in contents['source_sha256'].items():
                    if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                        raise ValueError('Canonical feature contents changed')
                for filename,digest in contents['output_sha256'].items():
                    if hashlib.sha256((data/filename).read_bytes()).hexdigest() != digest:
                        raise ValueError('Prepared feature changed after publication')
            model = run/'models/gnn4sirna/model.keras'
            metadata = model.parent/'train_meta.json'
            if model.exists() and not metadata.exists():
                raise ValueError('Partial training checkpoint; refuse silent resume')
            command = ['bash',str(competitors/'run_tool.sh'),'--tool','gnn4sirna',
                       '--run-dir',str(run),'--seed',str(seed)]
            for part,path in inputs.items(): command.extend(['--'+part,str(path)])
            if original_schedule: command.append('--original')
            with (args.output/f'{name}.log').open('a') as log:
                subprocess.run(command,cwd=competitors,env={**os.environ,'QUIET':'0'},stdout=log,stderr=subprocess.STDOUT,check=True)
            result = run/'results/gnn4sirna'
            verify_predictions(inputs['test'],result/'preds.csv')
            verify_predictions(inputs['leftout'],result/'preds_leftout.csv')
            config = json.loads(metadata.read_text())['configuration']
            if config['epochs'] != (10 if original_schedule else 100) or config['transductive']:
                raise ValueError('Unexpected training policy')
            records.append({'tool':'gnn4sirna','series':args.series,'variant':variant,
                'axis':'grouped','fold':0,'training_seed':seed,'run_dir':str(run),
                'test_predictions':str(result/'preds.csv'),'hela_predictions':str(result/'preds_leftout.csv'),
                'train_meta':str(metadata),'status':'verified',
                'prediction_sha256':hashlib.sha256((result/'preds.csv').read_bytes()).hexdigest(),
                'hela_sha256':hashlib.sha256((result/'preds_leftout.csv').read_bytes()).hexdigest()})
            pd.DataFrame(records).to_csv(args.output/'run_index.csv',index=False)
            print(name,'verified',flush=True)
    (args.output/'completed.json').write_text(json.dumps({'verified_runs':len(records)})+'\n')


if __name__ == '__main__':
    main()
