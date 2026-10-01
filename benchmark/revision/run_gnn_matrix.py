"""Execute frozen GNN partitions and verify prediction identity after every run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


def verify_predictions(source, predictions):
    expected = pd.read_csv(source).set_index('record_id')
    actual = pd.read_csv(predictions).set_index('id')
    if not actual.index.is_unique or set(actual.index) != set(expected.index):
        raise ValueError(f'Prediction IDs do not match: {predictions}')
    actual = actual.loc[expected.index]
    if not np.allclose(actual.label, expected.efficiency, rtol=0, atol=1e-12):
        raise ValueError(f'Prediction labels do not match: {predictions}')
    if not np.isfinite(actual.pred_label).all():
        raise ValueError(f'Nonfinite predictions: {predictions}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    competitors = repo / 'benchmark/competitors'
    args.output.resolve().relative_to(repo)
    args.output.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(args.protocol / 'run_matrix.csv')
    records = []
    environment = {**os.environ, 'QUIET':'0', 'SIRBENCH_REPO_ROOT':str(repo)}
    for row in matrix.itertuples(index=False):
        name = f'{row.axis}-fold{row.fold}-seed{row.training_seed}'
        run = args.output / name
        log = args.output / f'{name}.log'
        command = ['bash',str(competitors/'run_tool.sh'),'--tool','gnn4sirna',
                   '--train',row.train,'--val',row.val,'--test',row.test,
                   '--leftout',row.hela_full,'--seed',str(row.training_seed),'--run-dir',str(run)]
        with log.open('a') as stream:
            subprocess.run(command, cwd=competitors, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        predictions = run/'results/gnn4sirna/preds.csv'
        hela = run/'results/gnn4sirna/preds_leftout.csv'
        verify_predictions(row.test, predictions)
        verify_predictions(row.hela_full, hela)
        metadata = run/'models/gnn4sirna/train_meta.json'
        records.append({'tool':'gnn4sirna','axis':row.axis,'fold':row.fold,'training_seed':row.training_seed,
                        'run_dir':str(run),'test_predictions':str(predictions),'hela_predictions':str(hela),
                        'train_meta':str(metadata),'status':'verified',
                        'prediction_sha256':hashlib.sha256(predictions.read_bytes()).hexdigest(),
                        'hela_sha256':hashlib.sha256(hela.read_bytes()).hexdigest()})
        pd.DataFrame(records).to_csv(args.output/'run_index.csv',index=False)
        print(name,'verified',flush=True)
    (args.output/'completed.json').write_text(json.dumps({'runs':len(records),'run_matrix_sha256':hashlib.sha256((args.protocol/'run_matrix.csv').read_bytes()).hexdigest()},indent=2)+'\n')


if __name__ == '__main__':
    main()
