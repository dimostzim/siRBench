#!/usr/bin/env python3
"""Run one explicit sensitivity specification without changing frozen model wrappers."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

from run_attsioff_discovery_matrix import check_model_completion, digest


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def execute_case(repo, specification):
    tool = specification['tool']
    run = repo / specification['run_relative']
    assets = repo / specification['assets_relative']
    data = run / 'data' / tool
    model = run / 'models' / tool
    results = run / 'results' / tool
    scripts = repo / 'benchmark/competitors/scripts'
    check_model_completion(run, tool, specification['seed'])
    completion = run / 'SENSITIVITY_COMPLETE.json'
    if completion.exists():
        if json.loads((run / 'sensitivity_specification.json').read_text()) != specification:
            raise ValueError('Completed sensitivity specification differs')
        for filename, expected in json.loads(completion.read_text()).items():
            if digest(run / filename) != expected:
                raise ValueError(f'Completed sensitivity checksum differs: {filename}')
        return
    if model.exists() and any(model.iterdir()):
        raise ValueError('Sensitivity requires a fresh model directory; inspect existing artifacts before retry')
    data.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    asset_rows = {row['id']: row for row in read_rows(assets / 'all.csv')}
    asset_inputs = {row['record_id']: row for row in read_rows(assets / 'input.csv')}
    for role in ('train', 'val', 'test', 'leftout'):
        inputs = read_rows(run / 'inputs' / (role + '.csv'))
        prepared = []
        for row in inputs:
            identifier = row['record_id']
            source = asset_inputs[identifier]
            for column in ('siRNA', 'extended_mRNA'):
                if row[column].upper().replace('T', 'U') != source[column].upper().replace('T', 'U'):
                    raise ValueError(f'Feature asset sequence differs: {identifier}/{column}')
            prepared.append(asset_rows[identifier])
        with (data / (role + '.csv')).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prepared[0]))
            writer.writeheader()
            writer.writerows(prepared)
    names = ['data'] if tool == 'attsioff' else ['siRNA_split_preprocess', 'RNA_AGO2']
    for name in names:
        shutil.copytree(assets / name, data / name, copy_function=os.link, symlinks=False)
    shutil.copyfile(assets / 'SHA256SUMS', data / 'ASSET_SHA256SUMS')
    params_flags = []
    if 'params' in specification:
        path = run / 'params.json'
        path.write_text(json.dumps(specification['params'], indent=2) + '\n')
        params_flags = ['--params-json', str(path)]
    extra = ['--data-dir', str(data)] if tool == 'attsioff' else [
        '--preprocess-dir', str(data / 'siRNA_split_preprocess'), '--rna-ago2-dir', str(data / 'RNA_AGO2')]
    train = ['python3', str(scripts / 'train.py'), '--tool', tool, '--train-csv', str(data / 'train.csv'),
             '--val-csv', str(data / 'val.csv'), '--model-dir', str(model), '--seed', str(specification['seed']),
             *extra, *params_flags, *specification['train_flags']]
    if 'training_entrypoint' in specification:
        train[1] = str(repo / specification['training_entrypoint'])
    for relative in specification.get('variant_source_files', []):
        archive = run / 'runtime_sources'
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / relative, archive / Path(relative).name)
    commands = [train]
    extension = 'pt' if tool == 'attsioff' else 'keras'
    for role, suffix in [('test', ''), ('leftout', '_leftout')]:
        commands.append(['python3', str(scripts / 'test.py'), '--tool', tool, '--test-csv', str(data / (role + '.csv')),
                         '--model-path', str(model / ('model.' + extension)), '--output-csv', str(results / ('preds' + suffix + '.csv')),
                         '--metrics-json', str(results / ('metrics' + suffix + '.json')), *extra, *params_flags])
    (run / 'sensitivity_specification.json').write_text(json.dumps(specification, indent=2) + '\n')
    (run / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    environment = dict(os.environ, QUIET='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    for command in commands:
        subprocess.run(command, check=True, env=environment)
    check_model_completion(run, tool, specification['seed'], required=True)
    files = ['sensitivity_specification.json', 'commands.json', f'models/{tool}/train_meta.json',
             f'results/{tool}/preds.csv', f'results/{tool}/preds_leftout.csv']
    (run / 'SENSITIVITY_COMPLETE.json').write_text(json.dumps({name: digest(run / name) for name in files}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--specification', type=Path, required=True)
    args = parser.parse_args()
    execute_case(args.repo, json.loads(args.specification.read_text()))


if __name__ == '__main__':
    main()
