"""Descriptive, seed-paired sensitivity metrics with strict cohort identity checks."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from assemble_primary_index import completed_runs
from evaluate_predictions import METRICS, metrics, replicate_mean, replicate_std
from prediction_artifacts import read_artifact_bytes, validate_artifact_paths


class ArtifactReader:
    def __init__(self):
        self.hashes = {}
        self.index_snapshots = {}

    def read(self, path):
        path = Path(path)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if str(path) in self.hashes and self.hashes[str(path)] != digest:
            raise ValueError(f'Input changed during analysis: {path}')
        self.hashes[str(path)] = digest
        return payload

    def csv(self, path, **kwargs):
        return pd.read_csv(BytesIO(self.read(path)), **kwargs)

    def artifact(self, row, column):
        payload, digest = read_artifact_bytes(row, column)
        path = str(getattr(row, column))
        if path in self.hashes and self.hashes[path] != digest:
            raise ValueError(f'Input changed during analysis: {path}')
        self.hashes[path] = digest
        return payload


def unique_labels(frame, id_column, label_column, description):
    if frame[id_column].isna().any() or not frame[id_column].is_unique:
        raise ValueError(f'Duplicate or missing IDs: {description}')
    if not np.isfinite(frame[label_column].to_numpy(dtype=float)).all():
        raise ValueError(f'Nonfinite labels: {description}')
    return frame.set_index(id_column)[label_column].sort_index()


def require_same_labels(left, right, description):
    if not left.index.equals(right.index):
        raise ValueError(f'Identity mismatch: {description}')
    if not np.allclose(left.to_numpy(), right.to_numpy(), rtol=0, atol=1e-12):
        raise ValueError(f'Label mismatch: {description}')


def select_run(index, selector, tool, seed):
    mask = index.tool.eq(tool) & index.training_seed.eq(seed)
    for field, value in selector['match'].items():
        if field not in index:
            raise ValueError(f'Missing selector field {field}')
        mask &= index[field].eq(value)
    selected = index[mask]
    if len(selected) > 1:
        raise ValueError(f'Duplicate selected run: {tool}, seed {seed}, {selector}')
    if selected.empty or not completed_runs(selected).all():
        return None
    return selected.iloc[0].to_dict()


def load_run(row, reader):
    for source, target in [('prediction_sha256','test_predictions_sha256'), ('hela_sha256','hela_predictions_sha256')]:
        if source in row and pd.notna(row[source]):
            if target in row and pd.notna(row[target]) and row[source] != row[target]:
                raise ValueError('Conflicting prediction seals in run index')
            row[target] = row[source]
    validate_artifact_paths(pd.DataFrame([row]))
    named = SimpleNamespace(**row)
    metadata = json.loads(reader.artifact(named, 'train_meta'))
    directory = Path(row['run_dir'])
    manifest_path = directory/'manifest.json'
    manifest = json.loads(reader.read(manifest_path))
    if manifest['settings']['seed'] != row['training_seed'] or manifest['settings']['tools'] != [row['tool']]:
        raise ValueError(f'Launch manifest identity mismatch: {directory}')
    inputs = {}
    for role in ['train','val','test','leftout']:
        path = directory/'inputs'/(role+'.csv')
        frame = reader.csv(path,usecols=['record_id','efficiency'])
        if reader.hashes[str(path)] != manifest['inputs'][role]['sha256']:
            raise ValueError(f'Archived input hash differs from launch manifest: {path}')
        inputs[role] = unique_labels(frame,'record_id','efficiency',str(path))
    predictions = {}
    for role, column in [('test','test_predictions'),('leftout','hela_predictions')]:
        frame = pd.read_csv(BytesIO(reader.artifact(named,column))).rename(columns={'record_id':'id'})
        labels = unique_labels(frame,'id','label',row[column])
        require_same_labels(inputs[role],labels,row[column])
        if not np.isfinite(frame.pred_label.to_numpy(dtype=float)).all():
            raise ValueError(f'Nonfinite predictions: {row[column]}')
        predictions[role] = frame.set_index('id').sort_index()
    return {'row':row,'metadata':metadata,'inputs':inputs,'predictions':predictions}


def compare_runs(reference, variant, hela_ids):
    if reference['row']['training_seed'] != variant['row']['training_seed']:
        raise ValueError('Training seed mismatch between paired runs')
    if reference['row']['tool'] != variant['row']['tool']:
        raise ValueError('Tool mismatch between paired runs')
    if Path(reference['row']['run_dir']).resolve() == Path(variant['row']['run_dir']).resolve():
        raise ValueError('Reference and variant reuse the same run')
    for column in ['test_predictions','hela_predictions','train_meta']:
        if Path(reference['row'][column]).resolve() == Path(variant['row'][column]).resolve():
            raise ValueError('Reference and variant reuse an artifact')
    for role in ['train','val','test','leftout']:
        require_same_labels(reference['inputs'][role],variant['inputs'][role],role)
    rows = []
    for role, cohort in [('test','test'),('leftout','hela')]:
        left, right = reference['predictions'][role], variant['predictions'][role]
        if role == 'leftout' and not set(left.index) <= hela_ids:
            raise ValueError('Held-out predictions include an ID outside frozen full HeLa')
        if role == 'test' and set(left.index) & hela_ids:
            raise ValueError('Main-fold test predictions contain a HeLa record')
        record = {'cohort':cohort,'n':len(left),'full_hela_n':len(hela_ids),
                  'excluded_full_hela':len(hela_ids)-len(left) if role=='leftout' else pd.NA}
        a, b = metrics(left.label,left.pred_label), metrics(right.label,right.pred_label)
        for metric in METRICS:
            record['reference_'+metric] = a[metric]
            record['variant_'+metric] = b[metric]
            record['delta_'+metric] = b[metric]-a[metric]
        rows.append(record)
    return rows


def summarize_deltas(per_seed, statuses):
    rows = []
    for (comparison,tool,cohort), frame in per_seed.groupby(['comparison','tool','cohort']):
        for field in ['n','full_hela_n']:
            if frame[field].nunique() != 1:
                raise ValueError(f'Cohort size differs across sensitivity seeds: {comparison}')
        status = statuses[comparison]
        for metric in METRICS:
            row = {'comparison':comparison,'tool':tool,'cohort':cohort,'metric':metric,
                   'n':int(frame.n.iloc[0]),'full_hela_n':int(frame.full_hela_n.iloc[0]),
                   'excluded_full_hela':frame.excluded_full_hela.iloc[0],
                   'paired_seeds':len(frame),'expected_seeds':status['expected_seeds'],
                   'status':status['status']}
            for prefix in ['reference','variant','delta']:
                values = frame[prefix+'_'+metric]
                row[prefix+'_mean'] = replicate_mean(values)
                row[prefix+'_sd'] = replicate_std(values)
                row[prefix+'_finite_seeds'] = int(np.isfinite(values).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def analyze(descriptors, workspace, hela_ids, reader):
    index_cache, run_cache, statuses, rows = {}, {}, {}, []
    for descriptor in descriptors['comparisons']:
        comparison, tool = descriptor['id'], descriptor['tool']
        if comparison in statuses:
            raise ValueError(f'Duplicate comparison descriptor: {comparison}')
        seeds = descriptor['seeds']
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f'Duplicate expected seed: {comparison}')
        paired, pending = [], []
        cohort_reference = None
        for seed in seeds:
            selected = {}
            for side in ['reference','variant']:
                selector = descriptor[side]
                path = workspace/selector['index']
                if not path.is_file():
                    pending.append(f'seed {seed}: {side} index absent')
                    continue
                if path not in index_cache:
                    payload = reader.read(path)
                    reader.index_snapshots[str(path)] = payload
                    index_cache[path] = pd.read_csv(BytesIO(payload))
                row = select_run(index_cache[path],selector,tool,seed)
                if row is None:
                    pending.append(f'seed {seed}: {side} run not verified')
                    continue
                cache_key = (str(path),row['run_dir'],tool,seed)
                if cache_key not in run_cache:
                    run_cache[cache_key] = load_run(row,reader)
                selected[side] = run_cache[cache_key]
            if len(selected) != 2:
                continue
            if cohort_reference is not None:
                for role in ['train','val','test','leftout']:
                    require_same_labels(cohort_reference[role],selected['reference']['inputs'][role],
                                        f'{comparison}: cohort across seeds, {role}')
            cohort_reference = selected['reference']['inputs']
            for result in compare_runs(selected['reference'],selected['variant'],hela_ids):
                rows.append({'comparison':comparison,'tool':tool,'training_seed':seed,
                             'reference_run':selected['reference']['row']['run_dir'],
                             'variant_run':selected['variant']['row']['run_dir'],**result})
            paired.append(seed)
        statuses[comparison] = {'comparison':comparison,'tool':tool,'description':descriptor['description'],
                                'cohort_description':descriptor['cohort_description'],
                                'expected_seeds':len(seeds),'paired_seeds':len(paired),
                                'verified_seed_values':','.join(map(str,paired)),
                                'status':'complete' if len(paired)==len(seeds) else ('partial' if paired else 'pending'),
                                'pending_reason':'; '.join(pending)}
    per_seed = pd.DataFrame(rows)
    summary = summarize_deltas(per_seed,statuses) if rows else pd.DataFrame()
    return per_seed,summary,pd.DataFrame(statuses.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--descriptors',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a new, empty sensitivity output directory')
    reader = ArtifactReader()
    descriptors = json.loads(reader.read(args.descriptors))
    hela = reader.csv(args.workspace/descriptors['full_hela_csv'],usecols=['record_id'])
    if not hela.record_id.is_unique:
        raise ValueError('Frozen full HeLa has duplicate IDs')
    per_seed,summary,statuses = analyze(descriptors,args.workspace,set(hela.record_id),reader)
    # A mutable controller index may gain rows while reporting; retry a fresh snapshot.
    for path, expected in reader.hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Input changed during analysis: {path}')
    args.output.mkdir(parents=True,exist_ok=True)
    per_seed.to_csv(args.output/'per_seed_metrics.csv',index=False)
    summary.to_csv(args.output/'paired_summary.csv',index=False)
    statuses.to_csv(args.output/'comparison_status.csv',index=False)
    (args.output/'descriptors.json').write_text(json.dumps(descriptors,indent=2)+'\n')
    snapshots = args.output/'index_snapshots'
    snapshots.mkdir()
    snapshot_names = {}
    for number,(path,payload) in enumerate(sorted(reader.index_snapshots.items())):
        target = snapshots/f'{number:02d}-{Path(path).name}'
        target.write_bytes(payload)
        snapshot_names[path] = str(target.relative_to(args.output))
    source_paths = [Path(__file__), *[Path(__file__).with_name(name) for name in
                    ['assemble_primary_index.py', 'evaluate_predictions.py', 'prediction_artifacts.py']]]
    manifest = {'inputs':reader.hashes,'index_snapshots':snapshot_names,'code':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                'delta_definition':'Variant minus reference, computed within identical seed and cohort before aggregation.',
                'uncertainty':'Sample SD across the available paired seeds is descriptive, not a standard error or confidence interval. No rankings, p-values, or superiority claims.',
                'undefined_metrics':'Any undefined seed metric propagates into its mean and SD; finite and total paired seed counts are reported.',
                'cohorts':'All train/validation/test/HeLa IDs and labels must match between each paired fit. Eligible HeLa subsets are compared only within that matched cohort; excluded_full_hela counts frozen full-HeLa IDs absent from the cohort.',
                'completion':'Missing or unverified fits remain pending/partial. A completed comparison requires every prespecified seed; pending comparisons do not receive fabricated metrics.'}
    (args.output/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(statuses[['comparison','status','paired_seeds']].to_string(index=False))


if __name__ == '__main__':
    main()
