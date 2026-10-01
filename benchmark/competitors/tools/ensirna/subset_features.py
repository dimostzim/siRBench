#!/usr/bin/env python3
"""Subset deterministic per-record features; inject labels by stable record ID."""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
import sys

from prepare import clean_seq, revcomp


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_canonical(jsonl, processed):
    records = [json.loads(line) for line in jsonl.read_text().splitlines()]
    record_map = {item['id']: item for item in records}
    if len(record_map) != len(records):
        raise ValueError('Duplicate canonical record IDs')
    metadata = json.loads((processed / '_metainfo').read_text())
    features = []
    for name, count in zip(metadata['file_names'], metadata['file_num_entries']):
        with Path(name).open('rb') as handle:
            part = pickle.load(handle)
        if len(part) != count:
            raise ValueError('Feature part length disagrees with metadata')
        features.extend(part)
    feature_map = {item[0].benchmark_record_id: item for item in features}
    if len(feature_map) != len(features) or set(feature_map) != set(record_map):
        raise ValueError('Canonical features and records do not have identical unique IDs')
    if metadata['num_entry'] != len(records):
        raise ValueError('Canonical feature count differs from records')
    return record_map, feature_map, metadata


def select_records(split_rows, records, features):
    ids = [row['record_id'] for row in split_rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate split record IDs')
    selected_records, selected_features = [], []
    for row in split_rows:
        record_id = row['record_id']
        item = dict(records[record_id])
        guide = row['siRNA'].upper().replace('T', 'U')
        context = clean_seq(row['extended_mRNA'])
        if item['anti seq'] != guide or item['sense seq'] != revcomp(guide) or item['mRNA_seq'] != context:
            raise ValueError(f'Sequence mismatch for {record_id}')
        item['efficiency'] = float(row['efficiency'])
        feature = list(features[record_id])
        feature[1] = item['efficiency']
        selected_records.append(item)
        selected_features.append(feature)
    return selected_records, selected_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical-jsonl', type=Path, required=True)
    parser.add_argument('--canonical-processed', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--asset', type=Path, action='append', default=[])
    parser.add_argument('--split-csv', type=Path)
    parser.add_argument('--output-jsonl', type=Path)
    args = parser.parse_args()
    src = Path(__file__).resolve().parent / 'ensirna_src/ENsiRNA'
    sys.path.insert(0, str(src))
    records, features, metadata = load_canonical(args.canonical_jsonl, args.canonical_processed)
    tracked = [args.canonical_jsonl, args.canonical_processed / '_metainfo',
               *map(Path, metadata['file_names']), *sorted(src.rglob('*.py')),
               Path(__file__).resolve().parent / 'prepare.py', Path(__file__).resolve(), *args.asset]
    tracked.extend(Path(item['pdb_data_path']) for item in records.values())
    if args.freeze:
        if args.manifest.exists():
            raise FileExistsError(args.manifest)
        manifest = {'record_ids': list(records), 'files': {str(path.resolve()): sha256(path) for path in tracked}}
        temporary = args.manifest.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(args.manifest)
        return
    manifest = json.loads(args.manifest.read_text())
    if list(records) != manifest['record_ids']:
        raise ValueError('Canonical record manifest changed')
    for name, expected in manifest['files'].items():
        if sha256(name) != expected:
            raise ValueError(f'Canonical cache dependency changed: {name}')
    if not args.split_csv or not args.output_jsonl:
        parser.error('--split-csv and --output-jsonl are required unless freezing')
    with args.split_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    selected_records, selected_features = select_records(rows, records, features)
    processed = args.output_jsonl.with_name(args.output_jsonl.stem + '_processed')
    if args.output_jsonl.exists() or processed.exists():
        raise FileExistsError('Refusing to overwrite a prepared split')
    processed.mkdir(parents=True)
    part = processed / 'part_0.pkl'
    with part.open('wb') as handle:
        pickle.dump(selected_features, handle)
    (processed / '_metainfo').write_text(json.dumps({'num_entry': len(rows),
        'file_names': [str(part.resolve())], 'file_num_entries': [len(rows)]}))
    args.output_jsonl.write_text(''.join(json.dumps(item) + '\n' for item in selected_records))
    (processed / 'revision_manifest.json').write_text(json.dumps({
        'canonical_manifest_sha256': sha256(args.manifest), 'split_csv_sha256': sha256(args.split_csv),
        'record_ids': [item['id'] for item in selected_records]}, indent=2) + '\n')


if __name__ == '__main__':
    main()
