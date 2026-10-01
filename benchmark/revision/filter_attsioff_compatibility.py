"""Reproduce conservative extent eligibility from the completed provenance audit."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def compatible_extents(extents, audit):
    if not extents.record_id.is_unique or not audit.record_id.is_unique:
        raise ValueError('Extent and audit record IDs must be unique')
    if set(extents.record_id) != set(audit.record_id) or audit.status.isna().any():
        raise ValueError('Every extent record must have exactly one audit status')
    accepted = audit.loc[audit.status.eq('source_and_scaled_label_compatible'), 'record_id']
    return extents.loc[extents.record_id.isin(accepted)].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extents', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a new, empty output directory')
    extents, audit = pd.read_csv(args.extents), pd.read_csv(args.audit)
    selected = compatible_extents(extents, audit)
    args.output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output/'attsioff_original21_59_eligible.csv', index=False)
    audit.to_csv(args.output/'eligibility_audit.csv', index=False)
    manifest = {
        'eligible': len(selected), 'excluded': len(extents)-len(selected),
        'policy': 'Retain only source_and_scaled_label_compatible records from the completed audit. Source and scaled-label agreement verify compatibility; exact historical assay identity is not established. No annotations or labels are changed. This secondary eligibility check is not label-blind; no model performance or label-distribution optimization is used.',
        'input_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [args.extents, args.audit, Path(__file__)]},
        'output_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in args.output.glob('*.csv')},
    }
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
