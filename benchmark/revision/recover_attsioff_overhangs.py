"""Recover experimentally listed 21-nt guides; never infer overhangs from targets."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def recover_guides(records, original_guides):
    candidates = {}
    for guide in original_guides.dropna():
        guide = str(guide).strip().upper().replace('T', 'U')
        if len(guide) != 21 or set(guide) - set('ACGU'):
            raise ValueError(f'Invalid original 21-nt guide: {guide}')
        candidates.setdefault(guide[:19], set()).add(guide)
    rows = []
    for record in records.itertuples(index=False):
        matches = candidates.get(record.siRNA.upper().replace('T', 'U'), set())
        status = ('unique_original_21nt' if len(matches) == 1 else
                  'conflicting_original_21nt' if matches else 'original_21nt_unavailable')
        rows.append({'record_id': record.record_id, 'status': status,
                     'original_guide_21': next(iter(matches)) if len(matches) == 1 else '',
                     'candidates': ';'.join(sorted(matches))})
    audit = pd.DataFrame(rows)
    eligible = records.merge(audit[['record_id', 'original_guide_21']],
                             on='record_id', validate='one_to_one')
    eligible = eligible[eligible.original_guide_21.ne('')].copy()
    eligible['standardized_guide_19'] = eligible.siRNA
    eligible['siRNA'] = eligible.pop('original_guide_21')
    return eligible, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context59', type=Path, required=True)
    parser.add_argument('--source-xls', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a new, empty output directory')
    records = pd.read_csv(args.context59)
    if not records.extended_mRNA.str.len().eq(59).all():
        raise ValueError('Expected 59-nt contexts')
    source = pd.read_excel(args.source_xls, sheet_name='Sheet1')
    eligible, audit = recover_guides(records, source['Antisense, 21 mer'])
    args.output.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(args.output/'attsioff_original21_59_eligible.csv', index=False)
    audit.to_csv(args.output/'attsioff_overhang_audit.csv', index=False)
    manifest = {
        'eligible': len(eligible), 'status_counts': audit.status.value_counts().to_dict(),
        'rule': 'Exact 19-nt guide core matches a unique experimentally listed 21-nt antisense strand in File007. No target-derived overhang inference.',
        'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [args.context59, args.source_xls, Path(__file__)]},
    }
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
