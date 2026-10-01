"""Derive input-extent sensitivity candidates without guessing among references."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from map_targets import read_fasta


def padded_window(sequence, start, end):
    return 'X'*max(0,-start) + sequence[max(0,start):min(len(sequence),end)] + 'X'*max(0,end-len(sequence))


def extent_candidates(record, candidates, references, sequences, full_transcript_policy="consensus"):
    strong = candidates[candidates.match_quality.isin(['context_exact','context_unknown_bases'])]
    output = {'record_id':record.record_id,'strong_candidate_count':len(strong)}
    for width in [59,61]:
        flank = (width-19)//2
        windows = set()
        for candidate in strong.itertuples(index=False):
            reference = sequences[candidate.reference_id]
            window = padded_window(reference,candidate.target_start_0based-flank,candidate.target_start_0based+19+flank)
            added = (width-57)//2
            # Preserve the complete standardized57nt core, including unknown bases.
            windows.add(window[:added]+record.extended_mRNA+window[-added:])
        output[f'context_{width}'] = next(iter(windows)) if len(windows)==1 else ''
        output[f'context_{width}_status'] = 'consensus' if len(windows)==1 else ('no_strong_match' if not windows else 'conflicting_extensions')
    transcript_candidates = []
    for candidate in strong.itertuples(index=False):
        metadata = references.loc[candidate.reference_id]
        if metadata.molecule_type == 'mRNA' or metadata.accession_base == 'EGFP':
            transcript_candidates.append(candidate)
    available = transcript_candidates
    archived = [c for c in available if c.match_quality == 'context_exact'
                and ('/upstream/' in references.loc[c.reference_id].origin
                     or '/repository_history/' in references.loc[c.reference_id].origin)] if full_transcript_policy == 'archived_preferred' else []
    if archived:
        transcript_candidates = archived
    output['full_mRNA_selection_rule'] = 'matching_archived_transcript' if archived else 'all_confirmed_transcript_candidates'
    output['available_transcript_reference_ids'] = ';'.join(sorted(c.reference_id for c in available))
    output['alternative_full_sequence_count'] = len({sequences[c.reference_id] for c in available})
    full_sequences = {sequences[c.reference_id] for c in transcript_candidates}
    output['full_mRNA'] = next(iter(full_sequences)) if len(full_sequences)==1 else ''
    output['full_mRNA_status'] = 'consensus' if len(full_sequences)==1 else ('no_confirmed_transcript' if not full_sequences else 'conflicting_transcript_sequences')
    output['transcript_reference_ids'] = ';'.join(sorted(c.reference_id for c in transcript_candidates))
    output['full_mRNA_target_positions_0based'] = ';'.join(map(str,sorted({c.target_start_0based for c in transcript_candidates})))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--targets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--full-transcript-policy',choices=['consensus','archived_preferred'],default='consensus')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a new, empty extent output directory')
    records = pd.read_csv(args.records)
    references = pd.read_csv(args.targets/'references.csv').fillna('').set_index('reference_id')
    candidates = pd.read_csv(args.targets/'target_candidates.csv')
    sequences = dict(read_fasta(args.targets/'references.fasta'))
    grouped = {key:frame for key,frame in candidates.groupby('record_id')}
    rows = [extent_candidates(record,grouped[record.record_id],references,sequences,args.full_transcript_policy) for record in records.itertuples(index=False)]
    audit = pd.DataFrame(rows)
    args.output.mkdir(parents=True,exist_ok=True)
    audit.to_csv(args.output/'extent_candidates.csv',index=False)
    for extent,column in [('context59','context_59'),('context61','context_61'),('full_mRNA','full_mRNA')]:
        merged = records.merge(audit[['record_id',column]],on='record_id',validate='one_to_one')
        merged = merged[merged[column].ne('')].copy()
        merged['standardized_context_57'] = merged.extended_mRNA
        merged['extended_mRNA'] = merged.pop(column)
        merged.to_csv(args.output/f'{extent}_eligible.csv',index=False)
    status = {column:audit[column].value_counts().to_dict() for column in audit if column.endswith('_status')}
    status['full_transcript_policy'] = args.full_transcript_policy
    status['rules'] = 'Only full-context matching candidates; 19nt-only matches excluded. Local extensions must agree across all candidates; original57nt center retained exactly. Full sequences must agree within the selected candidate tier (matching archived transcripts when preferred; otherwise all mRNA candidates); genomic DNA excluded, archived EGFP reporter accepted. These are reconstructed input extents, not proof of the exact historical assay isoform.'
    status['input_sha256'] = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [args.records,*[args.targets/name for name in ['references.csv','references.fasta','target_candidates.csv']]]}
    (args.output/'manifest.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2))


if __name__ == '__main__':
    main()
