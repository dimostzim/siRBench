"""Describe frozen cohorts after selection; never choose partitions from outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--groups',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a new, empty description directory')
    records = pd.read_csv(args.records)
    groups = pd.read_csv(args.groups)
    if set(records.record_id)!=set(groups.record_id):
        raise ValueError('Group annotations must cover exactly the same records')
    records = records.merge(groups,on='record_id',validate='one_to_one')
    is_hela = records.cell_line.str.lower().eq('hela')
    development = records[~is_hela]
    cohorts = {'nonhela':development, 'hela_full':records[is_hela],
               'hela_aligned':records[is_hela & records.hela_aligned]}
    descriptions = []
    for name,frame in cohorts.items():
        descriptions.append({'cohort':name,'n':len(frame),
            'target_groups':frame.target_group_id.nunique(),
            'split_groups':frame.split_group_id.nunique(),
            'efficacy_mean':frame.efficiency.mean(),'efficacy_sd':frame.efficiency.std(),
            'efficacy_median':frame.efficiency.median(),
            'ks_statistic_against_nonhela':ks_2samp(development.efficiency,frame.efficiency).statistic,
            'records_with_target_group_seen_in_nonhela':frame.target_group_id.isin(development.target_group_id).sum()})
    args.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(descriptions).to_csv(args.output/'cohort_summary.csv',index=False)
    records.groupby(['source','cell_line']).agg(n=('record_id','size'),
        efficacy_mean=('efficiency','mean'),efficacy_sd=('efficiency','std')).reset_index().to_csv(
            args.output/'source_cell_summary.csv',index=False)
    development.groupby('split_group_id').size().rename('n').to_csv(args.output/'nonhela_group_sizes.csv')
    manifest = {'purpose':__doc__,'source_cell_policy':'Released annotations; unresolved historical attribution is not corrected by this descriptive analysis.',
        'input_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [args.records,args.groups,Path(__file__)]}}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
