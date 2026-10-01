"""Recompute the archived numerical tables from sealed predictions, without fitting models."""
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
import pandas as pd
import numpy as np
from io import BytesIO

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--package',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();package=args.package.resolve();out=args.output.resolve()
out.mkdir(parents=True,exist_ok=False)
original=Path('/SCRATCH/dtzim01/sirbench-revision-20260921')
work=out/'workspace';shutil.copytree(package/'workspace',work)

def relocate_index(relative):
    path=work/relative;frame=pd.read_csv(path)
    for col in ['run_dir','test_predictions','hela_predictions','train_meta']:
        frame[col]=[str(work/Path(p).relative_to(original)) for p in frame[col]]
    frame.to_csv(path,index=False)
    return path

primary=relocate_index(Path('evaluation/primary-index-v1/run_index.csv'))
tabpfn=relocate_index(Path('evaluation/tabpfn-benchmark-v1/run_index.csv'))
descriptors=json.loads((work/'evaluation/sensitivity-analysis-v3/descriptors.json').read_text())
seen={primary,tabpfn}
for comparison in descriptors['comparisons']:
    for side in ['reference','variant']:
        relative=Path(comparison[side]['index']);path=work/relative
        if path not in seen:relocate_index(relative);seen.add(path)
        comparison[side]['index']=str(path)
desc=out/'relocated-descriptors.json';desc.write_text(json.dumps(descriptors,indent=2)+'\n')
code=work/'siRBench/benchmark/revision'
helpers=code/'reference_results/code'
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}

def run(arguments):
    subprocess.run([sys.executable,*map(str,arguments)],check=True,env=env)

expanded=out/'expanded-analysis'
run([helpers/'evaluate_predictions.py','--records',work/'datasets/corrected-v1/records_features.csv',
 '--protocol',work/'evaluation/protocol-v1','--baselines',work/'evaluation/baselines-v1/predictions.csv',
 '--run-index',primary,'--run-index',tabpfn,'--output',expanded,'--bootstrap-draws','2000'])
sensitivity=out/'sensitivity-analysis'
run([work/'runtime_snapshots/sensitivity-analysis-v3/summarize_sensitivities.py',
 '--workspace',work,'--descriptors',desc,'--output',sensitivity])
references=out/'reference-analysis'
run([helpers/'integrate_reference_results.py','--root',work,'--selected',package/'reference-evaluation',
 '--helpers',helpers,'--output',references])
checks=[]
serialization_checks=[]
for relative,new in [('evaluation/primary-analysis-with-tabpfn-v1',expanded),
 ('evaluation/sensitivity-analysis-v3',sensitivity),('evaluation/primary-analysis-with-reference-models-v1',references)]:
    for expected in sorted((package/'workspace'/relative).glob('*.csv')):
        actual=new/expected.name
        normalized=actual.read_bytes().replace(str(work).encode(),str(original).encode())
        if normalized!=expected.read_bytes():
            # This archived table was serialized after a CSV round trip.
            # Require exact identities/missingness and bound floating formatting loss.
            if expected.name!='reference_replicate_metrics.csv':
                raise ValueError(f'Replay differs: {relative}/{expected.name}')
            archived=pd.read_csv(expected,float_precision='round_trip')
            replayed=pd.read_csv(BytesIO(normalized),float_precision='round_trip')
            assert archived.shape==replayed.shape and list(archived.columns)==list(replayed.columns)
            maximum=0.0
            for column in archived:
                left,right=archived[column],replayed[column]
                if left.dtype.kind=='f':
                    assert np.array_equal(left.isna(),right.isna()),column
                    difference=float((left-right).abs().max())
                    assert difference<=1e-14,(column,difference)
                    maximum=max(maximum,difference)
                else:
                    pd.testing.assert_series_equal(left,right,check_exact=True)
            serialization_checks.append({'table':str(Path(relative)/expected.name),
                'max_absolute_float_difference':maximum,'tolerance':1e-14,
                'identity_columns_and_missingness_exact':True})
        checks.append(str(Path(relative)/expected.name))
report={'status':'PASS','verified_numerical_tables':checks,'serialization_checks':serialization_checks,'bootstrap_draws':2000,
 'scope':'Analysis replay from archived predictions, including TabPFN and five reference models; no training or inference rerun.',
 'path_relocation':'Only copied index paths and sensitivity descriptors were changed; sealed prediction/metadata bytes preserved.'}
(out/'replay-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
