import json,os,subprocess,sys,tempfile
from pathlib import Path
import pandas as pd
sys.path.insert(0,'/work/benchmark/revision')
from gnn_rnaup_cached import main
root=Path('/revision')
records=pd.read_csv(root/'evaluation/sensitivity-v1/gnn4sirna_full_mrna/original/fold_0/train.csv')
records=records[records.extended_mRNA.str.len().le(720)].iloc[:3]
with tempfile.TemporaryDirectory(dir=root/'tmp') as directory:
 work=Path(directory);os.chdir(work)
 rows=[]
 for row in records.itertuples():
  sense=row.siRNA.replace('U','T').translate(str.maketrans('ACGT','TGCA'))[::-1]
  rows.append([row.record_id,sense,'target',row.extended_mRNA])
 pd.DataFrame(rows).to_csv('datase_tofold.csv',index=False,header=False)
 subprocess.run([sys.executable,'/work/benchmark/competitors/tools/gnn4sirna/gnn4sirna_src/preprocessing/4_make_RNAUp.py'],check=True)
 expected=pd.read_csv('dataset_folded.csv')
 args=['--cache',str(work/'cache'),'--workers','2']
 main(args)
 pd.testing.assert_frame_equal(pd.read_csv('dataset_folded.csv'),expected)
 timestamps={p:p.stat().st_mtime_ns for p in (work/'cache').glob('*.json')}
 main(args)
 assert timestamps=={p:p.stat().st_mtime_ns for p in (work/'cache').glob('*.json')}
 path=next(iter(timestamps));entry=json.loads(path.read_text());entry['values'][0]+=1;path.write_text(json.dumps(entry))
 try:main(args)
 except ValueError:pass
 else:raise AssertionError('Corrupt cache accepted')
print('PASS: cached and published serial RNAup outputs exactly equal; cache resume unchanged; corruption rejected')
