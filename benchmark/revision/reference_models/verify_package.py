"""Check packaged artifacts and prediction parity against frozen evaluation outputs."""
import argparse,csv,hashlib,json,math,subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--evaluation-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    records=[]
    for entry in json.loads((args.package/'manifest.json').read_text()):
        fold=entry['fold'];project=args.package/f'fold_{fold}';reference=args.evaluation_root/f'fold_{fold}'
        for name,expected in entry['files'].items():
            assert hashlib.sha256((project/name).read_bytes()).hexdigest()==expected['sha256'],name
        subprocess.run(['uv','run','--frozen','--project',str(project),'python',str(project/'train.py'),'--help'],check=True,stdout=subprocess.DEVNULL)
        for split in ('validation','test','test_hela_full'):
            output=args.output_dir/f'fold_{fold}-{split}.csv'
            subprocess.run(['uv','run','--frozen','--project',str(project),'python',str(project/'inference.py'),'--input',str(reference/'inputs'/split),'--output',str(output)],check=True)
            with output.open() as stream:actual=list(csv.DictReader(stream))
            with (reference/'predictions'/f'{split}.csv').open() as stream:expected=list(csv.DictReader(stream))
            assert [row['id'] for row in actual]==[row['id'] for row in expected]
            differences=[abs(float(a['prediction'])-float(e['prediction'])) for a,e in zip(actual,expected)]
            assert all(math.isfinite(x) for x in differences)
            maximum=max(differences,default=0)
            assert maximum<=1e-6,(fold,split,maximum)
            records.append(dict(fold=fold,split=split,rows=len(actual),maximum_absolute_difference=maximum,status='PASS'))
            (args.output_dir/'verification.json').write_text(json.dumps(records,indent=2)+'\n')
            print(records[-1],flush=True)
    print('All artifact hashes, training CLI imports, prediction IDs, and prediction parity checks passed.',flush=True)

if __name__=='__main__':main()
