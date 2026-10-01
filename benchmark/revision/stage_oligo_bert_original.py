"""Create isolated schedule-control code without changing any primary runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def replace_once(path, old, new):
    source = path.read_text()
    if source.count(old) != 1:
        raise ValueError(f'Frozen source does not match the reviewed patch: {path}')
    path.write_text(source.replace(old,new,1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-runtime',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    shutil.copytree(args.frozen_runtime,args.output)
    relative = Path('siRBench/benchmark/competitors/tools')
    bert = args.output/relative/'sirnabert/train.py'
    replace_once(bert,'        if improved:\n',
                 '        if args.original_params or improved:\n')
    replace_once(bert,'        "best_epoch": best_epoch,\n',
                 '        "best_epoch": best_epoch,\n'
                 '        "epochs_completed": epoch + 1,\n'
                 '        "checkpoint_selection": "final_epoch" if args.original_params else "best_validation",\n')
    oligo = args.output/relative/'oligoformer/prepare.py'
    replace_once(oligo,
                 'y_vals = (df[args.efficiency_col].astype(float) >= args.binary_threshold).astype(int)',
                 'y_vals = (df[args.efficiency_col].astype(float) > args.binary_threshold).astype(int)')
    changes = {}
    for path in [bert,oligo]:
        name = path.relative_to(args.output)
        changes[str(name)] = {
            'before':hashlib.sha256((args.frozen_runtime/name).read_bytes()).hexdigest(),
            'after':hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {
        'purpose':'Original-schedule sensitivity only; primary runtime is unchanged.',
        'changes':changes,
        'bert':'Thirty epochs, final-epoch checkpoint, no early stopping. Evaluation remains in eval mode; no per-epoch test scoring from the demonstration script is reproduced.',
        'oligo':'Auxiliary binary labels use the released strict >0.7 boundary for joint validation-loss/AUC selection.',
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output/'original_schedule_patch.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
