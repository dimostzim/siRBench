"""Runtime checks: run in a PyTorch environment with pandas and scikit-learn."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import pandas as pd
import torch

TOOLS = Path(__file__).resolve().parents[2] / 'competitors' / 'tools'


def load_script(tool, script):
    spec = importlib.util.spec_from_file_location(f'{tool}_{script}', TOOLS / tool / f'{script}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_efficacy_r2():
    module = load_script('oligoformer', 'train')

    class PerfectRegressor(torch.nn.Module):
        def forward(self, sirna, mrna, sirna_fm, mrna_fm, td):
            return torch.stack((1 - sirna, sirna), dim=1), None, None

    # Deliberately different continuous/binary labels; unequal batches exercise aggregation.
    efficacy = torch.tensor([0.1, 0.3, 0.8])
    binary = torch.tensor([0, 0, 1])
    batches = []
    for sl in [slice(0, 2), slice(2, 3)]:
        y = efficacy[sl]
        batches.append((y, y, y, y, y, binary[sl], y))
    loss, auc, r2 = module.eval_epoch(PerfectRegressor(), batches, torch.nn.MSELoss(), 'cpu')
    assert loss == 0.0, loss
    assert auc == 1.0, auc
    assert r2 == 1.0, f'Perfect efficacy predictions must have R2=1, got {r2}'


def check_record_ids():
    for tool in ['oligoformer', 'sirnabert']:
        module = load_script(tool, 'prepare')
        with tempfile.TemporaryDirectory() as folder:
            input_path = Path(folder) / 'input.csv'
            pd.DataFrame({'record_id': ['sb_a', 'sb_b'], 'siRNA': ['A'*19, 'C'*19],
                          'extended_mRNA': ['U'*57, 'G'*57], 'efficiency': [0.1, 0.8]}).to_csv(input_path, index=False)
            argv = ['prepare.py', '--input-csv', str(input_path), '--output-dir', folder, '--dataset-name', 'sample']
            with patch.object(sys, 'argv', argv):
                if tool == 'oligoformer':
                    with patch.object(module, 'rnafm_ready', return_value=True):
                        module.main()
                else:
                    module.main()
            actual = pd.read_csv(Path(folder) / 'sample.csv')['id'].tolist()
            assert actual == ['sb_a', 'sb_b'], (tool, actual)



def check_common_patience():
    module = load_script('oligoformer', 'train')
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'input.csv'
        pd.DataFrame({'label': [0.1, 0.8], 'y': [0, 1]}).to_csv(path, index=False)
        argv = ['train.py', '--train-csv', str(path), '--val-csv', str(path),
                '--model-dir', str(Path(folder)/'models'), '--epochs', '6',
                '--early-stopping', '2', '--early-stop-metric', 'r2']
        with patch.object(sys, 'argv', argv), \
             patch.object(module, 'load_modules', return_value=(lambda *a: [0, 1], lambda: torch.nn.Linear(1, 1))), \
             patch.object(module, 'train_epoch', return_value=0.0), \
             patch.object(module, 'eval_epoch', return_value=(1.0, 0.5, 0.1)) as evaluate:
            module.main()
        # Initial best epoch, then exactly two non-improving epochs.
        assert evaluate.call_count == 3, evaluate.call_count

if __name__ == '__main__':
    check_efficacy_r2()
    check_record_ids()
    check_common_patience()
    print('OligoFormer efficacy R2 and OligoFormer/BERT record-ID checks passed.')
