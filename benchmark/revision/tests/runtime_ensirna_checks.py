"""Run inside ensirna:revision; uses actual PyTorch, no pretrained model import."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2] / 'competitors/tools/ensirna'
sys.path.insert(0, str(ROOT / 'ensirna_src/ENsiRNA'))
from data.embedding_utils import pad_mrna_embedding
from trainer.RNAmaskModel_trainer import RNAmaskModelTrainer
from trainer.abs_trainer import Trainer

# Distinct sentinels represent BOS, three mRNA tokens, two antisense tokens.
embedding = torch.tensor([[99.], [1.], [2.], [3.], [7.], [8.]])
padded = pad_mrna_embedding(embedding, 3, 2, 1)
assert padded[:, 0].tolist() == [99., 0., 0., 1., 2., 3., 0., 7., 8.]
assert pad_mrna_embedding(embedding, 3, 0, 0).equal(embedding)

for maximize in (False, True):
    trainer = RNAmaskModelTrainer.__new__(RNAmaskModelTrainer)
    trainer.log_alpha = -0.001
    trainer.config = SimpleNamespace(metric_min_better=not maximize)
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    scheduler = trainer.get_scheduler(optimizer)['scheduler']
    scheduler.step(0.5)
    scheduler.step(0.7 if maximize else 0.3)
    assert scheduler.num_bad_epochs == 0
    assert scheduler.mode == ('max' if maximize else 'min')

trainer = Trainer.__new__(Trainer)
trainer.config = SimpleNamespace(metric_min_better=False, legacy_stopping=False)
trainer.best_valid_metric = .8
trainer.last_valid_metric = .5
assert not trainer._metric_better(.6)
assert trainer._metric_better(.9)
trainer.config.legacy_stopping = True
assert trainer._metric_better(.6)
print('ENsiRNA embedding alignment, scheduler direction, and global-best stopping checks passed')

import subprocess
import tempfile
from unittest.mock import patch
from data.get_pdb import Data_Prepare

with tempfile.TemporaryDirectory() as directory:
    prepare = Data_Prepare.__new__(Data_Prepare)
    prepare.pdb_dir, prepare.ff, prepare.ex = directory, 'rosetta', 'extract.py'
    prepare.database, prepare.rosetta_dir = 'database', 'rosetta_dir'

    def fail_folding(command, **kwargs):
        if command[0] == 'RNAplex':
            return SimpleNamespace(stdout='((((&)))) 1,4 : 1,4 (-4.0)')
        raise subprocess.CalledProcessError(1, command)

    with patch('data.get_pdb.subprocess.run', side_effect=fail_folding):
        try:
            prepare.get_secondary_structure({'sense seq': 'acgu', 'anti seq': 'acgu', 'siRNA': 'record'})
        except subprocess.CalledProcessError:
            pass
        else:
            raise AssertionError('Rosetta failure was swallowed')
    assert list(Path(directory).iterdir()) == []
print('ENsiRNA failed-folding atomicity check passed')


class FixedPredictions(torch.nn.Module):
    def test(self, pct, prediction):
        return prediction, None, None, None


trainer = Trainer.__new__(Trainer)
trainer.model = FixedPredictions()
trainer.config = SimpleNamespace(val_metric='r2', metric_min_better=False,
                                 legacy_stopping=False, patience=20)
trainer.valid_loader = [
    {'pct': torch.tensor([0., .2, .4, .6]), 'prediction': torch.tensor([0., .1, .4, .6])},
    {'pct': torch.tensor([.8]), 'prediction': torch.tensor([.2])},
]
trainer.valid_global_step = 0
trainer.best_valid_metric = None
trainer.last_valid_metric = None
trainer.writer_buffer = {}
trainer.epoch = 0
trainer.to_device = lambda batch, device: batch
trainer._is_main_proc = lambda: False
seen_metrics = []
trainer.scheduler = SimpleNamespace(step=seen_metrics.append)
# Five observations have SSE .37 and SST .4; averaging batch R² is different.
metric = trainer._valid_epoch(torch.device('cpu'))
assert abs(metric - .075) < 1e-6
assert seen_metrics == [metric]
assert trainer.best_valid_metric == metric
print('ENsiRNA unequal-batch global validation R² check passed')
