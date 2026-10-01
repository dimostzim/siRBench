"""Run inside attsioff:revision to check feature isolation and clipping."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch

path = Path(__file__).resolve().parents[2] / 'competitors/tools/attsioff/train.py'
spec = importlib.util.spec_from_file_location('attsioff_train', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

frame = pd.DataFrame({'Antisense':['A'*19,'C'*19], 'mrna':['U'*57]*2,
                      'inhibition':[.4,.6], 'RNAFM_ind':['a','b']})
training_pssm = np.full((4,19),.25)
def must_not_fit(sequences):
    raise AssertionError('Evaluation sequences must not refit the training PSSM')
data, pssm = module.build_dataset(frame, must_not_fit, training_pssm)
assert pssm is training_pssm
assert data['seq'].tolist() == ['A'*19,'C'*19]

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
    def forward(self, batch):
        return self.weight * batch['x']
model=Model()
optimizer=torch.optim.Adam(model.parameters(), lr=.005)
loader=[{'x':torch.tensor([100.,100.]),'inhibit':torch.tensor([0.,0.])}]
module.train_epoch(model,loader,optimizer,torch.nn.MSELoss(),'cpu')
assert 4.99 <= model.weight.grad.abs().item() <= 5.01
print('Passed: evaluation PSSM reuse; actual optimizer-path gradient clipping.')
