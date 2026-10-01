"""Self-contained three-predictor inference; no labels or fitted query statistics."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '16'
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from sequence_model import device_setup, predict
from architecture import make_sequence_model, make_thermodynamic_model, combine_predictions
from features import read_inputs, transform, ThermoScaler
from calibration import apply_affine

DEFAULT_ARTIFACTS = Path(__file__).resolve().parent / 'training_artifacts'

def run(frame, artifacts=DEFAULT_ARTIFACTS):
    if not len(frame):
        return np.empty(0, dtype=np.float64)
    artifacts = Path(artifacts)
    device = device_setup()
    scaler = ThermoScaler.load(artifacts / 'scaler.npz')
    calibration = json.loads((artifacts / 'calibration.json').read_text())
    models = []
    for name, factory in [('sequence', make_sequence_model), ('thermo', make_thermodynamic_model)]:
        model = factory(0)
        model.load_state_dict(torch.load(artifacts / (name+'.pt'), map_location='cpu', weights_only=True))
        models.append(model.to(device).eval())
    tree = CatBoostRegressor(thread_count=16)
    tree.load_model(str(artifacts / 'tree.cbm'))
    result = []
    with torch.inference_mode():
        for start in range(0, len(frame), 256):
            x = transform(frame.iloc[start:start+256], scaler)
            predictions = []
            for model, key in zip(models, ['sequence', 'thermo_cnn']):
                predictions.append(predict(model, [torch.from_numpy(v) for v in x[key].values()], device))
            predictions.append(tree.predict(x['tree'], thread_count=16))
            result.append(apply_affine(combine_predictions(*predictions), calibration))
    result = np.concatenate(result)
    if result.shape != (len(frame),) or not np.isfinite(result).all():
        raise RuntimeError('Incomplete or non-finite predictions')
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--artifacts-dir', default=str(DEFAULT_ARTIFACTS))
    args = parser.parse_args()
    frame = read_inputs(args.input)
    predictions = run(frame, args.artifacts_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'id':frame['id'].to_numpy(), 'prediction':predictions}).to_csv(output, index=False)

if __name__ == '__main__':
    main()
