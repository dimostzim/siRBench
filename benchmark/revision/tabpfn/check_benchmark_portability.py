"""Independent same-host and returned-worker validation-only fitted-state reloads."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import socket
import sys

import numpy as np
import pandas as pd

from run_validation import (ROOT, PROTOCOL_SHA, FEATURE_MANIFEST_SHA, checked_bytes,
                            representation, sha256, write_json)
from run_benchmark import load_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.root/'siRBench/benchmark/revision'))
    from baselines import sequence_features
    from evaluate_predictions import metrics
    import torch
    from tabpfn.model_loading import load_fitted_tabpfn_model
    torch.set_num_threads(4)
    protocol = args.root/'evaluation/protocol-v1'
    manifest = json.loads(checked_bytes(protocol/'manifest.json', PROTOCOL_SHA))
    feature_path = args.root/'datasets/corrected-v1/records_features.manifest.json'
    columns = json.loads(checked_bytes(feature_path, FEATURE_MANIFEST_SHA))['features']
    validation = load_frame(protocol, manifest, 'grouped/fold_0/val.csv')
    features = representation(validation, columns, sequence_features)
    rows = []
    for seed in [0, 1]:
        attempt = args.root/f'evaluation/tabpfn-benchmark-v1/runs/grouped/fold_0/seed_{seed}/attempt_2'
        for tool in ['tabpfn35_frozen', 'tabpfn35_finetuned']:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            directory = attempt/tool
            metadata = json.loads((directory/'train_meta.json').read_text())
            expected = pd.read_csv(directory/'validation_predictions.csv')
            if expected.record_id.tolist() != validation.record_id.tolist():
                raise ValueError('Validation IDs differ from frozen input order')
            if sha256(directory/'model.tabpfn_fit') != metadata['fitted_model_sha256']:
                raise ValueError('Fitted-state hash mismatch')
            model = load_fitted_tabpfn_model(directory/'model.tabpfn_fit', device='cuda')
            prediction = np.asarray(model.predict(features), float)
            if prediction.shape != (len(validation),) or not np.isfinite(prediction).all():
                raise ValueError('Invalid reload predictions')
            absolute = np.abs(prediction-expected.pred_label.to_numpy())
            original_metrics = metrics(validation.efficiency.to_numpy(), expected.pred_label.to_numpy())
            reload_metrics = metrics(validation.efficiency.to_numpy(), prediction)
            output = args.output/f'{tool}_grouped0_seed{seed}_validation_predictions.csv'
            pd.DataFrame({'record_id': validation.record_id, 'label': validation.efficiency,
                          'original_prediction': expected.pred_label, 'reloaded_prediction': prediction}).to_csv(output, index=False)
            row = {'tool': tool, 'axis': 'grouped', 'fold': 0, 'seed': seed,
                   'source_host': metadata['host'], 'source_gpu': metadata['gpu'],
                   'reload_host': socket.gethostname(), 'reload_gpu': torch.cuda.get_device_name(),
                   'same_host': metadata['host'] == socket.gethostname(), 'records': len(validation),
                   'max_absolute_difference': float(absolute.max()),
                   'median_absolute_difference': float(np.median(absolute)),
                   'mean_absolute_difference': float(absolute.mean()),
                   'allclose_rtol1e-5_atol1e-6': bool(np.allclose(prediction, expected.pred_label, rtol=1e-5, atol=1e-6)),
                   'original_metrics': original_metrics, 'reloaded_metrics': reload_metrics,
                   'metric_delta_reload_minus_original': {key: reload_metrics[key]-value for key, value in original_metrics.items()},
                   'prediction_comparison_sha256': sha256(output),
                   'fitted_model_sha256': sha256(directory/'model.tabpfn_fit'),
                   'original_validation_prediction_sha256': sha256(directory/'validation_predictions.csv'),
                   'inference_precision_setting': str(model.inference_precision),
                   'runtime': {'python_numpy_torch_seed_reset': seed,
                               'torch_threads': torch.get_num_threads(),
                               'thread_environment': {name: os.environ.get(name) for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
                               'float32_matmul_precision': torch.get_float32_matmul_precision(),
                               'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                               'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32}}
            rows.append(row)
            print(json.dumps(row), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    write_json(args.output/'report.json', {'scope': 'Fresh-process same-node and returned-worker validation-only reloads; no training, held-out rescoring or policy changes.',
               'rows': rows, 'code_sha256': sha256(Path(__file__)),
               'originals_preserved': True,
               'interpretation': 'Same-run same-device acceptance checks and cross-device portability are different tests; report measured numerical differences.'})


if __name__ == '__main__':
    main()
