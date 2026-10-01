"""Reload one sealed TabPFN fit and predict one frozen benchmark cohort."""
import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

from run_benchmark import load_frame
from run_validation import (PROTOCOL_SHA, FEATURE_MANIFEST_SHA, checked_bytes,
                            representation, sha256, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input', type=Path, help='CSV with record_id, guide and the 100 precomputed manifest features')
    inputs.add_argument('--partition',
                        help='Manifest key, e.g. grouped/fold_0/val.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    sys.path.insert(0, str(args.root/'siRBench/benchmark/revision'))
    from baselines import sequence_features
    import torch
    from tabpfn.model_loading import load_fitted_tabpfn_model
    metadata = json.loads((args.model_dir/'train_meta.json').read_text())
    model_path = args.model_dir/'model.tabpfn_fit'
    if sha256(model_path) != metadata['fitted_model_sha256']:
        raise ValueError('Fitted-model checksum mismatch')
    columns = json.loads(checked_bytes(args.root/'datasets/corrected-v1/records_features.manifest.json',
                                      FEATURE_MANIFEST_SHA))['features']
    if args.partition:
        protocol = args.root/'evaluation/protocol-v1'
        manifest = json.loads(checked_bytes(protocol/'manifest.json', PROTOCOL_SHA))
        frame = load_frame(protocol, manifest, args.partition)
        input_sha = manifest['outputs'][args.partition]
    else:
        frame = pd.read_csv(args.input)
        input_sha = sha256(args.input)
    if not len(frame) or frame.record_id.isna().any() or not frame.record_id.is_unique:
        raise ValueError('Input requires nonempty unique record IDs')
    features = representation(frame, columns, sequence_features)
    seed = metadata['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    model = load_fitted_tabpfn_model(model_path, device='cuda')
    prediction = np.asarray(model.predict(features), float)
    if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
        raise ValueError('Invalid prediction shape or values')
    args.output.mkdir(parents=True)
    pd.DataFrame({'record_id': frame.record_id, 'pred_label': prediction}).to_csv(
        args.output/'predictions.csv', index=False)
    report = {'model_sha256': sha256(model_path), 'partition': args.partition,
              'input_sha256': input_sha, 'seed': seed,
              'rows': len(frame), 'feature_count': features.shape[1],
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__}
    if args.evaluate:
        from evaluate_predictions import metrics
        report['metrics'] = metrics(frame.efficiency.to_numpy(float), prediction)
    write_json(args.output/'report.json', report)


if __name__ == '__main__':
    main()
