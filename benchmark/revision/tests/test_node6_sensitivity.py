import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_node6_sensitivity import execute_case
from run_attsioff_discovery_matrix import digest


def test_complete_run_resumes_only_matching_checked_outputs(tmp_path):
    specification = dict(tool='attsioff', seed=2, run_relative='runs/one', assets_relative='assets/one')
    run = tmp_path / specification['run_relative']
    model = run / 'models/attsioff'
    model.mkdir(parents=True)
    (model / 'train_meta.json').write_text(json.dumps({'seed': 2}))
    (run / 'sensitivity_specification.json').write_text(json.dumps(specification))
    output = run / 'prediction.csv'
    output.write_text('id,label,pred_label\na,0.1,0.2\n')
    (run / 'SENSITIVITY_COMPLETE.json').write_text(json.dumps({'prediction.csv': digest(output)}))
    execute_case(tmp_path, specification)
    output.write_text('changed')
    with pytest.raises(ValueError, match='checksum differs'):
        execute_case(tmp_path, specification)


def test_wrong_sequence_asset_is_rejected_before_training(tmp_path):
    specification = dict(tool='attsioff', seed=0, run_relative='runs/one', assets_relative='assets/one')
    assets = tmp_path / 'assets/one'
    inputs = tmp_path / 'runs/one/inputs'
    assets.mkdir(parents=True)
    inputs.mkdir(parents=True)
    (assets / 'all.csv').write_text('id,Antisense,mrna\nsb_1,AAA,CCC\n')
    (assets / 'input.csv').write_text('record_id,siRNA,extended_mRNA\nsb_1,AAA,CCC\n')
    (inputs / 'train.csv').write_text('record_id,siRNA,extended_mRNA\nsb_1,AAA,CCG\n')
    with pytest.raises(ValueError, match='Feature asset sequence differs'):
        execute_case(tmp_path, specification)
