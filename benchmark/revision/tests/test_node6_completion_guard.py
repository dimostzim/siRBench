import importlib.util
import json
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'run_attsioff_discovery_matrix.py'
spec=importlib.util.spec_from_file_location('node6_matrix',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_fresh_run_is_allowed_but_partial_checkpoint_is_rejected(tmp_path):
    module.check_model_completion(tmp_path,'attsioff',0)
    model=tmp_path/'models/attsioff'
    model.mkdir(parents=True)
    (model/'model.pt').write_bytes(b'partial')
    with pytest.raises(ValueError,match='completion metadata'):
        module.check_model_completion(tmp_path,'attsioff',0)


def test_completion_requires_metadata_with_matching_seed(tmp_path):
    with pytest.raises(ValueError,match='completion metadata'):
        module.check_model_completion(tmp_path,'attsioff',0,required=True)
    model=tmp_path/'models/attsioff'
    model.mkdir(parents=True)
    (model/'train_meta.json').write_text(json.dumps({'seed':0}))
    module.check_model_completion(tmp_path,'attsioff',0,required=True)
    with pytest.raises(ValueError,match='seed differs'):
        module.check_model_completion(tmp_path,'attsioff',1,required=True)
