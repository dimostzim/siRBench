import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[2]/'competitors/tools/sirnadiscovery/scripts/fetch_rpiseq.py'
spec=importlib.util.spec_from_file_location('fetch_rpiseq',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_extracts_rf_not_svm_from_response():
    html='<table><tr><td>&gt;rna_1</td><td>0.75</td><td>0.95</td></tr></table>'
    assert module.parse_probabilities(html,['rna_1']) == {'rna_1':.75}


@pytest.mark.parametrize('html', ['<html>Service error</html>', '<tr><td>&gt;wrong_id</td><td>.7</td><td>.9</td></tr>', '<tr><td>&gt;rna_1</td><td>NaN</td><td>.9</td></tr>'])
def test_rejects_incomplete_or_invalid_response(html):
    with pytest.raises(ValueError):
        module.parse_probabilities(html,['rna_1'])
