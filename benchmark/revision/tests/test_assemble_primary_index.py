import itertools
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from assemble_primary_index import TOOLS, KEY, complete_index


def fixture(tmp_path):
    frame = pd.DataFrame(itertools.product(TOOLS,['grouped','random'],range(5),range(3)),columns=KEY)
    frame['status'] = 'verified'
    for row in frame.itertuples():
        directory = tmp_path / f'{row.tool}_{row.axis}_{row.fold}_{row.training_seed}'
        directory.mkdir()
        for column in ['test_predictions','hela_predictions','train_meta']:
            artifact = directory / column
            metadata = {'config':{'seed':row.training_seed}} if row.tool == 'ensirna' else {'seed':row.training_seed}
            artifact.write_text(json.dumps(metadata) if column == 'train_meta' else 'id,label,pred_label\nx,0.1,0.2\n')
            frame.loc[row.Index,column] = str(artifact)
    return frame


def test_sharded_indexes_require_every_method_and_replicate(tmp_path):
    frame = fixture(tmp_path)
    actual = complete_index([frame.iloc[:60],frame.iloc[60:]])
    assert len(actual) == 180
    assert actual.train_meta_sha256.str.fullmatch('[0-9a-f]{64}').all()
    with pytest.raises(ValueError,match='all180'):
        complete_index([frame.iloc[:-1]])
    with pytest.raises(ValueError,match='Duplicate'):
        complete_index([frame,frame.iloc[:1]])


def test_unverified_or_missing_metadata_cannot_enter_primary_analysis(tmp_path):
    frame = fixture(tmp_path)
    frame.loc[0,'status'] = 'running'
    with pytest.raises(ValueError,match='all180'):
        complete_index([frame])
    frame.loc[0,'status'] = 'verified'
    with pytest.raises(ValueError,match='Missing primary artifact column'):
        complete_index([frame.drop(columns='train_meta')])


def test_distinct_runs_cannot_reuse_artifact_paths(tmp_path):
    frame = fixture(tmp_path)
    frame.loc[1,'test_predictions'] = frame.loc[0,'test_predictions']
    with pytest.raises(ValueError,match='reused across run identities'):
        complete_index([frame])


@pytest.mark.parametrize('metadata', [{'seed':9}, {'config':{'seed':9}}, {'seed':0,'configuration':{'seed':9}}, {}])
def test_actual_training_metadata_must_match_seed_identity(tmp_path,metadata):
    frame = fixture(tmp_path)
    Path(frame.loc[0,'train_meta']).write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='[Tt]raining seed'):
        complete_index([frame])


def test_sealed_artifacts_cannot_change_before_assembly(tmp_path):
    index = complete_index([fixture(tmp_path)])
    Path(index.loc[0,'test_predictions']).write_text('changed\n')
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        complete_index([index])


def test_central_ensirna_completion_state_is_normalized_after_validation(tmp_path):
    frame = fixture(tmp_path)
    frame.loc[frame.tool.eq('ensirna'),'status'] = 'complete'
    actual = complete_index([frame])
    assert actual.status.eq('verified').all()
    assert actual.loc[actual.tool.eq('ensirna'),'source_status'].eq('complete').all()
    frame.loc[frame.tool.eq('oligoformer'),'status'] = 'complete'
    with pytest.raises(ValueError,match='all180'):
        complete_index([frame])


def test_ensirna_complete_still_requires_valid_artifacts(tmp_path):
    frame = fixture(tmp_path)
    frame.loc[frame.tool.eq('ensirna'),'status'] = 'complete'
    row = frame.loc[frame.tool.eq('ensirna')].iloc[0]
    Path(row.train_meta).write_text('{"config":{"seed":99}}')
    with pytest.raises(ValueError,match='Training seed differs'):
        complete_index([frame])
