import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filter_attsioff_compatibility import compatible_extents


def test_filter_preserves_order_and_values_without_relabeling():
    extents = pd.DataFrame({'record_id': ['c', 'a', 'b'], 'efficiency': [.4, .8, .3]})
    audit = pd.DataFrame({'record_id': ['a', 'b', 'c'], 'status': [
        'source_and_scaled_label_compatible', 'unresolved',
        'source_and_scaled_label_compatible']})
    pd.testing.assert_frame_equal(compatible_extents(extents, audit), extents.iloc[:2])
    with pytest.raises(ValueError, match='exactly one'):
        compatible_extents(extents, audit.iloc[:2])
    with pytest.raises(ValueError, match='unique'):
        compatible_extents(extents, pd.concat([audit, audit.iloc[:1]]))
