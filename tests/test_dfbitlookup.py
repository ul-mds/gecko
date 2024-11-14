import numpy as np
import pytest
import pandas as pd

from gecko import _dfbitlookup


@pytest.mark.parametrize(
    "rows,capacity,expected_columns",
    [(1, 1, 1), (10, 32, 1), (10, 64, 1), (100, 65, 2), (100, 128, 2), (100, 129, 3)],
)
def test_with_capacity(rows: int, capacity: int, expected_columns: int):
    df = _dfbitlookup.with_capacity(rows, capacity)

    assert len(df) == rows
    assert len(df.columns) == expected_columns
    assert (df == 0).all().all()  # double all() to cover all axes


def test_with_capacity_raise_rows_too_low():
    with pytest.raises(ValueError) as e:
        _dfbitlookup.with_capacity(0, 1)

    assert str(e.value) == "number of rows must be positive, is 0"


def test_with_capacity_raise_capacity_too_low():
    with pytest.raises(ValueError) as e:
        _dfbitlookup.with_capacity(1, 0)

    assert str(e.value) == "capacity must be positive, is 0"


@pytest.mark.parametrize(
    "rows,capacity,mask,index",
    [
        (100, 64, pd.Series([True] * 20 + [False] * 80), 63),
        (100, 128, pd.Series([False] * 20 + [True] * 60 + [False] * 20), 64),
    ],
)
def test_set_test_index(rows: int, capacity: int, mask: pd.Series, index: int):
    df = _dfbitlookup.with_capacity(rows, capacity)
    _dfbitlookup.set_index(df, mask, index)

    assert (_dfbitlookup.test_index(df, index) == mask).all()


def test_count_bits_per_index():
    df = pd.DataFrame([0b1001, 0b0101, 0b1101], dtype=np.uint64)

    # [0] =>  1  0  0  1
    # [1] =>  1  0  1  0
    # [2] =>  1  0  1  1
    #        [a][b][c][d]  =>  [a] = 3, [b] = 0, [c] = 2, [d] = 2

    expected = [(0, 3), (1, 0), (2, 2), (3, 2)]
    assert _dfbitlookup.count_bits_per_index(df, 4) == expected


def test_count_bits_per_index_infer_capacity():
    df = pd.DataFrame([0b1001, 0b0101, 0b1101], dtype=np.uint64)

    # indices 4-63 should be zero
    expected = [(0, 3), (1, 0), (2, 2), (3, 2)] + [(idx, 0) for idx in range(4, 64)]
    assert _dfbitlookup.count_bits_per_index(df) == expected


def test_count_bits_per_index_raise_capacity_too_low():
    df = pd.DataFrame([0b1001, 0b0101, 0b1101], dtype=np.uint64)

    with pytest.raises(ValueError) as e:
        _ = _dfbitlookup.count_bits_per_index(df, 0)

    assert str(e.value) == "capacity must be positive, is 0"


def test_count_bits_per_index_raise_capacity_too_high():
    df = pd.DataFrame([0b1001, 0b0101, 0b1101], dtype=np.uint64)

    with pytest.raises(ValueError) as e:
        _ = _dfbitlookup.count_bits_per_index(df, 65)

    assert str(e.value) == "capacity must not be higher than 64, is 65"
