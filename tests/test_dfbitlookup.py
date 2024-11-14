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
