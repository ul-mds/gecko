import numpy as np
import pandas as pd

import typing_extensions as _te

_UINT_CAPACITY = 64


def _pos(idx: int):
    return divmod(idx - 1, _UINT_CAPACITY)


def with_capacity(rows: int, capacity: int):
    if rows <= 0:
        raise ValueError(f"number of rows must be positive, is {rows}")

    if capacity <= 0:
        raise ValueError(f"capacity must be positive, is {capacity}")

    col_idx, _ = _pos(capacity)
    return pd.DataFrame(np.zeros((rows, col_idx + 1), dtype=np.uint64))


def set_index(df: pd.DataFrame, mask: _te.Union[pd.Series, slice], idx: int):
    col_idx, int_idx = _pos(idx)
    df.loc[mask, col_idx] |= 1 << int_idx


def test_index(df: pd.DataFrame, idx: int) -> pd.Series:
    col_idx, int_idx = _pos(idx)
    return (df.loc[:, col_idx] & (1 << int_idx)) != 0
