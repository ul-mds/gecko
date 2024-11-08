import numpy as np
import pandas as pd

import typing_extensions as _te

_UINT_CAPACITY = 64


def with_capacity(rows: int, capacity: int):
    if rows <= 0:
        raise ValueError(f"number of rows must be positive, is {rows}")

    if capacity <= 0:
        raise ValueError(f"capacity must be positive, is {capacity}")

    col_idx, _ = divmod(capacity - 1, _UINT_CAPACITY)
    return pd.DataFrame(np.zeros((rows, col_idx + 1), dtype=np.uint64))


def set_index(df: pd.DataFrame, mask: _te.Union[pd.Series, slice], idx: int):
    col_idx, int_idx = divmod(idx, _UINT_CAPACITY)
    df.loc[mask, col_idx] |= 1 << int_idx


def test_index(df: pd.DataFrame, idx: int) -> pd.Series:
    col_idx, int_idx = divmod(idx, _UINT_CAPACITY)
    return (df.loc[:, col_idx] & (1 << int_idx)) != 0


def any_set(df: pd.DataFrame) -> pd.Series:
    return (df != 0).any(axis=1)
