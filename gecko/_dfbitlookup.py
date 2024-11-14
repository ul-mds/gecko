"""
The dfbitlookup module provides functions for supporting bit tests on data frames.
A data frame returned by this module's `with_capacity` function contains series of 64-bit integers.
The amount of series and rows are decided by the user's requirements.
This data frame can then be used to set and perform bit tests across all of its rows at once.
"""

import numpy as np
import pandas as pd

import typing as _t
import typing_extensions as _te

_UINT_CAPACITY = 64


def with_capacity(rows: int, capacity: int, index: _t.Optional[pd.Index] = None):
    """
    Construct a new data frame that is capable of storing the desired capacity of bits for the desired
    number of rows.
    An index object can be supplied to align the data frame returned by this function with another
    Pandas object.

    Args:
        rows: amount of rows
        capacity: maximum amount of bits
        index: index to align to

    Returns:
        Data frame to set and test bits on across all of its rows
    """
    if rows <= 0:
        raise ValueError(f"number of rows must be positive, is {rows}")

    if capacity <= 0:
        raise ValueError(f"capacity must be positive, is {capacity}")

    col_idx = (capacity - 1) // _UINT_CAPACITY
    return pd.DataFrame(np.zeros((rows, col_idx + 1), dtype=np.uint64), index=index)


def set_index(df: pd.DataFrame, mask: _te.Union[pd.Series, slice], idx: int):
    """
    Set a bit on all selected rows at the specified index.

    Args:
        df: data frame to perform operation on
        mask: series or list of booleans to select rows to set bit in with
        idx: index of the bit to set
    """
    col_idx, int_idx = divmod(idx, _UINT_CAPACITY)
    df.loc[mask, col_idx] |= 1 << int_idx


def test_index(df: pd.DataFrame, idx: int) -> pd.Series:
    """
    Test a bit on all rows at the specified index.

    Args:
        df: data frame to perform operation on
        idx: index of the bit to test

    Returns:
        series of booleans representing rows where the selected bit is set
    """
    col_idx, int_idx = divmod(idx, _UINT_CAPACITY)
    return (df.loc[:, col_idx] & (1 << int_idx)) != 0


def any_set(df: pd.DataFrame) -> pd.Series:
    """
    Test whether any bits are set for each row.

    Args:
        df: data frame to perform operation on

    Returns:
        series of booleans representing rows where any bit is set
    """
    return (df != 0).any(axis=1)
