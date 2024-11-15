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


def _divmod(num: int, div: int):
    """
    Alternative for the built-in divmod function which returns ints instead of floats.

    Args:
        num: dividend
        div: divisor

    Returns:
        quotient and remainder as integers
    """
    a, b = divmod(num, div)
    return int(a), int(b)


def set_index(df: pd.DataFrame, mask: _te.Union[pd.Series, slice, list[bool]], idx: int):
    """
    Set a bit on all selected rows at the specified index.

    Args:
        df: data frame to perform operation on
        mask: series or list of booleans to select rows to set bit in with
        idx: index of the bit to set
    """
    col_idx, int_idx = _divmod(idx, _UINT_CAPACITY)
    df.loc[mask, col_idx] |= np.left_shift(1, int_idx, dtype=np.uint64)


def test_index(df: pd.DataFrame, idx: int) -> pd.Series:
    """
    Test a bit on all rows at the specified index.

    Args:
        df: data frame to perform operation on
        idx: index of the bit to test

    Returns:
        series of booleans representing rows where the selected bit is set
    """
    col_idx, int_idx = _divmod(idx, _UINT_CAPACITY)
    return (df.loc[:, col_idx] & np.left_shift(1, int_idx, dtype=np.uint64)) != 0


def any_set(df: pd.DataFrame) -> pd.Series:
    """
    Test whether any bits are set for each row.

    Args:
        df: data frame to perform operation on

    Returns:
        series of booleans representing rows where any bit is set
    """
    return (df != 0).any(axis=1)


def count_bits_per_index(df: pd.DataFrame, capacity: _t.Optional[int] = None) -> list[tuple[int, int]]:
    """
    Count the bits set for each index across all rows.
    If provided, this function will use the capacity argument as the upper bound for indices to count.
    If not provided, it will be inferred from the number of columns in the data frame.

    Args:
        df: data frame to perform operation on
        capacity: maximum amount of bits to test

    Returns:
        list of tuples where the first int representing the index and the second int representing the number of set bits
    """
    max_df_capacity = int(_UINT_CAPACITY * len(df.columns))

    if capacity is None:
        capacity = max_df_capacity
    else:
        capacity = int(capacity)

        if capacity <= 0:
            raise ValueError(f"capacity must be positive, is {capacity}")

        if capacity > max_df_capacity:
            raise ValueError(f"capacity must not be higher than {max_df_capacity}, is {capacity}")

    return list((idx, test_index(df, idx).sum()) for idx in range(capacity))
