from os import PathLike
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from numpy.random import Generator
from typing_extensions import ParamSpec  # required for 3.9 backport

P = ParamSpec("P")
NumericType = Union[float, int]

GeneratorFunc = Callable[[int], list[pd.Series]]


def from_function(func: Callable[P, str], *args, **kwargs) -> GeneratorFunc:
    """
    Generate a series from an arbitrary function that returns a single value at a time.
    This generator should be used sparingly as it is not vectorized, meaning values have to be generated one by one.
    Use this generator for testing purposes or if performance is not critical.

    :param func: function that takes in any arguments and returns a single string
    :return: function returning a Pandas series with values generated from the custom function
    """

    def _generate(count: int) -> list[pd.Series]:
        return [pd.Series(data=[func(*args, **kwargs) for _ in np.arange(count)])]

    return _generate


def _generate_from_uniform_distribution_float(
    low: float,
    high: float,
    rng: Generator,
) -> GeneratorFunc:
    """Generate a series of floats and format them as strings."""

    def _generate(count: int) -> list[pd.Series]:
        return [pd.Series(np.char.mod("%f", rng.uniform(low, high, count)))]

    return _generate


def _generate_from_uniform_distribution_int(
    low: int,
    high: int,
    rng: Generator,
) -> GeneratorFunc:
    """Generate a series of integers and format them as strings."""

    def _generate(count: int) -> list[pd.Series]:
        return [pd.Series(np.char.mod("%d", rng.integers(low, high, size=count)))]

    return _generate


def from_uniform_distribution(
    low: NumericType = 0,
    high: NumericType = 1,
    dtype: type[Union[int, float]] = float,
    rng: Optional[Generator] = None,
) -> GeneratorFunc:
    """
    Generate a series of numbers drawn from a uniform distribution within the specified bounds.
    These numbers are formatted into strings.

    :param low: lower (inclusive) bound of the uniform distribution (default: `0`)
    :param high: upper (exclusive) bound of the uniform distribution (default: `1`)
    :param dtype: int or float (default: `float`)
    :param rng: random number generator to use (default: `None`)
    :return: function returning Pandas series of numbers from uniform distribution with specified parameters
    """
    if rng is None:
        rng = np.random.default_rng()

    if dtype is float:
        return _generate_from_uniform_distribution_float(low, high, rng)

    if dtype is int:
        return _generate_from_uniform_distribution_int(low, high, rng)

    raise NotImplementedError(f"unexpected data type: {dtype}")


def from_normal_distribution(
    mean: float = 0,
    sd: float = 1,
    rng: Optional[Generator] = None,
) -> GeneratorFunc:
    """
    Generate a series of numbers drawn from a normal distribution with the specified parameters.
    These numbers are formatted into strings.

    :param mean: mean of the normal distribution (default: `0`)
    :param sd: standard deviation of the normal distribution (default: `1`)
    :param rng: random number generator to use (default: `None`)
    :return: function returning Pandas series of numbers from normal distribution with specified parameters
    """
    # TODO add option to return ints as well
    if rng is None:
        rng = np.random.default_rng()

    def _generate(count: int) -> list[pd.Series]:
        return [pd.Series(np.char.mod("%f", rng.normal(mean, sd, count)))]

    return _generate


def from_frequency_table(
    csv_file_path: Union[str, PathLike[str]],
    header: bool = False,
    value_column: Union[str, int] = 0,
    freq_column: Union[str, int] = 1,
    encoding: str = "utf-8",
    delimiter: str = ",",
    rng: Optional[Generator] = None,
) -> GeneratorFunc:
    """
    Generate a series of values from a CSV file.
    This CSV file must contain at least two columns, one holding values and one holding their absolute frequencies.
    Values are generated using their assigned absolute frequencies.
    Therefore, the values in the resulting series should have a similar distribution compared to the input file.

    :param csv_file_path: CSV file to read from
    :param header: `True` if the file contains a header, `False` otherwise (default: `False`)
    :param value_column: name of the value column if the file contains a header, otherwise the column index (default: `0`)
    :param freq_column: name of the frequency column if the file contains a header, otherwise the column index (default: `1`)
    :param encoding: character encoding of the CSV file (default: `utf-8`)
    :param delimiter: column delimiter (default: `,`)
    :param rng: random number generator to use (default: `None`)
    :return: function returning Pandas series of values with a distribution similar to that of the input file
    """
    if rng is None:
        rng = np.random.default_rng()

    if type(value_column) is not type(freq_column):
        raise ValueError("value and frequency column must both be of the same type")

    # read csv file
    df = pd.read_csv(
        csv_file_path,
        header=0 if header else None,  # header row index (`None` if not present)
        usecols=[value_column, freq_column],
        dtype={freq_column: "int"},
        sep=delimiter,
        encoding=encoding,
    )

    # convert absolute to relative frequencies
    srs_value = df[value_column]
    srs_prob = df[freq_column] / df[freq_column].sum()

    def _generate(count: int) -> list[pd.Series]:
        return [pd.Series(rng.choice(srs_value, count, p=srs_prob))]

    return _generate


def from_multicolumn_frequency_table(
    csv_file_path: Union[str, PathLike[str]],
    header: bool = False,
    value_columns: Union[int, str, list[int], list[str]] = 0,
    freq_column: Union[int, str] = 1,
    encoding: str = "utf-8",
    delimiter: str = ",",
    rng: Optional[Generator] = None,
) -> GeneratorFunc:
    """
    Generate a series of values from a CSV file where columns are inter-dependent.
    This CSV file must contain at least two columns, one holding values and one holding their absolute frequencies.
    This corruptor generates a series for each value column that's passed into it.
    Values are generated using their assigned absolute frequencies.
    Therefore, the values in the resulting series should have a similar distribution compared to the input file.

    :param csv_file_path: CSV file to read from
    :param header: `True` if the file contains a header, `False` otherwise (default: `False`)
    :param value_columns: name of the volue column(s) if the file contains a header, otherwise the column index/indices (default: `0`)
    :param freq_column: name of the frequency column(s) if the file contains a header, otherwise the column index/indices (default: `1`)
    :param encoding: character encoding of the CSV file (default: `utf-8`)
    :param delimiter: column delimiter (default: `,`)
    :param rng: random number generator to use (default: `None`)
    :return: function returning Pandas series of values with a distribution similar to that of the input file
    """
    if rng is None:
        rng = np.random.default_rng()

    # if the value columns are a list, then read the type of its entries from the first one
    if type(value_columns) is list:
        if len(value_columns) == 0:
            raise ValueError("value column list cannot be empty")

        value_columns_type = type(value_columns[0])
    else:
        value_columns_type = type(value_columns)

    if value_columns_type is not type(freq_column):
        raise ValueError("value and frequency column must both be of the same type")

    # if value_columns is an int or str, wrap it into a list
    value_columns = [value_columns] if value_columns_type is not list else value_columns

    df = pd.read_csv(
        csv_file_path,
        header=0 if header else None,
        usecols=value_columns + [freq_column],
        dtype={freq_column: "int"},
        sep=delimiter,
        encoding=encoding,
    )

    # sum of absolute frequencies
    freq_total = df[freq_column].sum()
    # new series to track the relative frequencies
    rel_freq_lst = np.ones(len(df))
    # keep reference to original array to skip to_numpy() call later
    srs_rel_freq = pd.Series(rel_freq_lst, copy=False)

    for value_column in value_columns:
        # for each unique value in this column, sum up its absolute frequencies
        grouped_column_values_sum = (
            df[[value_column, freq_column]].groupby(value_column).sum()
        )
        # convert absolute into relative frequencies
        grouped_column_values_sum[freq_column] /= freq_total
        # create a dict of value to relative frequency
        value_to_rel_freq_dict = dict(
            zip(grouped_column_values_sum.index, grouped_column_values_sum[freq_column])
        )
        # use this dict to create a new series that replaces each value with its relative frequency
        srs_rel_freq_per_value = df[value_column].replace(value_to_rel_freq_dict)
        # multiply this series with the accumulating column for each row's probabilities
        srs_rel_freq *= srs_rel_freq_per_value

    # get the list of row tuples from selected value columns
    value_tuple_list = [tuple(r) for r in df[value_columns].to_numpy()]

    # noinspection PyTypeChecker
    def _generate(count: int) -> list[pd.Series]:
        x = rng.choice(value_tuple_list, count, p=rel_freq_lst)
        return [pd.Series(list(t) for t in zip(*x))]  # dark magic

    return _generate


def to_dataframe(
    generators: list[tuple[GeneratorFunc, Union[str, list[str]]]], count: int
):
    """
    Generate a dataframe by using multiple generators at once.
    This function takes a list of generators and the names for each column that a generator will create.

    :param generators: list of generators and assigned column names
    :param count: number of records to generate
    :return: dataframe with columns generated as specified
    """
    if len(generators) == 0:
        raise ValueError("list of generators may not be empty")

    col_to_srs_dict: dict[str, pd.Series] = {}

    for gen, gen_col_names in generators:
        # if a single string is provided, concat by wrapping it into a list
        if isinstance(gen_col_names, str):
            gen_col_names = [gen_col_names]

        # generate values
        gen_col_values = gen(count)

        # check that the generator returned as many columns as expected
        if len(gen_col_values) != len(gen_col_names):
            raise ValueError(
                f"generator returned {len(gen_col_values)} columns, but requires {len(gen_col_names)} to "
                f"fill column(s) for: {','.join(gen_col_names)}"
            )

        # assign name to series
        for i in range(len(gen_col_values)):
            col_to_srs_dict[gen_col_names[i]] = gen_col_values[i]

    # finally create df from the list of named series
    return pd.DataFrame(data=col_to_srs_dict)
