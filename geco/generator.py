import csv
from pathlib import Path
from typing import Callable, ParamSpec

import numpy as np
from numpy.random import Generator

P = ParamSpec('P')

CallableGeneratorFunc = Callable[P, str]
GeneratorFunc = Callable[[int], list[list[str]]]


def from_function(
        func: CallableGeneratorFunc,
        *args,
        **kwargs
) -> GeneratorFunc:
    def _generate(count: int) -> list[list[str]]:
        return [[func(*args, **kwargs) for _ in np.arange(count)]]

    return _generate


def from_frequency_table(
        csv_file_path: Path,
        header: bool = False,
        encoding: str = "utf-8",
        delimiter: str = ",",
        rng: Generator | None = None
) -> GeneratorFunc:
    if rng is None:
        rng = np.random.default_rng()

    with csv_file_path.open(mode="r", encoding=encoding, newline="") as f:
        # create csv reader instance
        reader = csv.reader(f, delimiter=delimiter)
        # setup local vars
        value_list: list[str] = []
        abs_freq_list: list[int] = []
        total_abs_freq = 0

        # skip header row
        if header:
            next(reader)

        for line in reader:
            if len(line) != 2:
                raise ValueError("CSV file must contain two columns")

            line_val, line_freq = line[0], int(line[1])

            if line_freq < 0:
                raise ValueError("absolute frequency must not be negative")

            value_list.append(line_val)
            abs_freq_list.append(line_freq)
            total_abs_freq += line_freq

    rel_freq_list = [freq / total_abs_freq for freq in abs_freq_list]

    # return type can be treated as a list
    # noinspection PyTypeChecker
    def _generate(count: int) -> list[list[str]]:
        return [rng.choice(value_list, count, p=rel_freq_list)]

    return _generate
