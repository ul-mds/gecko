import os
import string
import warnings
from pathlib import Path

import typing as _t
from uuid import uuid4

from collections import abc as _abc
import numpy as np


def get_asset_path(file_name: str) -> Path:
    return Path(__file__).parent / "assets" / file_name


def write_temporary_csv_file(
    dir_path: Path,
    header: _t.Optional[_abc.Iterable[str]] = None,
    rows: _t.Optional[_abc.Iterable[_abc.Iterable[str]]] = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
):
    file_name = f"{uuid4()}.csv"
    file_path = dir_path / file_name

    if header is None and rows is None:
        raise ValueError("either header or rows must be set")

    with file_path.open(mode="w", encoding=encoding) as f:
        if header is not None:
            f.write(f"{delimiter.join(str(h) for h in header)}{os.linesep}")

        if rows is not None:
            for row in rows:
                f.write(f"{delimiter.join([str(v) for v in row])}{os.linesep}")

    return file_path


def random_strings(
    n_strings: int = 1_000,
    str_len: int = 20,
    charset=string.ascii_letters,
    rng: _t.Optional[np.random.Generator] = None,
):
    if rng is None:
        rng = np.random.default_rng()
        warnings.warn("unseeded rng detected, test outcomes might not be reproducible")

    charset_lst = list(charset)

    def _random_string():
        return "".join(rng.choice(charset_lst, size=str_len))

    return [_random_string() for _ in range(n_strings)]
