__all__ = ["Generator", "Mutator", "GeckoWarning"]

import typing as _t
import pandas as pd

Generator = _t.Callable[[int], list[pd.Series]]
Mutator = _t.Callable[[list[pd.Series], _t.Optional[float]], list[pd.Series]]


class GeckoWarning(UserWarning):
    """
    Generic class that represents all warnings emitted by Gecko.
    """

    pass
