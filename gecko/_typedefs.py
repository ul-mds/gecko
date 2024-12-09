__all__ = ["Generator", "Mutator", "GeckoWarning"]

import typing as _t
import pandas as pd

Generator = _t.Callable[[int], list[pd.Series]]
Mutator = _t.Callable[[list[pd.Series], _t.Optional[float]], list[pd.Series]]
DateTimeUnit = _t.Literal["d", "days", "h", "hours", "m", "minutes", "s", "seconds"]


class GeckoWarning(UserWarning):
    """
    Generic class that represents all warnings emitted by Gecko.
    """

    pass


_gecko_to_pd_dt_unit_dict = {
    "d": "D",
    "days": "D",
    "h": "h",
    "hours": "h",
    "m": "m",
    "minutes": "m",
    "s": "s",
    "seconds": "s",
}


def convert_gecko_date_time_unit_to_pandas(unit: str) -> str:
    pd_dt_unit = _gecko_to_pd_dt_unit_dict.get(unit)

    if pd_dt_unit is None:
        raise ValueError(
            f"unrecognized unit `{unit}`, must be one of: `{'`, `'.join(sorted(_gecko_to_pd_dt_unit_dict.keys()))}`"
        )

    return pd_dt_unit
