"""
The mutator module provides mutator functions for mutating data.
These mutators implement common error sources such as typos based on keymaps, random edit errors and more.
"""

__all__ = [
    "Mutator",
    "with_cldr_keymap_file",
    "with_phonetic_replacement_table",
    "with_replacement_table",
    "with_missing_value",
    "with_insert",
    "with_delete",
    "with_transpose",
    "with_substitute",
    "with_noop",
    "with_categorical_values",
    "with_function",
    "with_permute",
    "with_lowercase",
    "with_uppercase",
    "with_datetime_offset",
    "with_generator",
    "with_regex_replacement_table",
    "with_repeat",
    "with_group",
    "mutate_data_frame",
    "PNotMetWarning",
]

import itertools
import re
import string
import warnings
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
import typing as _t

import numpy as np
import pandas as pd
from lxml import etree
import typing_extensions as _te

from gecko import dfidx
from gecko.cldr import decode_iso_kb_pos, unescape_kb_char, get_neighbor_kb_pos_for
from gecko.generator import Generator

Mutator = _t.Callable[[list[pd.Series], _t.Optional[float]], list[pd.Series]]


class _PhoneticReplacementRule(_t.NamedTuple):
    pattern: str
    replacement: str
    flags: str


def _check_probability_in_bounds(p: float):
    if p < 0 or p > 1:
        raise ValueError("probability is out of range, must be between 0 and 1")


@dataclass(frozen=True)
class KeyMutation:
    row: list[str] = field(default_factory=list)
    col: list[str] = field(default_factory=list)


P = _te.ParamSpec("P")


class PNotMetWarning(UserWarning):
    pass


def _warn_p(fn_name: str, p_expected: float, p_actual: float):
    warnings.warn(
        f"{fn_name}: desired probability of {p_expected} cannot be met since percentage of rows "
        f"that could possibly be mutated is {p_actual}",
        PNotMetWarning,
    )


def with_function(
    func: _t.Callable[_te.Concatenate[str, P], str],
    rng: _t.Optional[np.random.Generator] = None,
    *args: object,
    **kwargs: object,
) -> Mutator:
    """
    Mutate data using an arbitrary function that mutates a single value at a time.

    Notes:
        This function should be used sparingly since it is not vectorized.
        Only use it for testing purposes or if performance is not important.

    Args:
        func: function to invoke to mutate data with
        *args: positional arguments to pass to `func`
        **kwargs: keyword arguments to pass to `func`

    Returns:
        function returning list with strings mutated using custom function
    """

    if rng is None:
        rng = np.random.default_rng()

    def _bound_apply(val: object) -> str:
        x = func(str(val), *args, **kwargs)
        return str(x)

    def _mutate_series(srs: pd.Series, p: float) -> pd.Series:
        srs_out = srs.copy(deep=True)
        srs_rows_to_mutate = pd.Series(rng.random(size=len(srs)) < p, index=srs.index)
        srs_out.update(srs.loc[srs_rows_to_mutate].apply(_bound_apply))

        return srs_out

    def _mutate(srs_lst: list[pd.Series], p: float = 1.0) -> list[pd.Series]:
        _check_probability_in_bounds(p)
        return [_mutate_series(srs, p) for srs in srs_lst]

    return _mutate


def with_cldr_keymap_file(
    cldr_path: _t.Union[PathLike, str],
    charset: _t.Optional[str] = None,
    rng: _t.Optional[np.random.Generator] = None,
) -> Mutator:
    """
    Mutate data by randomly introducing typos.
    Potential typos are sourced from a Common Locale Data Repository (CLDR) keymap.
    Any character may be replaced with one of its horizontal or vertical neighbors on a keyboard.
    They may also be replaced with its upper- or lowercase variant.
    It is possible for a string to not be modified if a selected character has no possible replacements.

    Args:
        cldr_path: path to CLDR keymap file
        charset: string with characters that may be mutated
        rng: random number generator to use

    Returns:
        function returning list with strings mutated by applying typos according to keymap file
    """
    if rng is None:
        rng = np.random.default_rng()

    with Path(cldr_path).open(mode="r", encoding="utf-8") as f:
        tree = etree.parse(f)

    root = tree.getroot()

    # compute the row and column count
    max_row, max_col = 0, 0

    for map_node in root.iterfind("./keyMap/map"):
        # decode_iso_kb_pos is cached so calling this repeatedly shouldn't have an impact on performance
        kb_row, kb_col = decode_iso_kb_pos(map_node.get("iso"))
        max_row = max(max_row, kb_row)
        max_col = max(max_col, kb_col)

    kb_map = np.chararray(
        shape=(
            max_row + 1,
            max_col + 1,
            2,
        ),  # + 1 because rows and cols are zero-indexed, 2 to accommodate shift
        itemsize=1,  # each cell holds one unicode char
        unicode=True,
    )
    kb_map[:] = ""  # initialize with empty strings

    # remember the kb pos for each character
    kb_char_to_kb_pos_dict: dict[str, (int, int, int)] = {}

    for key_map_node in root.iterfind("./keyMap"):
        key_map_mod = key_map_node.get("modifiers")

        if key_map_mod is None:
            kb_mod = 0
        elif key_map_mod == "shift":
            kb_mod = 1
        else:
            continue

        for map_node in key_map_node.iterfind("./map"):
            kb_row, kb_col = decode_iso_kb_pos(map_node.get("iso"))
            kb_char = unescape_kb_char(map_node.get("to"))

            # check that char is listed if charset of permitted chars is provided
            if charset is not None and kb_char not in charset:
                continue

            kb_char_to_kb_pos_dict[kb_char] = (kb_row, kb_col, kb_mod)
            kb_map[kb_row, kb_col, kb_mod] = kb_char

    # map each character with other nearby characters that it could be replaced with due to a typo
    kb_char_to_candidates_dict: dict[str, str] = {}

    with np.nditer(kb_map, flags=["multi_index"], op_flags=[["readonly"]]) as it:
        for kb_char in it:
            # iterator returns str as array of unicode chars. convert it to str.
            kb_char = str(kb_char)

            # skip keys that don't have a character assigned to them
            if kb_char == "":
                continue

            kb_pos = it.multi_index
            # noinspection PyTypeChecker
            kb_pos_neighbors = get_neighbor_kb_pos_for(kb_pos, max_row, max_col)
            kb_char_candidates = set()

            for kb_pos_neighbor in kb_pos_neighbors:
                kb_char_candidate = kb_map[kb_pos_neighbor]

                # check that the key pos has a char assigned to it. it may also happen that the char is the same
                # despite the kb modifier. that needs to be accounted for.
                if kb_char_candidate != "" and kb_char_candidate != kb_char:
                    kb_char_candidates.add(kb_char_candidate)

            # check that there are any candidates
            if len(kb_char_candidates) > 0:
                kb_char_to_candidates_dict[kb_char] = "".join(
                    sorted(
                        list(kb_char_candidates)
                    )  # needs to be sorted to ensure reproducibility
                )

    def _mutate_series(srs: pd.Series, p: float) -> pd.Series:
        _check_probability_in_bounds(p)

        srs_out = srs.copy()
        srs_len = len(srs_out)

        # make sure it aligns with the original index
        srs_candidate_chars = pd.Series([""] * srs_len, index=srs_out.index)

        for candidate_char in kb_char_to_candidates_dict.keys():
            # check for rows where candidate charis president
            srs_contains_candidate_char = srs.str.contains(candidate_char, regex=False)
            # add it as a candidate char to all applicable rows
            srs_candidate_chars.loc[srs_contains_candidate_char] += candidate_char

        # create a mask that selects all rows affected by mutation
        # first drop all rows which have no candidate chars
        srs_selected_for_mutation = srs_candidate_chars != ""

        # now check if the desired p value can be reached, warn if not
        p_candidates = srs_selected_for_mutation.sum() / srs_len

        if p_candidates < p:
            _warn_p("with_cldr_keymap_file", p, p_candidates)

        # select p for all eligible rows, avoid values > 1
        p_subset_select = min(1.0, p / p_candidates)
        # draw random rows to actually perform mutation on
        arr_rng_vals = rng.random(size=srs_selected_for_mutation.sum())
        # update the mutation mask (it may happen that rows that were True might flip to False)
        srs_selected_for_mutation.loc[srs_selected_for_mutation] = (
            arr_rng_vals < p_subset_select
        )
        # this is now the amount of all rows that will definitely be mutated
        rows_to_mutate_count = srs_selected_for_mutation.sum()

        # now srs_selected_for_mutation accurately represents the rows to perform mutation on, satisfying p
        # and rows that could actually be affected
        srs_candidates_selected = pd.Series([""] * srs_len, index=srs.index)

        # draw random candidate chars
        arr_rng_vals = rng.random(size=rows_to_mutate_count)
        arr_rng_idxs = np.floor(
            srs_candidate_chars.loc[srs_selected_for_mutation].str.len() * arr_rng_vals
        ).astype(int)

        # iterate over all unique indices and select the char at the i-th position
        for idx in arr_rng_idxs.unique():
            idx_mask = arr_rng_idxs == idx
            srs_this_idx = srs_candidate_chars.loc[srs_selected_for_mutation].loc[
                idx_mask
            ]
            srs_candidates_selected.update(srs_this_idx.str[idx])

        # now do the same process for replacement chars
        srs_replacements_selected = pd.Series([""] * srs_len, index=srs.index)

        for candidate_char in srs_candidates_selected.loc[
            srs_selected_for_mutation
        ].unique():
            srs_this_candidate_char = srs_candidates_selected == candidate_char
            # count all affected rows
            candidate_char_count = srs_this_candidate_char.sum()
            # fetch replacement chars
            replacement_chars = kb_char_to_candidates_dict[candidate_char]
            # draw random replacement chars for each row
            arr_rng_repl = rng.choice(
                list(replacement_chars), size=candidate_char_count
            )
            # and update the replacement char series
            srs_replacements_selected.loc[srs_this_candidate_char] = arr_rng_repl

        # and now we do the actual replacements
        for candidate_char in srs_candidates_selected.loc[
            srs_selected_for_mutation
        ].unique():
            # filter by global mask and rows that contain this candidate char
            srs_this_replacement_char = srs_selected_for_mutation & (
                srs_candidates_selected == candidate_char
            )

            for replacement_char in srs_replacements_selected.loc[
                srs_this_replacement_char
            ].unique():
                srs_out.update(
                    srs_out.loc[srs_this_replacement_char].str.replace(
                        candidate_char, replacement_char, n=1
                    )
                )

        return srs_out

    def _mutate(srs_lst: list[pd.Series], p: float) -> list[pd.Series]:
        return [_mutate_series(srs, p) for srs in srs_lst]

    return _mutate


def with_phonetic_replacement_table(
    data_source: _t.Union[PathLike, str, pd.DataFrame],
    source_column: _t.Union[int, str] = 0,
    target_column: _t.Union[int, str] = 1,
    flags_column: _t.Union[int, str] = 2,
    encoding: str = "utf-8",
    delimiter: str = ",",
    rng: _t.Optional[np.random.Generator] = None,
) -> Mutator:
    """
    Mutate data by randomly replacing characters with others that sound similar.
    The rules for similar-sounding character sequences are sourced from a CSV file.
    This table must have at least three columns: a source, target and a flag column.
    A source pattern is mapped to its target under the rules imposed by the provided flags.
    These flags determine where such a replacement can take place within a string.
    If no flags are defined, it is implied that this replacement can take place anywhere in a string.
    Conversely, if `^`, `$`, `_`, or any combination of the three are set, it implies that a replacement
    can only occur at the start, end or in the middle of a string.
    If the source, target and flags column are provided as strings, then it is automatically assumed that the CSV file
    has a header row.

    Args:
        data_source: path to CSV file or data frame containing phonetic replacement rules
        source_column: name or index of source column
        target_column: name or index of target column
        flags_column: name or index of flag column
        encoding: character encoding of the CSV file
        delimiter: column delimiter of the CSV file
        rng: random number generator to use

    Returns:
        function returning list with strings mutated by applying phonetic errors according to rules in CSV file
    """

    # list of all flags. needs to be sorted for rng.
    _all_flags = "".join(sorted("^$_"))

    def _validate_flags(flags_str: _t.Optional[str]) -> str:
        """Check a string for valid flags. Returns all flags if string is empty, `NaN` or `None`."""
        if pd.isna(flags_str) or flags_str == "" or flags_str is None:
            return _all_flags

        for char in flags_str:
            if char not in _all_flags:
                raise ValueError(f"unknown flag: {char}")

        return flags_str

    def __new_unknown_flag_error(flag: str):
        return ValueError(f"invalid state: unknown flag `{flag}`")

    if rng is None:
        rng = np.random.default_rng()

    if type(source_column) is not type(target_column) or type(
        target_column
    ) is not type(flags_column):
        raise ValueError("source, target and flags columns must be of the same type")

    # skip check for source and target column bc they are all of the same type already
    if not isinstance(flags_column, str) and not isinstance(flags_column, int):
        raise ValueError(
            "source, target and flags columns must be either a string or an integer"
        )

    if isinstance(data_source, pd.DataFrame):
        df = data_source
    else:
        header = isinstance(flags_column, str)

        # read csv file
        df = pd.read_csv(
            data_source,
            header=0 if header else None,
            dtype=str,
            usecols=[source_column, target_column, flags_column],
            keep_default_na=False,
            sep=delimiter,
            encoding=encoding,
        )

    # parse replacement rules
    phonetic_replacement_rules: list[_PhoneticReplacementRule] = []

    for _, row in df.iterrows():
        pattern = row[source_column]
        replacement = row[target_column]
        flags = _validate_flags(row[flags_column])

        phonetic_replacement_rules.append(
            _PhoneticReplacementRule(pattern, replacement, flags)
        )

    def _mutate_series(srs: pd.Series) -> pd.Series:
        # create a copy of input series
        srs_out = srs.copy()
        # get series of string lengths
        srs_str_out_len = srs_out.str.len()
        # get series length
        str_count = len(srs_out)
        # create series to compute substitution probability
        srs_str_sub_prob = pd.Series(dtype=float, index=srs_out.index)
        srs_str_sub_prob[:] = 0
        # track possible replacements for each rule
        rule_to_flag_dict: dict[_PhoneticReplacementRule, pd.Series] = {}

        for rule in phonetic_replacement_rules:
            # increment absolute frequency for each string where rule applies
            srs_str_flags = pd.Series(dtype=str, index=srs_out.index)
            srs_str_flags[:] = ""

            # find pattern in series
            srs_pattern_idx = srs_out.str.find(rule.pattern)

            if "^" in rule.flags:
                # increment counter for all strings where pattern is found at start of string
                mask_pattern_at_start = srs_pattern_idx == 0
                srs_str_flags[mask_pattern_at_start] += "^"

            if "$" in rule.flags:
                # increment counter for all strings where pattern is found at end of string
                mask_pattern_at_end = (
                    srs_pattern_idx + len(rule.pattern) == srs_str_out_len
                )
                srs_str_flags[mask_pattern_at_end] += "$"

            if "_" in rule.flags:
                # increment counter for all strings where pattern is not at the start and at the end
                mask_pattern_in_middle = (srs_pattern_idx > 0) & (
                    srs_pattern_idx + len(rule.pattern) < srs_str_out_len
                )
                srs_str_flags[mask_pattern_in_middle] += "_"

            rule_to_flag_dict[rule] = srs_str_flags
            srs_str_sub_prob[srs_str_flags != ""] += 1

        # prevent division by zero
        mask_eligible_strs = srs_str_sub_prob != 0
        # absolute -> relative frequency
        srs_str_sub_prob[mask_eligible_strs] = 1 / srs_str_sub_prob[mask_eligible_strs]
        # keep track of modified rows
        mask_modified_rows = pd.Series(dtype=bool, index=srs_out.index)
        mask_modified_rows[:] = False

        for rule in phonetic_replacement_rules:
            # draw random numbers for each row
            arr_rand_vals = rng.random(size=str_count)
            # get flags that were generated for each row
            srs_str_flags = rule_to_flag_dict[rule]
            # get candidate row mask
            mask_candidate_rows = (arr_rand_vals < srs_str_sub_prob) & (
                srs_str_flags != ""
            )

            # create copy of rule flags and shuffle it in-place
            arr_rand_flags = list(rule.flags)
            rng.shuffle(arr_rand_flags)

            for flag in arr_rand_flags:
                # select rows that can have the current rule applied to them, fit into the correct flag
                # and haven't been modified yet
                if flag == "^":
                    mask_current_flag = srs_out.str.startswith(rule.pattern)
                elif flag == "$":
                    mask_current_flag = srs_out.str.endswith(rule.pattern)
                elif flag == "_":
                    # not at the start and not at the end
                    mask_current_flag = ~srs_out.str.startswith(rule.pattern) & (
                        ~srs_out.str.endswith(rule.pattern)
                    )
                else:
                    raise __new_unknown_flag_error(flag)

                mask_current_candidate_rows = (
                    mask_candidate_rows & mask_current_flag & ~mask_modified_rows
                )

                # skip if there are no replacements to be made
                if mask_current_candidate_rows.sum() == 0:
                    continue

                if flag == "^":
                    srs_out[mask_current_candidate_rows] = srs_out[
                        mask_current_candidate_rows
                    ].str.replace(f"^{rule.pattern}", rule.replacement, n=1, regex=True)
                elif flag == "$":
                    srs_out[mask_current_candidate_rows] = srs_out[
                        mask_current_candidate_rows
                    ].str.replace(f"{rule.pattern}$", rule.replacement, n=1, regex=True)
                elif flag == "_":
                    # matching groups are the parts that are supposed to be preserved
                    # (anything but the string to replace).
                    srs_out[mask_current_candidate_rows] = srs_out[
                        mask_current_candidate_rows
                    ].str.replace(
                        f"^(.+){rule.pattern}(.+)$",
                        f"\\1{rule.replacement}\\2",
                        n=1,
                        regex=True,
                    )
                else:
                    raise __new_unknown_flag_error(flag)

                # update modified row series
                mask_modified_rows |= mask_current_candidate_rows

        return srs_out

    def _mutate(srs_lst: list[pd.Series]) -> list[pd.Series]:
        return [_mutate_series(srs) for srs in srs_lst]

    return _mutate


def with_replacement_table(
    data_source: _t.Union[PathLike, str, pd.DataFrame],
    source_column: _t.Union[str, int] = 0,
    target_column: _t.Union[str, int] = 1,
    inline: bool = False,
    reverse: bool = False,
    encoding: str = "utf-8",
    delimiter: str = ",",
    rng: _t.Optional[np.random.Generator] = None,
) -> Mutator:
    """
    Mutate data by randomly substituting sequences from a replacement table.
    The table must have at least two columns: a source and a target value column.
    A source value may have multiple target values that it can map to.
    Strings that do not contain any possible source values are not mutated.
    It is possible for a string to not be modified if no target value could be picked for its assigned source value.
    This can only happen if a source value is mapped to multiple target values.
    In this case, each target value will be independently selected or not.
    If the source and target column are provided as strings, then it is automatically assumed that the CSV file
    has a header row.

    Args:
        data_source: path to CSV file or data frame containing replacement table
        source_column: name or index of the source column
        target_column: name or index of the target column
        inline: whether to perform replacements inline
        reverse: whether to allow replacements from target to source column
        encoding: character encoding of the CSV file
        delimiter: column delimiter of the CSV file
        rng: random number generator to use

    Returns:
        function returning list with strings mutated by replacing characters as specified in CSV file
    """

    if rng is None:
        rng = np.random.default_rng()

    if type(source_column) is not type(target_column):
        raise ValueError("source and target columns must be of the same type")

    # skip check for source_column bc they are both of the same type already
    if not isinstance(target_column, str) and not isinstance(target_column, int):
        raise ValueError(
            "source and target columns must be either a string or an integer"
        )

    if isinstance(data_source, pd.DataFrame):
        df = data_source
    else:
        header = isinstance(target_column, str)

        df = pd.read_csv(
            data_source,
            header=0 if header else None,
            dtype=str,
            usecols=[source_column, target_column],
            keep_default_na=False,
            sep=delimiter,
            encoding=encoding,
        )

    if reverse:
        # flip columns and concat
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    data={
                        source_column: df.loc[:, target_column],
                        target_column: df.loc[:, source_column],
                    }
                ),
            ],
            ignore_index=True,
        )

    # unique() returns a ndarray
    arr_unique_source_values = df.loc[:, source_column].unique()

    def _mutate_series(srs: pd.Series, p: float) -> pd.Series:
        # create copy
        srs_out = srs.copy(deep=True)
        # create index df
        df_idx = dfidx.with_capacity(len(srs), len(arr_unique_source_values))

        for src_idx, source in enumerate(arr_unique_source_values):
            if inline:
                dfidx.set_index(df_idx, srs_out.str.contains(source), src_idx)
            else:
                dfidx.set_index(df_idx, srs_out == source, src_idx)

        # check rows that can be mutated
        srs_rows_to_mutate = dfidx.any_set(df_idx)
        possible_rows_to_mutate = srs_rows_to_mutate.sum()
        p_actual = possible_rows_to_mutate / len(srs)

        # warn if p cannot be met
        if p_actual < p:
            _warn_p("with_replacement_table", p, p_actual)

        # perform actual selection
        p_subset_select = min(1.0, p / p_actual)
        arr_rng_vals = rng.random(size=possible_rows_to_mutate)
        srs_rows_to_mutate.loc[srs_rows_to_mutate] = arr_rng_vals < p_subset_select

        # create new series to track selected source values
        srs_source_values = pd.Series([pd.NA] * len(srs), dtype=str, index=srs.index)
        # randomize order of source indices
        # TODO instead of shuffling, this should probably favor source values that don't appear often
        # one way could be to have dfidx output sums for every index, then sort them in ascending order
        arr_rng_src_idx = np.arange(0, len(arr_unique_source_values))
        rng.shuffle(arr_rng_src_idx)

        # populate the source value series
        for src_idx in arr_rng_src_idx:
            srs_src_indexed = dfidx.test_index(df_idx, src_idx)
            # check which rows will have this source value replaced
            srs_src_selected = (
                srs_rows_to_mutate  # select rows that are eligible for mutation
                & pd.isna(srs_source_values)  # AND that have no source value set yet
                & srs_src_indexed  # AND that contain this source value
            )
            # update the source value series accordingly
            srs_source_values.loc[srs_src_selected] = arr_unique_source_values[src_idx]

        # populate the target value series
        srs_target_values = pd.Series([pd.NA] * len(srs), dtype=str, index=srs.index)

        for source_value in srs_source_values.dropna().unique():
            # filter by all rows that have this source value applied to them
            srs_this_source = srs_source_values == source_value
            # collect all possible target values
            target_values = df.loc[
                df[source_column] == source_value, target_column
            ].array
            # select random target values
            target_values_selected = rng.choice(
                target_values, size=srs_this_source.sum()
            )
            # update target values
            srs_target_values.loc[srs_this_source] = target_values_selected

        # perform the actual replacements
        for source_value in srs_source_values.dropna().unique():
            srs_this_source = srs_source_values == source_value

            for target_value in srs_target_values.loc[srs_this_source].unique():
                srs_to_mutate = srs_this_source & (srs_target_values == target_value)

                if inline:
                    srs_out.update(
                        srs_out.loc[srs_to_mutate].str.replace(
                            source_value, target_value, n=1, regex=False
                        )
                    )
                else:
                    srs_out.loc[srs_to_mutate] = target_value

        return srs_out

    def _mutate(srs_lst: list[pd.Series], p: float = 1) -> list[pd.Series]:
        _check_probability_in_bounds(p)
        return [_mutate_series(srs, p) for srs in srs_lst]

    return _mutate


def with_missing_value(
    value: str = "",
    rng: _t.Optional[np.random.Generator] = None,
) -> Mutator:
    """
    Mutate data by replacing select entries with a representative "missing" value.

    Args:
        value: "missing" value to replace select entries with
        rng: random number generator to use

    Returns:
        function returning list where select strings will be replaced with the "missing" value
    """
    if rng is None:
        rng = np.random.default_rng()

    def _mutate_series(srs: pd.Series, p: float) -> pd.Series:
        srs_out = srs.copy(deep=True)

        srs_rows_to_mutate = srs != value
        possible_rows_to_mutate = srs_rows_to_mutate.sum()
        p_actual = possible_rows_to_mutate / len(srs)

        if p_actual < p:
            _warn_p("with_missing_value", p, p_actual)

        # select subset of rows to mutate
        p_subset_select = min(1.0, p / p_actual)
        arr_rng_vals = rng.random(size=possible_rows_to_mutate)
        srs_rows_to_mutate.loc[srs_rows_to_mutate] = arr_rng_vals < p_subset_select

        # update rows to mutate
        srs_out.loc[srs_rows_to_mutate] = value

        return srs_out

    def _mutate(srs_lst: list[pd.Series], p: float = 1.0) -> list[pd.Series]:
        _check_probability_in_bounds(p)
        return [_mutate_series(srs, p) for srs in srs_lst]

    return _mutate


def with_insert(
    charset: _t.Union[str, list[str]] = string.ascii_letters,
    rng: _t.Optional[np.random.Generator] = None,
) -> Mutator:
    """
    Mutate data by inserting random characters.
    The characters are drawn from the provided charset.

    Args:
        charset: string to sample random characters from
        rng: random number generator to use

    Returns:
        function returning list with strings mutated by inserting random characters
    """
    if rng is None:
        rng = np.random.default_rng()

    if isinstance(charset, str):
        charset = list(charset)

    def _mutate_series(srs: pd.Series, p: float) -> pd.Series:
        srs_out = srs.copy(deep=True)

        # select rows (don't need to do p-check because a new character can always be appended)
        srs_rows_to_mutate = pd.Series(rng.random(size=len(srs)) < p, index=srs.index)

        # count rows that will be mutated
        rows_to_mutate_count = srs_rows_to_mutate.sum()

        # generate random indices to insert values at
        arr_rng_vals = rng.random(size=rows_to_mutate_count)
        arr_ins_idx = np.floor(
            (srs.str.len() + 1)
            * arr_rng_vals  # +1 because letters can be inserted at the end
        ).astype(int)

        # generate random characters to insert
        arr_rng_chars = rng.choice(charset, size=rows_to_mutate_count)
        # create view on values to mutate
        srs_in_rows = srs.loc[srs_rows_to_mutate]

        for ins_idx in arr_ins_idx.unique():
            # select all rows that have an insertion at this index
            arr_this_idx = arr_ins_idx == ins_idx
            srs_this_idx = srs_in_rows.loc[arr_this_idx]
            # then update all affected rows
            srs_out.update(
                srs_this_idx.str[:ins_idx]
                + arr_rng_chars[arr_this_idx]
                + srs_this_idx.str[ins_idx:]
            )

        return srs_out

    def _mutate(srs_lst: list[pd.Series], p: float = 1.0) -> list[pd.Series]:
        return [_mutate_series(srs, p) for srs in srs_lst]

    return _mutate


def with_delete(rng: _t.Optional[np.random.Generator] = None) -> Mutator:
    """
    Mutate data by randomly deleting characters.

    Args:
        rng: random number generator to use

    Returns:
        function returning list with strings mutated by deleting random characters
    """
    if rng is None:
        rng = np.random.default_rng()

    def _mutate_series(srs: pd.Series, p: float) -> pd.Series:
        srs_out = srs.copy(deep=True)

        # limit to strings that have at least a single character
        srs_rows_to_mutate = srs.str.len() >= 1
        possible_rows_to_mutate = srs_rows_to_mutate.sum()
        p_actual = possible_rows_to_mutate / len(srs)

        if p_actual < p:
            _warn_p("with_delete", p, p_actual)

        # select subset of rows to mutate
        p_subset_select = min(1.0, p / p_actual)
        arr_rng_vals = rng.random(size=possible_rows_to_mutate)
        srs_rows_to_mutate.loc[srs_rows_to_mutate] = arr_rng_vals < p_subset_select

        # generate random indices
        arr_rng_vals = rng.random(size=possible_rows_to_mutate)
        arr_rng_idx = np.floor(
            srs.loc[srs_rows_to_mutate].str.len() * arr_rng_vals
        ).astype(int)

        # perform the character deletion (don't need to dropna() here)
        for del_idx in arr_rng_idx.unique():
            srs_this_idx = srs.loc[srs_rows_to_mutate].loc[arr_rng_idx == del_idx]
            srs_out.update(srs_this_idx.str.slice_replace(del_idx, del_idx + 1, ""))

        return srs_out

    def _mutate(srs_lst: list[pd.Series], p: float = 1.0) -> list[pd.Series]:
        _check_probability_in_bounds(p)
        return [_mutate_series(srs, p) for srs in srs_lst]

    return _mutate


def with_transpose(rng: _t.Optional[np.random.Generator] = None) -> Mutator:
    """
    Mutate data by randomly swapping neighboring characters.

    Notes:
        It is possible for the same two neighboring characters to be swapped.

    Args:
        rng: random number generator to use

    Returns:
        function returning list with strings mutated by transposing random neighboring characters
    """
    if rng is None:
        rng = np.random.default_rng()

    def _mutate_series(srs: pd.Series, p: float) -> pd.Series:
        srs_out = srs.copy(deep=True)

        # limit to strings that have at least two characters
        srs_rows_to_mutate = srs.str.len() >= 2
        possible_rows_to_mutate = srs_rows_to_mutate.sum()
        p_actual = possible_rows_to_mutate / len(srs)

        if p_actual < p:
            _warn_p(with_transpose.__name__, p, p_actual)

        # select subset of rows to mutate
        p_subset_select = min(1.0, p / p_actual)
        arr_rng_vals = rng.random(size=possible_rows_to_mutate)
        srs_rows_to_mutate.loc[srs_rows_to_mutate] = arr_rng_vals < p_subset_select

        # count rows that will be mutated
        rows_to_mutate_count = srs_rows_to_mutate.sum()

        # generate random indices to transpose characters at
        arr_rng_vals = rng.random(size=rows_to_mutate_count)
        arr_rng_idx = np.floor(
            (srs.loc[srs_rows_to_mutate].str.len() - 1)
            * arr_rng_vals  # -1 to account for neighboring chars
        ).astype(int)

        for idx in arr_rng_idx.unique():
            srs_this_idx = srs.loc[srs_rows_to_mutate].loc[arr_rng_idx == idx]
            srs_out.update(
                srs_this_idx.str[:idx]
                + srs_this_idx.str[idx + 1]
                + srs_this_idx.str[idx]
                + srs_this_idx.str[idx + 2 :]
            )

        return srs_out

    def _mutate(srs_lst: list[pd.Series], p: float = 1.0) -> list[pd.Series]:
        _check_probability_in_bounds(p)
        return [_mutate_series(srs, p) for srs in srs_lst]

    return _mutate


def with_substitute(
    charset: _t.Union[str, list[str]] = string.ascii_letters,
    rng: _t.Optional[np.random.Generator] = None,
) -> Mutator:
    """
    Mutate data by replacing single characters with a new one.
    The characters are drawn from the provided charset.

    Notes:
        It is possible for a character to be replaced by itself.

    Args:
        charset: string to sample random characters from
        rng: random number generator to use

    Returns:
        function returning list with strings mutated by substituting random characters
    """
    if rng is None:
        rng = np.random.default_rng()

    if isinstance(charset, str):
        charset = list(charset)

    def _mutate_series(srs: pd.Series, p: float) -> pd.Series:
        srs_out = srs.copy(deep=True)

        # limit to strings that have at least a single character
        srs_rows_to_mutate = srs.str.len() >= 1
        possible_rows_to_mutate = srs_rows_to_mutate.sum()
        p_actual = possible_rows_to_mutate / len(srs)

        if p_actual < p:
            _warn_p(with_substitute.__name__, p, p_actual)

        # select subset of rows to mutate
        p_subset_select = min(1.0, p / p_actual)
        arr_rng_vals = rng.random(size=possible_rows_to_mutate)
        srs_rows_to_mutate.loc[srs_rows_to_mutate] = arr_rng_vals < p_subset_select

        # count rows that will be mutated
        rows_to_mutate_count = srs_rows_to_mutate.sum()

        # generate random indices
        arr_rng_vals = rng.random(size=rows_to_mutate_count)
        arr_rng_idx = np.floor(
            srs.loc[srs_rows_to_mutate].str.len() * arr_rng_vals
        ).astype(int)

        # generate random characters to insert
        arr_rng_chars = rng.choice(charset, size=rows_to_mutate_count)

        for idx in arr_rng_idx.unique():
            arr_this_idx = arr_rng_idx == idx
            srs_this_idx = srs.loc[srs_rows_to_mutate].loc[arr_this_idx]
            srs_out.update(
                srs_this_idx.str[:idx]
                + arr_rng_chars[arr_this_idx]
                + srs_this_idx.str[idx + 1 :]
            )

        return srs_out

    def _mutate(srs_lst: list[pd.Series], p: float = 1.0) -> list[pd.Series]:
        _check_probability_in_bounds(p)
        return [_mutate_series(srs, p) for srs in srs_lst]

    return _mutate


def with_noop() -> Mutator:
    """
    Mutate data by not mutating it at all.
    This mutator returns the input series as-is.
    You might use it to leave a certain percentage of records in a series untouched.

    Returns:
        function returning list of strings as-is
    """

    def _mutate(srs_lst: list[pd.Series], _: float) -> list[pd.Series]:
        return srs_lst

    return _mutate


def with_categorical_values(
    data_source: _t.Union[PathLike, str, pd.DataFrame],
    value_column: _t.Union[str, int] = 0,
    encoding: str = "utf-8",
    delimiter: str = ",",
    rng: _t.Optional[np.random.Generator] = None,
) -> Mutator:
    """
    Mutate data by replacing it with another from a list of categorical values.
    This mutator reads all unique values from a column within a CSV file.
    All strings within a series will be replaced with a different random value from this column.
    If the value column is provided as a string, then it is automatically assumed that the CSV file
    has a header row.

    Args:
        data_source: path to CSV file or data frame containing values
        value_column: name or index of value column
        encoding: character encoding of the CSV file
        delimiter: column delimiter of the CSV file
        rng: random number generator to use

    Returns:
        function returning list with strings mutated by replacing values with a different one from a limited set of permitted values
    """
    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(value_column, str) and not isinstance(value_column, int):
        raise ValueError("value column must be either a string or an integer")

    if isinstance(data_source, pd.DataFrame):
        df = data_source
    else:
        header = isinstance(value_column, str)

        # read csv file
        df = pd.read_csv(
            data_source,
            header=0 if header else None,
            dtype=str,
            usecols=[value_column],
            keep_default_na=False,
            sep=delimiter,
            encoding=encoding,
        )

    # fetch unique values
    arr_unique_values = np.array(sorted(df.loc[:, value_column].unique()))

    if len(arr_unique_values) < 2:
        raise ValueError(
            f"column must contain at least two unique values, has {len(arr_unique_values)}"
        )

    def _mutate_series(srs: pd.Series, p: float) -> pd.Series:
        # create copy
        srs_out = srs.copy(deep=True)
        # track which rows contain a value that can be mutated
        srs_rows_to_mutate = pd.Series([False] * len(srs), index=srs.index)

        # update by checking which rows contain a candidate value
        for val in arr_unique_values:
            srs_rows_to_mutate |= srs == val

        # check rows that can be mutated
        possible_rows_to_mutate = srs_rows_to_mutate.sum()
        p_actual = possible_rows_to_mutate / len(srs)

        # warn if p cannot be met
        if p_actual < p:
            _warn_p(with_categorical_values.__name__, p, p_actual)

        # perform selection
        p_subset_select = min(1.0, p / p_actual)
        arr_rng_vals = rng.random(size=possible_rows_to_mutate)
        srs_rows_to_mutate.loc[srs_rows_to_mutate] = arr_rng_vals < p_subset_select

        for val in arr_unique_values:
            # fetch all rows that match the value
            srs_this_val = srs_rows_to_mutate & (srs == val)
            rows_to_mutate_count = srs_this_val.sum()

            # skip if no rows match
            if rows_to_mutate_count == 0:
                continue

            # get the set of unique values minus the one that is currently processed
            arr_unique_values_without_this = np.setdiff1d(arr_unique_values, val)
            # perform the update
            srs_out.loc[srs_this_val] = rng.choice(
                arr_unique_values_without_this, size=rows_to_mutate_count
            )

        return srs_out

    def _mutate(srs_lst: list[pd.Series], p: float = 1.0) -> list[pd.Series]:
        _check_probability_in_bounds(p)
        return [_mutate_series(srs, p) for srs in srs_lst]

    return _mutate


def with_permute(rng: _t.Optional[np.random.Generator] = None) -> Mutator:
    """
    Mutate data from series by permuting their contents.
    This function ensures that for each row there is at least one permutation happening.

    Args:
        rng: random number generator to use

    Returns:
        function returning list with the entries in each series swapped
    """
    if rng is None:
        rng = np.random.default_rng()

    def _filter_permutations(tpl: tuple[int, ...]) -> bool:
        for idx, val in enumerate(tpl):
            if idx == val:
                return False

        return True

    def _mutate(srs_lst: list[pd.Series], p: float = 1.0) -> list[pd.Series]:
        _check_probability_in_bounds(p)
        srs_lst_len = len(srs_lst)

        if srs_lst_len < 2:
            raise ValueError("list must contain at least two series to permute")

        srs_0_len = len(srs_lst[0])

        for i in range(1, srs_lst_len):
            if len(srs_lst[i]) != srs_0_len:
                raise ValueError("all series must be of the same length")

        # nothing to permute
        if srs_0_len == 0:
            return srs_lst

        # generate all series index permutations and remove the tuple with all indices in order.
        # filter out all tuples that keep values from any column in the same column.
        srs_idx_permutations = sorted(
            filter(_filter_permutations, itertools.permutations(range(len(srs_lst))))
        )

        # select rows
        arr_rows_to_mutate = rng.random(size=srs_0_len) < p

        # choose random index permutations
        arr_rand_idx_tpl = rng.choice(srs_idx_permutations, size=srs_0_len)
        # map tuples to each series (zip indices -> convert to array -> wrap in list)
        arr_idx_per_srs = list(map(np.array, zip(*arr_rand_idx_tpl)))

        srs_lst_out = [srs.copy(deep=True) for srs in srs_lst]

        for i in range(srs_lst_len):
            for j in range(srs_lst_len):
                if i == j:
                    continue

                arr_this_srs = arr_rows_to_mutate & (arr_idx_per_srs[i] == j)
                srs_lst_out[i].loc[arr_this_srs] = srs_lst[j].loc[arr_this_srs]

        return srs_lst_out

    return _mutate


def with_lowercase(rng: _t.Optional[np.random.Generator] = None) -> Mutator:
    """
    Mutate data from a series by converting it into lowercase.

    Returns:
        function returning list with the entries in each series converted to lowercase
    """
    if rng is None:
        rng = np.random.default_rng()

    def _mutate_series(srs: pd.Series, p: float) -> pd.Series:
        srs_out = srs.copy(deep=True)

        # limit series to strings that are not all uppercase yet
        srs_rows_to_mutate = ~srs.str.islower()
        possible_rows_to_mutate = srs_rows_to_mutate.sum()
        p_actual = possible_rows_to_mutate / len(srs)

        if p_actual < p:
            _warn_p(with_lowercase.__name__, p, p_actual)

        # select subset of rows to mutate
        p_subset_select = min(1.0, p / p_actual)
        arr_rng_vals = rng.random(size=possible_rows_to_mutate)
        srs_rows_to_mutate.loc[srs_rows_to_mutate] = arr_rng_vals < p_subset_select

        # update selected rows
        srs_out.update(srs.loc[srs_rows_to_mutate].str.lower())

        return srs_out

    def _mutate(srs_lst: list[pd.Series], p: float = 1.0) -> list[pd.Series]:
        _check_probability_in_bounds(p)
        return [_mutate_series(srs, p) for srs in srs_lst]

    return _mutate


def with_uppercase(rng: _t.Optional[np.random.Generator] = None) -> Mutator:
    """
    Mutate data from a series by converting it into uppercase.

    Returns:
        function returning list with the entries in each series converted to uppercase
    """
    if rng is None:
        rng = np.random.default_rng()

    def _mutate_series(srs: pd.Series, p: float) -> pd.Series:
        srs_out = srs.copy(deep=True)

        # limit series to strings that are not all uppercase yet
        srs_rows_to_mutate = ~srs.str.isupper()
        possible_rows_to_mutate = srs_rows_to_mutate.sum()
        p_actual = possible_rows_to_mutate / len(srs)

        if p_actual < p:
            _warn_p(with_uppercase.__name__, p, p_actual)

        # select subset of rows to mutate
        p_subset_select = min(1.0, p / p_actual)
        arr_rng_vals = rng.random(size=possible_rows_to_mutate)
        srs_rows_to_mutate.loc[srs_rows_to_mutate] = arr_rng_vals < p_subset_select

        # update selected rows
        srs_out.update(srs.loc[srs_rows_to_mutate].str.upper())

        return srs_out

    def _mutate(srs_lst: list[pd.Series], p: float = 1.0) -> list[pd.Series]:
        _check_probability_in_bounds(p)
        return [_mutate_series(srs, p) for srs in srs_lst]

    return _mutate


def with_datetime_offset(
    max_delta: int,
    unit: _t.Literal["D", "h", "m", "s"],
    dt_format: str,
    prevent_wraparound: bool = False,
    rng: _t.Optional[np.random.Generator] = None,
) -> Mutator:
    """
    Mutate data from a series by treating it as datetime information and offsetting it by random amounts.
    The delta and the unit specify which datetime field should be affected, where `D`, `h`, `m` and `s` can
    be selected for days, hours, minutes and seconds respectively.
    The datetime format must include the same format codes as specified in the `datetime` Python module for the
    `strftime` function.
    By setting `prevent_wraparound` to `True`, this mutator will not apply a mutation if it will cause an
    unrelated field to change its value, e.g. when subtracting a day from July 1st, 2001.

    Args:
        max_delta: maximum amount of units to change by
        unit: affected datetime field
        dt_format: input and output datetime format
        prevent_wraparound: `True` if unrelated fields should not be modified, `False` otherwise
        rng: random number generator to use

    Returns:
        function returning list of datetime strings with random offsets applied to them
    """
    if max_delta <= 0:
        raise ValueError(f"delta must be positive, is {max_delta}")

    if rng is None:
        rng = np.random.default_rng()

    def _mutate_series(srs: pd.Series) -> pd.Series:
        srs_dt = pd.to_datetime(srs, format=dt_format, errors="raise")
        srs_dt_out = srs_dt.copy()

        arr_rng_vals = rng.integers(
            low=1, high=max_delta, size=len(srs_dt), endpoint=True
        ) * rng.choice((-1, 1), size=len(srs_dt))

        srs_vals = pd.Series(arr_rng_vals, copy=False, index=srs_dt.index)

        for sgn in (-1, 1):
            for val in range(1, max_delta + 1):
                # compute the delta
                this_delta = sgn * val
                # wrap it into a timedelta
                this_timedelta = pd.Timedelta(this_delta, unit)
                # select all rows that have this delta applied to them
                this_srs_mask = srs_vals == this_delta
                # update rows
                srs_dt_out.loc[this_srs_mask] += this_timedelta

                # patch stuff if it wrapped around on accident
                if prevent_wraparound:
                    if unit == "D":
                        wraparound_patch_mask = srs_dt_out.dt.month != srs_dt.dt.month
                    elif unit == "h":
                        wraparound_patch_mask = srs_dt_out.dt.day != srs_dt.dt.day
                    elif unit == "m":
                        wraparound_patch_mask = srs_dt_out.dt.hour != srs_dt.dt.hour
                    elif unit == "s":
                        wraparound_patch_mask = srs_dt_out.dt.minute != srs_dt.dt.minute
                    else:
                        raise ValueError(f"unrecognized unit: `{unit}`")

                    # use original values (could probably be solved a bit more elegantly?)
                    srs_dt_out.loc[wraparound_patch_mask] = srs_dt.loc[
                        wraparound_patch_mask
                    ]

        return srs_dt_out.dt.strftime(dt_format)

    def _mutate(srs_lst: list[pd.Series]) -> list[pd.Series]:
        return [_mutate_series(srs) for srs in srs_lst]

    return _mutate


def with_generator(
    generator: Generator,
    mode: _t.Literal["prepend", "append", "replace"],
    join_with: str = " ",
    rng: _t.Optional[np.random.Generator] = None,
) -> Mutator:
    """
    Mutate data from a series by appending, prepending or replacing it with data from another generator.
    A character to join generated data with when appending or prepending can be provided.

    Args:
        generator: generator to source data from
        mode: either append, prepend or replace
        join_with: join character when appending or prepending
        rng: random number generator to use

    Returns:
        function returning list of strings that have been appended, prepended or replaced with data from a generator
    """
    if rng is None:
        rng = np.random.default_rng()

    def _mutate(srs_lst: list[pd.Series], p: float = 1.0) -> list[pd.Series]:
        # check that all series are of the same length
        srs_lst_len_set = set([len(srs) for srs in srs_lst])

        if len(srs_lst_len_set) != 1:
            raise ValueError("series do not have the same length")

        # use this length as a param for the generator later
        srs_len = srs_lst_len_set.pop()

        # check that the indices of all input series are aligned
        if len(srs_lst) > 1:
            indices_aligned = [
                (srs_lst[0].index == srs_lst[i].index).all()
                for i in range(1, len(srs_lst))
            ]

            if not all(indices_aligned):
                raise ValueError("indices of input series are not aligned")

        arr_rows_to_mutate = rng.random(size=srs_len) < p
        rows_to_mutate_count = arr_rows_to_mutate.sum()

        srs_gen_lst = generator(rows_to_mutate_count)

        # check that the generator returns as many series as provided to the mutator.
        if len(srs_lst) != len(srs_gen_lst):
            raise ValueError(
                f"generator must generate as many series as provided to the mutator: "
                f"got {len(srs_gen_lst)}, expected {len(srs_lst)}"
            )

        # align indices with the input series index. use ffill to
        # avoid nas when reindexing.
        srs_gen_lst_aligned = [
            srs.reindex(srs_lst[i].index, method="ffill")
            for i, srs in enumerate(srs_gen_lst)
        ]

        srs_lst_out = [srs.copy() for srs in srs_lst]

        # perform the actual data mutation (this is where index alignment matters)
        for i, srs_gen in enumerate(srs_gen_lst_aligned):
            if mode == "replace":
                srs_lst_out[i].loc[arr_rows_to_mutate] = srs_gen
            elif mode == "prepend":
                srs_lst_out[i].loc[arr_rows_to_mutate] = (
                    srs_gen + join_with + srs_lst_out[i][:]
                )
            elif mode == "append":
                srs_lst_out[i].loc[arr_rows_to_mutate] += join_with + srs_gen
            else:
                raise ValueError(f"invalid mode: `{mode}`")

        return srs_lst_out

    return _mutate


def _new_regex_replacement_fn(srs: pd.Series) -> _t.Callable[[re.Match], str]:
    def _replace(match: re.Match) -> str:
        span_to_repl_col_dict: dict[tuple[int, int], str] = {}

        for i in range(len(match.groups())):
            span = match.span(i + 1)  # groups start at 1
            span_to_repl_col_dict[span] = str(i + 1)

        # overwrite indexed groups with group names, if present
        for group_name in match.groupdict().keys():
            span = match.span(group_name)
            span_to_repl_col_dict[span] = group_name

        # sort by starting index
        sorted_spans = sorted(span_to_repl_col_dict.keys(), key=lambda t: t[0])

        out_str, last_idx = "", 0

        for span in sorted_spans:
            out_str += match.string[last_idx : span[0]]
            repl_col = span_to_repl_col_dict[span]

            if repl_col not in srs.index:
                raise ValueError(
                    f"match group with index `{repl_col}` is not present in CSV file"
                )

            repl_value = srs[repl_col]

            for group_name in match.groupdict().keys():
                repl_value = repl_value.replace(
                    f"(?P<{group_name}>)", match.group(group_name)
                )

            out_str += repl_value
            last_idx = span[1]

        return out_str + match.string[last_idx:]

    return _replace


def _parse_regex_flags(regex_flags_val: str) -> int:
    _flags_lookup = {"a": re.ASCII, "i": re.IGNORECASE}

    flags = 0

    for regex_flag_char in list(regex_flags_val):
        int_flag = _flags_lookup.get(regex_flag_char, 0)
        flags |= int_flag

    return flags


def with_regex_replacement_table(
    data_source: _t.Union[PathLike, str, pd.DataFrame],
    pattern_column: str = "pattern",
    flags_column: _t.Optional[str] = None,
    encoding: str = "utf-8",
    delimiter: str = ",",
    rng: _t.Optional[np.random.Generator] = None,
) -> Mutator:
    """
    Mutate data by performing regex-based substitutions sourced from a CSV file.
    This file must contain a column with the regex patterns to look for and columns for each capture group to look up
    substitutions.
    When using regular capture groups, the columns must be numbered starting with 1.
    When using named capture groups, the columns must be named after the capture groups they are supposed to substitute.

    Args:
        data_source: path to CSV file or data frame
        pattern_column: name of regex pattern column
        flags_column: name of regex flag column
        encoding: character encoding of the CSV file
        delimiter: column delimiter of the CSV file
        rng: random number generator to use

    Returns:
        function returning list with strings mutated by regex-based substitutions
    """
    if rng is None:
        rng = np.random.default_rng()

    if isinstance(data_source, pd.DataFrame):
        df = data_source
    else:
        df = pd.read_csv(
            data_source,
            encoding=encoding,
            keep_default_na=False,
            sep=delimiter,
            dtype=str,
        )

    if pattern_column not in df.columns:
        raise ValueError(f"CSV file at `{data_source}` doesn't have a pattern column")

    regexes: list[re.Pattern] = []
    regex_repl_fns: list[_t.Callable[[re.Match], str]] = []

    for _, row in df.iterrows():
        regex = re.compile(
            row[pattern_column],
            0 if flags_column is None else _parse_regex_flags(row[flags_column]),
        )

        for group_name in regex.groupindex.keys():
            if group_name not in df.columns:
                raise ValueError(
                    f"regex pattern `{regex}` contains named group `{group_name}` which is "
                    f"not present as a column in the CSV file"
                )

        regexes.append(regex)
        regex_repl_fns.append(_new_regex_replacement_fn(row))

    regex_count = len(regexes)

    def _mutate_series(srs: pd.Series) -> pd.Series:
        # count matching regexes for each row
        srs_matching_regexes = pd.Series(
            np.zeros(len(srs), dtype=np.float64), index=srs.index
        )

        # increment for each row where regex applies
        for i in range(regex_count):
            srs_matching_regexes[srs.str.match(regexes[i])] += 1

        # filter out all rows that do not have any matches
        msk_eligible_rows = srs_matching_regexes != 0
        srs_matching_regexes[msk_eligible_rows] = (
            1 / srs_matching_regexes[msk_eligible_rows]
        )

        # this is where the real mutation begins
        srs_out = srs.copy()

        for i in range(regex_count):
            # apply substitution to all rows (including those that don't match, pandas will leave those untouched)
            srs_mut = srs.str.replace(regexes[i], regex_repl_fns[i], regex=True)
            # select the rows that have been changed and sample rows in case one row has more than one matching regex
            msk_update = (srs != srs_mut) & (
                rng.random(size=len(srs)) < srs_matching_regexes
            )
            srs_out.loc[msk_update] = srs_mut.loc[msk_update]

        return srs_out

    def _mutate(srs_lst: list[pd.Series]) -> list[pd.Series]:
        return [_mutate_series(srs) for srs in srs_lst]

    return _mutate


def with_repeat(
    join_with: str = " ", rng: _t.Optional[np.random.Generator] = None
) -> Mutator:
    """
    Mutate data in a series by repeating it.
    By default, it is appended with a whitespace.

    Args:
        join_with: joining character to use, space by default
        rng: random number generator to use

    Returns:
        function returning list with repeated series values
    """

    if rng is None:
        rng = np.random.default_rng()

    def _mutate_series(srs: pd.Series, p: float) -> pd.Series:
        srs_out = srs.copy(deep=True)
        srs_rows_to_mutate = pd.Series(rng.random(size=len(srs)) < p, index=srs.index)
        srs_out.update(
            srs_out.loc[srs_rows_to_mutate]
            + join_with
            + srs_out.loc[srs_rows_to_mutate]
        )

        return srs_out

    def _mutate(srs_lst: list[pd.Series], p: float = 1.0) -> list[pd.Series]:
        _check_probability_in_bounds(p)
        return [_mutate_series(srs, p) for srs in srs_lst]

    return _mutate


_WeightedMutatorDef = tuple[_t.Union[int, float], Mutator]


def _is_weighted_mutator_tuple(
    x: object,
) -> _te.TypeGuard[_WeightedMutatorDef]:
    return (
        isinstance(x, tuple)
        and len(x) == 2
        and isinstance(x[0], (float, int))
        and callable(x[1])
    )


def _is_weighted_mutator_tuple_list(
    x: object,
) -> _te.TypeGuard[list[_WeightedMutatorDef]]:
    if not isinstance(x, list):
        return False

    return all(_is_weighted_mutator_tuple(d) for d in x)


def with_group(
    mutator_lst: _t.Union[list[Mutator], list[_WeightedMutatorDef]],
    rng: _t.Optional[np.random.Generator] = None,
) -> Mutator:
    """
    Mutate data by applying multiple mutators on it.
    The mutators are applied in the order that they are provided in to this function.
    When providing a list of mutators, each row will be affected by each mutator with an equal probability.
    When providing a list of weighted mutators, each row will be affected by each mutator with the
    specified probabilities.
    If the probabilities do not sum up to 1, an additional mutator is added which does not modify input data.

    Args:
        mutator_lst: list of mutators or weighted mutators
        rng: random number generator to use

    Returns:
        function returning list with strings modified by mutators as specified
    """
    if all(callable(m) for m in mutator_lst):
        p = 1.0 / len(mutator_lst)
        mutator_lst = [(p, m) for m in mutator_lst]

    if not _is_weighted_mutator_tuple_list(mutator_lst):
        raise ValueError(
            "invalid argument, must be a list of mutators or weighted mutators"
        )

    p_sum = sum(t[0] for t in mutator_lst)

    if p_sum > 1:
        raise ValueError(f"sum of weights must not be higher than 1, is {p_sum}")

    if p_sum <= 0:
        raise ValueError(f"sum of weights must be higher than 0, is {p_sum}")

    # pad probabilities
    if p_sum != 1:
        mutator_lst.append((1 - p_sum, with_noop()))

    if rng is None:
        rng = np.random.default_rng()

    p_vals: tuple[_t.Union[int, float], ...]
    mut_lst: tuple[Mutator, ...]
    p_vals, mut_lst = zip(*mutator_lst)

    for mut_idx, p in enumerate(p_vals):
        if p <= 0:
            raise ValueError(
                f"weight of mutator at index {mut_idx} must be higher than zero, is {p}"
            )

    def _mutate(srs_lst: list[pd.Series], p: float = 1.0) -> list[pd.Series]:
        _check_probability_in_bounds(p)

        # check that all series have the same length
        if len(set(len(s) for s in srs_lst)) != 1:
            raise ValueError("series do not have the same length")

        srs_len = len(srs_lst[0])
        srs_lst_out = [srs.copy() for srs in srs_lst]

        # each row gets an index of the applied mutator
        arr_rows_to_mutate = rng.random(size=srs_len) < p
        arr_mut_idx = np.arange(len(mutator_lst))
        arr_mut_per_row = rng.choice(arr_mut_idx, p=p_vals, size=srs_len)

        # iterate over each mutator
        for i in arr_mut_idx:
            mutator = mut_lst[i]
            # select all rows that have this mutator applied to it
            msk_this_mut = arr_rows_to_mutate & (arr_mut_per_row == i)
            srs_mut_lst = mutator([srs[msk_this_mut] for srs in srs_lst_out], 1.0)

            for j, srs_mut in enumerate(srs_mut_lst):
                srs_lst_out[j].update(srs_mut)

        return srs_lst_out

    return _mutate


_MutatorDef = _t.Union[Mutator, tuple[_t.Union[int, float], Mutator]]
_MutatorSpec = list[
    tuple[_t.Union[str, tuple[str, ...]], _t.Union[_MutatorDef, list[_MutatorDef]]]
]


def mutate_data_frame(
    df_in: pd.DataFrame,
    mutator_lst: _MutatorSpec,
    rng: _t.Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Mutate a data frame by applying several mutators on select columns.
    This function takes a list which contains columns and mutators that are assigned to them.
    A column may be assigned a single mutator, a mutator with a probability, a list of mutators where each is applied
    with the same probability, and a list of weighted mutators where each is applied with its assigned probability.

    Args:
        df_in: data frame to mutate
        mutator_lst: list of columns with their mutator assignments
        rng: random number generator to use

    Returns:
        data frame with columns mutated as specified
    """

    if rng is None:
        rng = np.random.default_rng()

    df_out = df_in.copy()

    for col_to_mut_def in mutator_lst:
        column_spec, mutator_spec = col_to_mut_def

        # convert to list if there is only one column specified
        if isinstance(column_spec, str):
            column_spec = (column_spec,)

        # check that each column name is valid
        for column_name in column_spec:
            if column_name not in df_out.columns:
                raise ValueError(
                    f"column `{column_name}` does not exist, must be one of `{','.join(df_in.columns)}`"
                )

        # if the column is assigned a mutator, assign it a 100% weight and wrap it into a list
        if callable(mutator_spec):
            mutator_spec = [(1.0, mutator_spec)]

        # if the column is assigned a tuple, wrap it into a list
        if _is_weighted_mutator_tuple(mutator_spec):
            mutator_spec = [mutator_spec]

        # next step is to check the entries of the list. so if the spec has not been converted
        # to a list yet, then something went wrong.
        if not isinstance(mutator_spec, list):
            raise ValueError(
                f"invalid type `{type(mutator_spec)}` for mutator definition "
                f"of column `{', '.join(column_spec)}`"
            )

        # if the list contains functions only, create them into tuples with equal probability
        if all(callable(c) for c in mutator_spec):
            mutator_spec = [
                (1.0 / len(mutator_spec), mutator) for mutator in mutator_spec
            ]

        # if the end result is not a list of weighted mutators for each column, abort
        if not _is_weighted_mutator_tuple_list(mutator_spec):
            raise ValueError("malformed mutator definition")

        srs_lst_out = [df_out[column_name] for column_name in column_spec]

        for weighted_mut in mutator_spec:
            mut_p, mut_fn = weighted_mut

            if mut_p <= 0 or mut_p > 1:
                raise ValueError("probability for mutator must be in range of (0, 1]")

            mut_grp = with_group([weighted_mut], rng=rng)
            srs_lst_out = mut_grp(srs_lst_out)

        for mut_srs_idx, mut_srs in enumerate(srs_lst_out):
            col_name = column_spec[mut_srs_idx]
            df_out[col_name] = mut_srs

    return df_out
