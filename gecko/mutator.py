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
    "with_edit",
    "with_noop",
    "with_categorical_values",
    "with_function",
    "with_permute",
    "mutate_data_frame",
]

import itertools
import string
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Callable, Optional, Union, Literal, NamedTuple

import numpy as np
import pandas as pd
from lxml import etree
from typing_extensions import ParamSpec, Concatenate

from gecko.cldr import decode_iso_kb_pos, unescape_kb_char, get_neighbor_kb_pos_for

Mutator = Callable[[list[pd.Series]], list[pd.Series]]
_EditOp = Literal["ins", "del", "sub", "trs"]


class _PhoneticReplacementRule(NamedTuple):
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


P = ParamSpec("P")


def with_function(
    func: Callable[Concatenate[str, P], str],
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

    def _mutate_series(srs: pd.Series) -> pd.Series:
        srs_out = srs.copy()

        for i in range(len(srs_out)):
            srs_out.iloc[i] = func(srs_out.iloc[i], *args, **kwargs)

        return srs_out

    def _mutate(srs_lst: list[pd.Series]) -> list[pd.Series]:
        return [_mutate_series(srs) for srs in srs_lst]

    return _mutate


def with_cldr_keymap_file(
    cldr_path: Union[PathLike, str],
    charset: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
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
                        kb_char_candidates
                    )  # needs to be sorted to ensure reproducibility
                )

    def _mutate_series(srs: pd.Series) -> pd.Series:
        srs_out = srs.copy()
        str_count = len(srs_out)

        # string length series
        srs_str_out_len = srs_out.str.len()
        # random indices
        arr_rng_vals = rng.random(size=str_count)
        arr_rng_typo_indices = np.floor(srs_str_out_len * arr_rng_vals).astype(int)

        # create a new series containing the chars that have been randomly selected for replacement
        srs_typo_chars = pd.Series(dtype=str, index=srs_out.index)
        arr_uniq_idx = arr_rng_typo_indices.unique()

        for i in arr_uniq_idx:
            idx_mask = arr_rng_typo_indices == i
            srs_typo_chars[idx_mask] = srs_out[idx_mask].str[i]

        # create a new series that will track the replacement chars for the selected chars
        srs_repl_chars = pd.Series(dtype=str, index=srs_out.index)
        arr_uniq_chars = srs_typo_chars.unique()

        for char in arr_uniq_chars:
            # check if there are any possible replacements for this char
            if char not in kb_char_to_candidates_dict:
                continue

            # get candidate strings
            char_candidates = kb_char_to_candidates_dict[char]
            # count the rows that have this character selected
            char_count = (srs_typo_chars == char).sum()
            # draw replacements for the current character
            rand_chars = rng.choice(list(char_candidates), size=char_count)
            srs_repl_chars[srs_typo_chars == char] = rand_chars

        for i in arr_uniq_idx:
            # there is a possibility that a char might not have a replacement, so pd.notna() will have to
            # act as an extra filter to not modify strings that have no replacement
            idx_mask = (arr_rng_typo_indices == i) & pd.notna(srs_repl_chars)
            srs_out[idx_mask] = (
                srs_out[idx_mask].str[:i]
                + srs_repl_chars[idx_mask]
                + srs_out[idx_mask].str[i + 1 :]
            )

        return srs_out

    def _mutate(srs_lst: list[pd.Series]) -> list[pd.Series]:
        return [_mutate_series(srs) for srs in srs_lst]

    return _mutate


def with_phonetic_replacement_table(
    csv_file_path: Union[PathLike, str],
    source_column: Union[int, str] = 0,
    target_column: Union[int, str] = 1,
    flags_column: Union[int, str] = 2,
    encoding: str = "utf-8",
    delimiter: str = ",",
    rng: Optional[np.random.Generator] = None,
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
        csv_file_path: path to CSV file containing phonetic replacement rules
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

    def _validate_flags(flags_str: Optional[str]) -> str:
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

    header = isinstance(flags_column, str)

    # read csv file
    df = pd.read_csv(
        csv_file_path,
        header=0 if header else None,
        dtype=str,
        usecols=[source_column, target_column, flags_column],
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
    csv_file_path: Union[PathLike, str],
    source_column: Union[str, int] = 0,
    target_column: Union[str, int] = 1,
    encoding: str = "utf-8",
    delimiter: str = ",",
    rng: Optional[np.random.Generator] = None,
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
        csv_file_path: path to CSV file
        source_column: name or index of the source column
        target_column: name or index of the target column
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

    header = isinstance(target_column, str)

    df = pd.read_csv(
        csv_file_path,
        header=0 if header else None,
        dtype=str,
        usecols=[source_column, target_column],
        sep=delimiter,
        encoding=encoding,
    )

    srs_unique_source_values = df[source_column].unique()

    def _mutate_series(srs: pd.Series) -> pd.Series:
        # create copy of input series
        srs_out = srs.copy()
        str_count = len(srs_out)
        # create series to compute probability of substitution for each row
        srs_str_sub_prob = pd.Series(dtype=float, index=srs_out.index)
        srs_str_sub_prob[:] = 0

        for source in srs_unique_source_values:
            # increment absolute frequency for each string containing source value by one
            srs_str_sub_prob[srs_out.str.contains(source)] += 1

        # prevent division by zero
        mask_eligible_strs = srs_str_sub_prob != 0
        # convert absolute frequencies into relative frequencies
        srs_str_sub_prob[mask_eligible_strs] = 1 / srs_str_sub_prob[mask_eligible_strs]

        # create dataframe to track source and target for each row
        df_replacement = pd.DataFrame(
            index=srs_out.index, columns=["source", "target"], dtype=str
        )

        for source in srs_unique_source_values:
            # select all rows that contain the source value
            srs_str_contains_source = srs_out.str.contains(source)
            # draw random numbers for each row
            arr_rand_vals = rng.random(size=str_count)
            # select only rows that contain the source string, have a random number drawn that's
            # in range of its probability to be modified, and hasn't been marked for replacement yet
            mask_strings_to_replace = (
                srs_str_contains_source
                & (arr_rand_vals < srs_str_sub_prob)
                & pd.isna(df_replacement["source"])
            )
            # count all strings that meet the conditions above
            replacement_count = mask_strings_to_replace.sum()

            # skip if there are no replacements to be made
            if replacement_count == 0:
                continue

            # fill in the source column of the replacement table
            df_replacement.loc[mask_strings_to_replace, "source"] = source

            # select all target values that can be generated from the current source value
            replacement_options = df[df[source_column] == source][
                target_column
            ].tolist()

            # trivial case
            if len(replacement_options) == 1:
                target = replacement_options[0]
                df_replacement.loc[mask_strings_to_replace, "target"] = target
                continue

            # otherwise draw a random target value for each row
            df_replacement.loc[mask_strings_to_replace, "target"] = rng.choice(
                replacement_options, size=replacement_count
            )

        # iterate over all unique source values
        for source in df_replacement["source"].unique():
            # skip nan
            if pd.isna(source):
                continue

            # for each unique source value, iterate over its unique target values
            for target in df_replacement[df_replacement["source"] == source][
                "target"
            ].unique():
                # select all rows that have this specific source -> target replacement going
                mask = (df_replacement["source"] == source) & (
                    df_replacement["target"] == target
                )

                # perform replacement of source -> target
                srs_out[mask] = srs_out[mask].str.replace(source, target, n=1)

        return srs_out

    def _mutate(srs_lst: list[pd.Series]) -> list[pd.Series]:
        return [_mutate_series(srs) for srs in srs_lst]

    return _mutate


def _mutate_all_from_value(value: str) -> Mutator:
    """
    Mutate data by replacing all of its values with the same "missing" value.

    Args:
        value: "missing" value to replace entries with

    Returns:
        function returning list where all strings will be replaced with the "missing" value
    """

    def _mutate_series(srs: pd.Series) -> pd.Series:
        return pd.Series(
            data=[value] * len(srs),
            index=srs.index,
            dtype=str,
        )

    def _mutate(srs_lst: list[pd.Series]) -> list[pd.Series]:
        return [_mutate_series(srs) for srs in srs_lst]

    return _mutate


def _mutate_only_empty_from_value(value: str) -> Mutator:
    """
    Mutate data by replacing all of its empty values (string length = 0) with the same "missing" value.

    Args:
        value: "missing" value to replace empty entries with

    Returns:
        function returning list where all empty strings will be replaced with the "missing" value
    """

    def _mutate_series(srs: pd.Series) -> pd.Series:
        srs_out = srs.copy()
        srs_out[srs_out == ""] = value
        return srs_out

    def _mutate(srs_lst: list[pd.Series]) -> list[pd.Series]:
        return [_mutate_series(srs) for srs in srs_lst]

    return _mutate


def _mutate_only_blank_from_value(value: str) -> Mutator:
    """
    Mutate data by replacing all of its blank values (empty strings after trimming whitespaces) with the same
    "missing" value.

    Args:
        value: "missing" value to replace blank entries with

    Returns:
        function returning list where all blank strings will be replaced with the "missing" value
    """

    def _mutate_series(srs: pd.Series) -> pd.Series:
        srs_out = srs.copy()
        srs_out[srs_out.str.strip() == ""] = value
        return srs_out

    def _mutate(srs_lst: list[pd.Series]) -> list[pd.Series]:
        return [_mutate_series(srs) for srs in srs_lst]

    return _mutate


def with_missing_value(
    value: str = "",
    strategy: Literal["all", "blank", "empty"] = "blank",
) -> Mutator:
    """
    Mutate data by replacing select entries with a representative "missing" value.
    Strings are selected for replacement depending on the chosen strategy.
    If `all`, then all strings in the series will be replaced with the missing value.
    If `blank`, then all strings that are either empty or consist of whitespace characters only will be replaced with
    the missing value.
    If `empty`, then all strings that are empty will be replaced with the missing value.

    Args:
        value: "missing" value to replace select entries with
        strategy: `all`, `blank` or `empty` to select values to replace

    Returns:
        function returning list where select strings will be replaced with the "missing" value
    """
    if strategy == "all":
        return _mutate_all_from_value(value)
    elif strategy == "blank":
        return _mutate_only_blank_from_value(value)
    elif strategy == "empty":
        return _mutate_only_empty_from_value(value)
    else:
        raise ValueError(f"unrecognized replacement strategy: {strategy}")


def with_insert(
    charset: str = string.ascii_letters,
    rng: Optional[np.random.Generator] = None,
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

    def _mutate_series(srs: pd.Series) -> pd.Series:
        srs_out = srs.copy()
        str_count = len(srs_out)

        # get series of lengths of all strings in series
        srs_str_out_len = srs_out.str.len()
        # draw random values
        arr_rng_vals = rng.random(size=str_count)
        # compute indices from random values (+1 because letters can be inserted at the ned)
        arr_rng_insert_indices = np.floor((srs_str_out_len + 1) * arr_rng_vals).astype(
            int
        )
        # generate random char for each string
        srs_rand_chars = pd.Series(
            rng.choice(list(charset), size=str_count),
            copy=False,  # use np array
            index=srs_out.index,  # align index
        )
        # determine all unique random indices
        arr_uniq_idx = arr_rng_insert_indices.unique()

        for i in arr_uniq_idx:
            # select all strings with the same random insert index
            srs_idx_mask = arr_rng_insert_indices == i
            # insert character at current index
            srs_out[srs_idx_mask] = (
                srs_out[srs_idx_mask].str[:i]
                + srs_rand_chars[srs_idx_mask]
                + srs_out[srs_idx_mask].str[i:]
            )

        return srs_out

    def _mutate(srs_lst: list[pd.Series]) -> list[pd.Series]:
        return [_mutate_series(srs) for srs in srs_lst]

    return _mutate


def with_delete(rng: Optional[np.random.Generator] = None) -> Mutator:
    """
    Mutate data by randomly deleting characters.

    Args:
        rng: random number generator to use

    Returns:
        function returning list with strings mutated by deleting random characters
    """
    if rng is None:
        rng = np.random.default_rng()

    def _mutate_series(srs: pd.Series) -> pd.Series:
        # get series of string lengths
        srs_str_out_len = srs.str.len()
        # limit view to strings that have at least one character
        srs_str_out_min_len = srs[srs_str_out_len >= 1]

        # check that there are any strings to modify
        if len(srs_str_out_min_len) == 0:
            return srs

        # create copy after length check
        srs_out = srs.copy()
        # generate random indices
        arr_rng_vals = rng.random(size=len(srs_str_out_min_len))
        arr_rng_delete_indices = np.floor(
            srs_str_out_min_len.str.len() * arr_rng_vals
        ).astype(int)
        # determine unique indices
        arr_uniq_idx = arr_rng_delete_indices.unique()

        for i in arr_uniq_idx:
            # select all strings with the same random delete index
            srs_idx_mask = arr_rng_delete_indices == i
            # delete character at selected index
            srs_out.update(
                srs_str_out_min_len[srs_idx_mask].str.slice_replace(i, i + 1, "")
            )

        return srs_out

    def _mutate(srs_lst: list[pd.Series]) -> list[pd.Series]:
        return [_mutate_series(srs) for srs in srs_lst]

    return _mutate


def with_transpose(rng: Optional[np.random.Generator] = None) -> Mutator:
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

    def _mutate_series(srs: pd.Series) -> pd.Series:
        # length of strings
        srs_str_out_len = srs.str.len()
        # limit view to strings that have at least two characters
        srs_str_out_min_len = srs[srs_str_out_len >= 2]

        # check that there are any strings to modify
        if len(srs_str_out_min_len) == 0:
            return srs

        # create a copy only after running the length check
        srs_out = srs.copy()
        # generate random numbers
        arr_rng_vals = rng.random(size=len(srs_str_out_min_len))

        # -1 as neighboring char can be transposed
        arr_rng_transpose_indices = np.floor(
            (srs_str_out_min_len.str.len() - 1) * arr_rng_vals
        ).astype(int)
        # unique indices
        arr_uniq_idx = arr_rng_transpose_indices.unique()

        for i in arr_uniq_idx:
            # select strings that have the same transposition
            srs_idx_mask = arr_rng_transpose_indices == i
            srs_masked = srs_str_out_min_len[srs_idx_mask]
            srs_out.update(
                srs_masked.str[:i]
                + srs_masked.str[i + 1]
                + srs_masked.str[i]
                + srs_masked.str[i + 2 :]
            )

        return srs_out

    def _mutate(srs_lst: list[pd.Series]) -> list[pd.Series]:
        return [_mutate_series(srs) for srs in srs_lst]

    return _mutate


def with_substitute(
    charset: str = string.ascii_letters,
    rng: Optional[np.random.Generator] = None,
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

    def _mutate_series(srs: pd.Series) -> pd.Series:
        # length of strings
        srs_str_out_len = srs.str.len()
        # limit view to strings that have at least 1 character
        srs_str_out_min_len = srs[srs_str_out_len >= 1]

        # check that there are any strings to modify
        if len(srs_str_out_min_len) == 0:
            return srs

        # create copy after length check
        srs_out = srs.copy()
        # count strings that may be modified
        str_count = len(srs_str_out_min_len)
        # random indices
        arr_rng_vals = rng.random(size=str_count)
        arr_rng_sub_indices = np.floor(
            srs_str_out_min_len.str.len() * arr_rng_vals
        ).astype(int)
        # random substitution chars
        srs_rand_chars = pd.Series(
            rng.choice(list(charset), size=str_count),
            copy=False,  # use np array
            index=srs_str_out_min_len.index,  # align index
        )
        arr_uniq_idx = arr_rng_sub_indices.unique()

        for i in arr_uniq_idx:
            srs_idx_mask = arr_rng_sub_indices == i
            srs_masked = srs_str_out_min_len[srs_idx_mask]
            srs_out.update(
                srs_masked.str[:i]
                + srs_rand_chars[srs_idx_mask]
                + srs_masked.str[i + 1 :]
            )

        return srs_out

    def _mutate(srs_lst: list[pd.Series]) -> list[pd.Series]:
        return [_mutate_series(srs) for srs in srs_lst]

    return _mutate


def with_edit(
    p_insert: float = 0.25,
    p_delete: float = 0.25,
    p_substitute: float = 0.25,
    p_transpose: float = 0.25,
    charset: str = string.ascii_letters,
    rng: Optional[np.random.Generator] = None,
) -> Mutator:
    """
    Mutate data by randomly applying insertion, deletion, substitution or transposition of characters.
    This mutator works as a wrapper around the respective mutators for the mentioned individual operations.
    The charset of allowed characters is passed on to the insertion and substitution mutators.
    Each mutator receives its own isolated RNG which is derived from the RNG passed into this function.
    The probabilities of each mutator must sum up to 1.

    Args:
        p_insert: probability of random character insertion on a string
        p_delete: probability of random character deletion on a string
        p_substitute: probability of random character substitution on a string
        p_transpose: probability of random character transposition on a string
        charset: string to sample random characters from for insertion and substitution
        rng: random number generator to use

    Returns:
        function returning list with strings mutated by random edit operations
    """
    if rng is None:
        rng = np.random.default_rng()

    edit_ops: list[_EditOp] = ["ins", "del", "sub", "trs"]
    edit_ops_prob = [p_insert, p_delete, p_substitute, p_transpose]

    for p in edit_ops_prob:
        _check_probability_in_bounds(p)

    try:
        # sanity check
        rng.choice(edit_ops, p=edit_ops_prob)
    except ValueError:
        raise ValueError("probabilities must sum up to 1.0")

    # equip every mutator with its own independent rng derived from this mutator's rng
    rng_ins, rng_del, rng_sub, rng_trs = rng.spawn(4)
    mut_ins, mut_del, mut_sub, mut_trs = (
        with_insert(charset, rng_ins),
        with_delete(rng_del),
        with_substitute(charset, rng_sub),
        with_transpose(rng_trs),
    )

    def _mutate_series(srs: pd.Series) -> pd.Series:
        srs_out = srs.copy()
        str_in_edit_ops = pd.Series(
            rng.choice(edit_ops, size=len(srs_out), p=edit_ops_prob),
            index=srs_out.index,
        )

        msk_ins = str_in_edit_ops == "ins"

        if msk_ins.sum() != 0:
            (srs_out[msk_ins],) = mut_ins([srs_out[msk_ins]])

        msk_del = str_in_edit_ops == "del"

        if msk_del.sum() != 0:
            (srs_out[msk_del],) = mut_del([srs_out[msk_del]])

        msk_sub = str_in_edit_ops == "sub"

        if msk_sub.sum() != 0:
            (srs_out[msk_sub],) = mut_sub([srs_out[msk_sub]])

        msk_trs = str_in_edit_ops == "trs"

        if msk_trs.sum() != 0:
            (srs_out[msk_trs],) = mut_trs([srs_out[msk_trs]])

        return srs_out

    def _mutate(srs_lst: list[pd.Series]) -> list[pd.Series]:
        return [_mutate_series(srs) for srs in srs_lst]

    return _mutate


def with_noop() -> Mutator:
    """
    Mutate data by not mutating it at all.
    This mutator returns the input series as-is.
    You might use it to leave a certain percentage of records in a series untouched.

    Returns:
        function returning list of strings as-is
    """

    def _mutate(srs_lst: list[pd.Series]) -> list[pd.Series]:
        return srs_lst

    return _mutate


def with_categorical_values(
    csv_file_path: Union[PathLike, str],
    value_column: Union[str, int] = 0,
    encoding: str = "utf-8",
    delimiter: str = ",",
    rng: Optional[np.random.Generator] = None,
) -> Mutator:
    """
    Mutate data by replacing it with another from a list of categorical values.
    This mutator reads all unique values from a column within a CSV file.
    All strings within a series will be replaced with a different random value from this column.
    If the value column is provided as a string, then it is automatically assumed that the CSV file
    has a header row.

    Args:
        csv_file_path: path to CSV file
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

    header = isinstance(value_column, str)

    # read csv file
    df = pd.read_csv(
        csv_file_path,
        header=0 if header else None,
        dtype=str,
        usecols=[value_column],
        sep=delimiter,
        encoding=encoding,
    )

    # fetch unique values
    unique_values = pd.Series(df[value_column].dropna().unique())

    def _mutate_series(srs: pd.Series) -> pd.Series:
        nonlocal unique_values

        # create a new series with which the original one will be updated.
        # for starters all rows will be NaN. dtype is to avoid typecast warning.
        srs_in_update = pd.Series(
            np.full(len(srs), np.nan), copy=False, dtype=str, index=srs.index
        )

        for unique_val in unique_values:
            # remove current value from list of unique values
            unique_vals_without_current = np.setdiff1d(unique_values, unique_val)
            # select all rows that equal the current value
            srs_in_matching_val = srs.str.fullmatch(unique_val)
            # count the rows that contain the current value
            unique_val_total = srs_in_matching_val.sum()

            # skip if there are no values to generate
            if unique_val_total == 0:
                continue

            # draw from the list of values excluding the current one
            new_unique_vals = rng.choice(
                unique_vals_without_current, size=unique_val_total
            )

            # populate the series that is used for updating the original one
            srs_in_update[srs_in_matching_val] = new_unique_vals

        # update() is performed in-place, so create a copy of the initial series first.
        srs_out = srs.copy()
        srs_out.update(srs_in_update)

        return srs_out

    def _mutate(srs_lst: list[pd.Series]) -> list[pd.Series]:
        return [_mutate_series(srs) for srs in srs_lst]

    return _mutate


def with_permute(rng: Optional[np.random.Generator] = None) -> Mutator:
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

    def _mutate(srs_lst: list[pd.Series]) -> list[pd.Series]:
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

        # e.g. for l=3, this will produce (0, 1, 2)
        tpl_idx_not_permuted = tuple(range(srs_lst_len))
        # generate all series index permutations and remove the tuple with all indices in order
        srs_idx_permutations = sorted(
            set(itertools.permutations(range(len(srs_lst)))) - {tpl_idx_not_permuted}
        )
        # choose random index permutations
        arr_rand_idx_tpl = rng.choice(srs_idx_permutations, size=srs_0_len)
        # map tuples to each series
        srs_idx_per_srs = list(zip(*arr_rand_idx_tpl))
        # transform each list of indices into series
        srs_lst_idx_per_srs = [
            pd.Series(
                data=srs_idx_per_srs[i],
                index=srs_lst[i].index,
                copy=False,
            )
            for i in range(srs_lst_len)
        ]

        srs_lst_out = [srs.copy() for srs in srs_lst]

        for i in range(srs_lst_len):
            for j in range(srs_lst_len):
                if i == j:
                    continue

                mask_this_srs = srs_lst_idx_per_srs[i] == j
                srs_lst_out[i][mask_this_srs] = srs_lst[j][mask_this_srs]

        return srs_lst_out

    return _mutate


def mutate_data_frame(
    df_in: pd.DataFrame,
    column_to_mutator_dict: dict[
        Union[str, tuple[str, ...]],
        Union[
            Mutator,
            tuple[float, Mutator],
            list[Mutator],
            list[tuple[float, Mutator]],
        ],
    ],
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Mutate a data frame by applying several mutators on select columns.
    This function takes a dictionary which has column names as keys and mutators as values.
    A column may be assigned a single mutator, a mutator with a probability, a list of mutators where each is applied
    with the same probability, and a list of weighted mutators where each is applied with its assigned probability.

    Args:
        df_in: data frame to mutate
        column_to_mutator_dict: mapping of column names to mutators
        rng: random number generator to use

    Returns:
        data frame with columns mutated as specified
    """

    def __is_weighted_mutator_tuple(x: object):
        return (
            isinstance(x, tuple)
            and len(x) == 2
            and isinstance(x[0], float)
            and isinstance(x[1], Callable)
        )

    if rng is None:
        rng = np.random.default_rng()

    df_out = df_in.copy()

    for column_spec, mutator_spec in column_to_mutator_dict.items():
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
        if isinstance(mutator_spec, Callable):
            mutator_spec = [(1.0, mutator_spec)]

        # if the column is assigned a tuple, wrap it into a list
        if __is_weighted_mutator_tuple(mutator_spec):
            mutator_spec = [mutator_spec]

        # next step is to check the entries of the list. so if the spec has not been converted
        # to a list yet, then something went wrong.
        if not isinstance(mutator_spec, list):
            raise ValueError(
                f"invalid type `{type(mutator_spec)}` for mutator definition "
                f"of column `{', '.join(column_spec)}`"
            )

        # if the list contains functions only, create them into tuples with equal probability
        if all(isinstance(c, Callable) for c in mutator_spec):
            mutator_spec = [
                (1.0 / len(mutator_spec), mutator) for mutator in mutator_spec
            ]

        # if the end result is not a list of weighted mutators for each column, abort
        if not all(__is_weighted_mutator_tuple(c) for c in mutator_spec):
            raise ValueError("malformed mutator definition")

        # mutator_spec is a list of tuples, which contain a float and a mutator func.
        # this one-liner collects all floats and mutator funcs into their own lists.
        p_values, mutator_funcs = list(zip(*mutator_spec))
        p_sum = sum(p_values)

        if p_sum > 1:
            raise ValueError(
                f"sum of probabilities may not be higher than 1.0, is {p_sum}"
            )

        # pad probabilities to sum up to 1.0
        if p_sum < 1:
            p_values = (*p_values, 1 - p_sum)
            mutator_funcs = (*mutator_funcs, with_noop())

        try:
            # sanity check
            rng.choice([i for i in range(len(p_values))], p=p_values)
        except ValueError:
            column_str = f"column{'s' if len(column_spec) > 1 else ''} `{', '.join(column_spec)}`"
            raise ValueError(f"probabilities for {column_str} must sum up to 1.0")

        mutator_count = len(mutator_funcs)
        # generate a series where each row gets an index of the mutator in mutator_funcs to apply.
        arr_mutator_idx = np.arange(mutator_count)
        arr_mutator_per_row = rng.choice(arr_mutator_idx, p=p_values, size=len(df_out))
        srs_mutator_idx = pd.Series(data=arr_mutator_per_row, index=df_out.index)
        srs_columns = [df_out[column_name] for column_name in column_spec]

        for i in arr_mutator_idx:
            mutator = mutator_funcs[i]
            mask_this_mutator = srs_mutator_idx == i
            srs_mutated_lst = mutator([srs[mask_this_mutator] for srs in srs_columns])

            # i would've liked to use .loc[mask_this_mutator, column_spec] = mutator(...) here
            # but there is apparently a shape mismatch that happens here. so instead i'm manually
            # iterating over each mutated series until i find a better solution.
            for j in range(len(column_spec)):
                column_name = column_spec[j]
                srs_mutated = srs_mutated_lst[j]

                df_out.loc[mask_this_mutator, column_name] = srs_mutated

    return df_out
