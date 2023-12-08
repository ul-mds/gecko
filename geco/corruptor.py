import csv
import string
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Callable, Optional, Union, Literal, NamedTuple

import numpy as np
import pandas as pd
from lxml import etree
from numpy.random import Generator

from geco.cldr import decode_iso_kb_pos, unescape_kb_char, get_neighbor_kb_pos_for

CorruptorFunc = Callable[[pd.Series], pd.Series]
PhoneticFlag = Literal["start", "end", "middle"]
_EditOp = Literal["ins", "del", "sub", "trs"]


class PhoneticReplacementRule(NamedTuple):
    pattern: str
    replacement: str
    flags: list[PhoneticFlag]


def _check_probability_in_bounds(p: float):
    if p < 0 or p > 1:
        raise ValueError("probability is out of range, must be between 0 and 1")


@dataclass(frozen=True)
class KeyMutation:
    row: list[str] = field(default_factory=list)
    col: list[str] = field(default_factory=list)


def with_cldr_keymap_file(
    cldr_path: Union[PathLike, str],
    rng: Optional[Generator] = None,
) -> CorruptorFunc:
    """
    Corrupt a series of strings by randomly introducing typos.
    Potential typos are sourced from a Common Locale Data Repository (CLDR) keymap.
    Any character may be replaced with one of its horizontal or vertical neighbors on a keyboard.
    They may also be replaced with its upper- or lowercase variant.
    It is possible for a string to not be modified if a selected character has no possible replacements.

    :param cldr_path: path to CLDR keymap file
    :param rng: random number generator to use (default: None)
    :return: function returning Pandas series of strings with random typos
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

    def _corrupt(srs_str_in: pd.Series) -> pd.Series:
        srs_str_out = srs_str_in.copy()
        str_count = len(srs_str_out)

        # string length series
        srs_str_out_len = srs_str_out.str.len()
        # random indices
        arr_rng_vals = rng.random(size=str_count)
        arr_rng_typo_indices = np.floor(srs_str_out_len * arr_rng_vals).astype(int)

        # create a new series containing the chars that have been randomly selected for replacement
        srs_typo_chars = pd.Series(dtype=str, index=srs_str_out.index)
        arr_uniq_idx = arr_rng_typo_indices.unique()

        for i in arr_uniq_idx:
            idx_mask = arr_rng_typo_indices == i
            srs_typo_chars[idx_mask] = srs_str_out[idx_mask].str[i]

        # create a new series that will track the replacement chars for the selected chars
        srs_repl_chars = pd.Series(dtype=str, index=srs_str_out.index)
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
            idx_mask = (arr_rng_typo_indices == i) & pd.notna(srs_typo_chars)
            srs_str_out[idx_mask] = (
                srs_str_out[idx_mask].str[:i]
                + srs_repl_chars[idx_mask]
                + srs_str_out[idx_mask].str[i + 1 :]
            )

        return srs_str_out

    return _corrupt


def with_phonetic_replacement_table(
    csv_file_path: Union[PathLike, str],
    header: bool = False,
    encoding: str = "utf-8",
    delimiter: str = ",",
    pattern_column: Union[int, str] = 0,
    replacement_column: Union[int, str] = 1,
    flags_column: Union[int, str] = 2,
    rng: Optional[Generator] = None,
) -> CorruptorFunc:
    def _parse_flags(flags_str: Optional[str]) -> list[PhoneticFlag]:
        if pd.isna(flags_str) or flags_str == "" or flags_str is None:
            return ["start", "end", "middle"]

        flags_list: list[PhoneticFlag] = []

        for char in flags_str:
            if char == "^":
                flags_list.append("start")
            elif char == "$":
                flags_list.append("end")
            elif char == "_":
                flags_list.append("middle")
            else:
                raise ValueError(f"unknown flag: {char}")

        return flags_list

    if rng is None:
        rng = np.random.default_rng()

    # read csv file
    df = pd.read_csv(
        csv_file_path,
        header=0 if header else None,
        dtype=str,
        usecols=[pattern_column, replacement_column, flags_column],
        sep=delimiter,
        encoding=encoding,
    )

    # test
    phonetic_replacement_rules: list[PhoneticReplacementRule] = []

    for _, row in df.iterrows():
        pattern = row[pattern_column]
        replacement = row[replacement_column]
        flags = _parse_flags(row[flags_column])

        phonetic_replacement_rules.append(
            PhoneticReplacementRule(pattern, replacement, flags)
        )

    def _corrupt(srs_str_in: pd.Series) -> pd.Series:
        def _corrupt_single(str_in: str) -> str:
            # noinspection PyTypeChecker
            rng.shuffle(phonetic_replacement_rules)

            for rule in phonetic_replacement_rules:
                max_pattern_idx = len(str_in) - len(rule.pattern)
                pattern_idx = str_in.find(rule.pattern)

                if pattern_idx == -1:
                    continue

                if pattern_idx == 0:
                    if "start" not in rule.flags:
                        continue
                elif pattern_idx == max_pattern_idx:
                    if "end" not in rule.flags:
                        continue
                elif "middle" not in rule.flags:
                    continue

                return str_in.replace(rule.pattern, rule.replacement)

            return str_in

        return srs_str_in.map(_corrupt_single)

    return _corrupt


def with_replacement_table(
    csv_file_path: Union[PathLike, str],
    header: bool = False,
    encoding: str = "utf-8",
    delimiter: str = ",",
    rng: Optional[Generator] = None,
    p: float = 0.1,
) -> CorruptorFunc:
    # TODO allow selection of source columns
    # TODO remove p
    _check_probability_in_bounds(p)

    if rng is None:
        rng = np.random.default_rng()

    mut_dict: dict[str, list[str]] = {}

    with Path(csv_file_path).open(mode="r", encoding=encoding, newline="") as f:
        # csv reader instance
        reader = csv.reader(f, delimiter=delimiter)

        # skip header if necessary
        if header:
            next(reader)

        for line in reader:
            if len(line) != 2:
                raise ValueError("CSV file must contain two columns")

            line_from, line_to = line[0], line[1]

            if line_from not in mut_dict:
                mut_dict[line_from] = []

            mut_dict[line_from].append(line_to)

    # keep track of all strings that can be mutated
    mutable_str_list = list(mut_dict.keys())

    def _corrupt(srs_str_in: pd.Series) -> pd.Series:
        # create copy of input series
        srs_str_out = srs_str_in.copy()
        # keep track of strs that have already been mutated
        mutated_mask = np.full(len(srs_str_in), False)
        # find() returns -1 when a substring wasn't found, so create an array to quickly compare against
        not_found_mask = np.full(len(srs_str_in), -1)
        # create randomized mask s.t. every string has a probability of `p` of being mutated
        rand_mask = rng.choice([False, True], len(srs_str_in), p=[p, 1 - p])
        # hack for now to move to pd series as quickly as possible
        str_in_list = srs_str_in.to_list()

        for mutable_str in mutable_str_list:
            # find index of mutable str within list of input strings
            mutable_str_idx_list = np.char.find(str_in_list, mutable_str)
            # this will create an array where every string in the input list will have its corresponding
            # index set to `True` if substr is not present (will be masked), or `False` if it is (will not be masked)
            mutable_str_mask = np.equal(mutable_str_idx_list, not_found_mask)
            # perform an AND s.t. strings that have been mutated aren't mutated again
            mutable_str_mask = mutated_mask | mutable_str_mask
            mutable_str_mask = mutable_str_mask | rand_mask
            # now mask the input list s.t. we get the elements that are supposed to be mutated
            str_in_list_masked = np.ma.array(str_in_list, mask=mutable_str_mask)

            for s_idx, s in np.ma.ndenumerate(str_in_list_masked):
                idx = s_idx[0]
                # perform replacement
                srs_str_out.iloc[idx] = s.replace(
                    mutable_str, rng.choice(mut_dict[mutable_str]), 1
                )
                # mark string as mutated
                mutated_mask[idx] = True

        return srs_str_out

    return _corrupt


def _corrupt_all_from_value(value: str) -> CorruptorFunc:
    def _corrupt_list(str_in_srs: pd.Series) -> pd.Series:
        str_out_srs = str_in_srs.copy()
        str_out_srs.loc[:] = value
        return str_out_srs

    return _corrupt_list


def _corrupt_only_empty_from_value(value: str) -> CorruptorFunc:
    def _corrupt_list(str_in_srs: pd.Series) -> pd.Series:
        str_out_srs = str_in_srs.copy()
        str_out_srs[str_out_srs == ""] = value
        return str_out_srs

    return _corrupt_list


def _corrupt_only_blank_from_value(value: str) -> CorruptorFunc:
    def _corrupt_list(str_in_srs: pd.Series) -> pd.Series:
        str_out_srs = str_in_srs.copy()
        str_out_srs[str_out_srs.str.strip() == ""] = value
        return str_out_srs

    return _corrupt_list


def with_missing_value(
    value: str = "",
    strategy: Literal["all", "blank", "empty"] = "blank",
) -> CorruptorFunc:
    """
    Corrupt a series of strings by replacing select entries with a representative "missing" value.
    Strings are selected for replacement depending on the chosen strategy.
    If `all`, then all strings in the series will be replaced with the missing value.
    If `blank`, then all strings that are either empty or consist of whitespace characters only will be replaced with the missing value.
    If `empty`, then all strings that are empty will be replaced with the missing value.

    :param value: "missing" value to replace select entries with (default: empty string)
    :param strategy: `all`, `blank` or `empty` to select values to replace (default: `blank`)
    :return: function returning Pandas series of strings where select entries are replaced with a "missing" value
    """
    if strategy == "all":
        return _corrupt_all_from_value(value)
    elif strategy == "blank":
        return _corrupt_only_blank_from_value(value)
    elif strategy == "empty":
        return _corrupt_only_empty_from_value(value)
    else:
        raise ValueError(f"unrecognized replacement strategy: {strategy}")


def with_insert(
    charset: str = string.ascii_letters,
    rng: Optional[Generator] = None,
) -> CorruptorFunc:
    """
    Corrupt a series of strings by inserting random characters.
    The characters are drawn from the provided charset.

    :param charset: string to sample random characters from (default: all ASCII letters)
    :param rng: random number generator to use (default: None)
    :return: function returning Pandas series of strings with randomly inserted characters
    """
    if rng is None:
        rng = np.random.default_rng()

    def _corrupt(srs_str_in: pd.Series) -> pd.Series:
        srs_str_out = srs_str_in.copy()
        str_count = len(srs_str_out)

        # get series of lengths of all strings in series
        srs_str_out_len = srs_str_out.str.len()
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
            index=srs_str_out.index,  # align index
        )
        # determine all unique random indices
        arr_uniq_idx = arr_rng_insert_indices.unique()

        for i in arr_uniq_idx:
            # select all strings with the same random insert index
            srs_idx_mask = arr_rng_insert_indices == i
            # insert character at current index
            srs_str_out[srs_idx_mask] = (
                srs_str_out[srs_idx_mask].str[:i]
                + srs_rand_chars[srs_idx_mask]
                + srs_str_out[srs_idx_mask].str[i:]
            )

        return srs_str_out

    return _corrupt


def with_delete(rng: Optional[Generator] = None) -> CorruptorFunc:
    """
    Corrupt a series of strings by randomly deleting characters.

    :param rng: random number generator to use (default: None)
    :return: function returning Pandas series of strings with randomly deleted characters
    """
    if rng is None:
        rng = np.random.default_rng()

    def _corrupt(srs_str_in: pd.Series) -> pd.Series:
        srs_str_out = srs_str_in.copy()

        # get series of string lengths
        srs_str_out_len = srs_str_out.str.len()
        # generate random indices
        arr_rng_vals = rng.random(size=len(srs_str_out))
        arr_rng_delete_indices = np.floor(srs_str_out_len * arr_rng_vals).astype(int)
        # determine unique indices
        arr_uniq_idx = arr_rng_delete_indices.unique()

        for i in arr_uniq_idx:
            # select all strings with the same random delete index
            srs_idx_mask = arr_rng_delete_indices == i
            # delete character at selected index
            srs_str_out[srs_idx_mask] = srs_str_out[srs_idx_mask].str.slice_replace(
                i, i + 1, ""
            )

        return srs_str_out

    return _corrupt


def with_transpose(rng: Optional[Generator] = None) -> CorruptorFunc:
    """
    Corrupt a series of strings by randomly swapping neighboring characters.
    Note that it is possible for the same two neighboring characters to be swapped.

    :param rng: random number generator to use (default: None)
    :return: function returning Pandas series of strings with randomly swapped neighboring characters
    """
    if rng is None:
        rng = np.random.default_rng()

    def _corrupt(srs_str_in: pd.Series) -> pd.Series:
        srs_str_out = srs_str_in.copy()

        # length of strings
        srs_str_out_len = srs_str_out.str.len()
        # generate random numbers
        arr_rng_vals = rng.random(size=len(srs_str_out))
        # -1 as neighboring char can be transposed
        arr_rng_transpose_indices = np.floor(
            (srs_str_out_len - 1) * arr_rng_vals
        ).astype(int)
        # unique indices
        arr_uniq_idx = arr_rng_transpose_indices.unique()

        for i in arr_uniq_idx:
            # select strings that have the same transposition
            srs_idx_mask = arr_rng_transpose_indices == i
            srs_str_out[srs_idx_mask] = (
                srs_str_out[srs_idx_mask].str[:i]
                + srs_str_out[srs_idx_mask].str[i + 1]
                + srs_str_out[srs_idx_mask].str[i]
                + srs_str_out[srs_idx_mask].str[i + 2 :]
            )

        return srs_str_out

    return _corrupt


def with_substitute(
    charset: str = string.ascii_letters,
    rng: Optional[Generator] = None,
) -> CorruptorFunc:
    """
    Corrupt a series of strings by replacing single characters with a new one.
    The characters are drawn from the provided charset.
    Note that it is possible for a character to be replaced by itself.

    :param charset: string to sample random characters from (default: all ASCII letters)
    :param rng: random number generator to use (default: None)
    :return: function returning Pandas series of strings with randomly inserted characters
    """
    if rng is None:
        rng = np.random.default_rng()

    def _corrupt(srs_str_in: pd.Series) -> pd.Series:
        srs_str_out = srs_str_in.copy()
        str_count = len(srs_str_out)

        # string length series
        srs_str_out_len = srs_str_out.str.len()
        # random indices
        arr_rng_vals = rng.random(size=str_count)
        arr_rng_sub_indices = np.floor(srs_str_out_len * arr_rng_vals).astype(int)
        # random substitution chars
        srs_rand_chars = pd.Series(
            rng.choice(list(charset), size=str_count),
            copy=False,  # use np array
            index=srs_str_out.index,  # align index
        )
        arr_uniq_idx = arr_rng_sub_indices.unique()

        for i in arr_uniq_idx:
            srs_idx_mask = arr_rng_sub_indices == i
            srs_str_out[srs_idx_mask] = (
                srs_str_out[srs_idx_mask].str[:i]
                + srs_rand_chars[srs_idx_mask]
                + srs_str_out[srs_idx_mask].str[i + 1 :]
            )

        return srs_str_out

    return _corrupt


def with_edit(
    p_insert: float = 0.25,
    p_delete: float = 0.25,
    p_substitute: float = 0.25,
    p_transpose: float = 0.25,
    charset: str = string.ascii_letters,
    rng: Optional[Generator] = None,
) -> CorruptorFunc:
    """
    Corrupt a series of strings by randomly applying insertion, deletion, substitution or transposition of characters.
    This corruptor works as a wrapper around the respective corruptors for the mentioned individual operations.
    The charset of allowed characters is passed on to the insertion and substitution corruptors.
    Each corruptor receives its own isolated RNG which is derived from the RNG passed into this function.
    The probabilities of each corruptor must sum up to 1.

    :param p_insert: probability of random character insertion on a string (default: 0.25, 25%)
    :param p_delete: probability of random character deletion on a string (default: 0.25, 25%)
    :param p_substitute: probability of random character substitution on a string (default: 0.25, 25%)
    :param p_transpose: probability of random character transposition on a string (default: 0.25, 25%)
    :param charset: string to sample random characters from for insertion and substitution (default: all ASCII letters)
    :param rng: random number generator to use (default: None)
    :return: function returning Pandas series of strings with randomly mutated characters
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

    # equip every corruptor with its own independent rng derived from this corruptor's rng
    rng_ins, rng_del, rng_sub, rng_trs = rng.spawn(4)
    corr_ins, corr_del, corr_sub, corr_trs = (
        with_insert(charset, rng_ins),
        with_delete(rng_del),
        with_substitute(charset, rng_sub),
        with_transpose(rng_trs),
    )

    def _corrupt_list(srs_in: pd.Series) -> pd.Series:
        str_in_edit_ops = pd.Series(
            rng.choice(edit_ops, size=len(srs_in), p=edit_ops_prob)
        )
        srs_out = srs_in.copy()

        msk_ins = str_in_edit_ops == "ins"

        if msk_ins.sum() != 0:
            srs_out[msk_ins] = corr_ins(srs_out[msk_ins])

        msk_del = str_in_edit_ops == "del"

        if msk_del.sum() != 0:
            srs_out[msk_del] = corr_del(srs_out[msk_del])

        msk_sub = str_in_edit_ops == "sub"

        if msk_sub.sum() != 0:
            srs_out[msk_sub] = corr_sub(srs_out[msk_sub])

        msk_trs = str_in_edit_ops == "trs"

        if msk_trs.sum() != 0:
            srs_out[msk_trs] = corr_trs(srs_out[msk_trs])

        return srs_out

    return _corrupt_list


def with_categorical_values(
    csv_file_path: Union[PathLike, str],
    header: bool = False,
    value_column: Union[str, int] = 0,
    encoding: str = "utf-8",
    delimiter: str = ",",
    rng: Optional[Generator] = None,
) -> CorruptorFunc:
    """
    Corrupt a series of strings by replacing it with another from a list of categorical values.
    This corruptor reads all unique values from a column within a CSV file.
    All strings within a series will be replaced with a different random value from this column.

    :param csv_file_path: CSV file to read from
    :param header: `True` if the file contains a header, `False` otherwise (default: `False`)
    :param value_column: name of column with categorical values if the file contains a header, otherwise the column index (default: `0`)
    :param encoding: character encoding of the CSV file (default: `UTF-8`)
    :param delimiter: column delimiter (default: `,`)
    :param rng: random number generator to use (default: `None`)
    :return: function returning Pandas series of strings that are replaced with a different value from a category
    """
    if rng is None:
        rng = np.random.default_rng()

    if header and type(value_column) is not str:
        raise ValueError("header present, but value column must be a string")

    # read csv file
    df = pd.read_csv(
        csv_file_path,
        header=0 if header else None,
        usecols=[value_column],
        sep=delimiter,
        encoding=encoding,
    )

    # fetch unique values
    unique_values = pd.Series(df[value_column].dropna().unique())

    def _corrupt_list(srs_in: pd.Series) -> pd.Series:
        nonlocal unique_values

        # create a new series with which the original one will be updated.
        # for starters all rows will be NaN. dtype is to avoid typecast warning.
        srs_in_update = pd.Series(np.full(len(srs_in), np.nan), copy=False, dtype=str)

        for unique_val in unique_values:
            # remove current value from list of unique values
            unique_vals_without_current = np.setdiff1d(unique_values, unique_val)
            # select all rows that equal the current value
            srs_in_matching_val = srs_in.str.fullmatch(unique_val)
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
        srs_out = srs_in.copy()
        srs_out.update(srs_in_update)

        return srs_out

    return _corrupt_list
