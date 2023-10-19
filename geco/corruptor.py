import csv
import html
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union, Literal, NamedTuple

import numpy as np
import pandas as pd
from lxml import etree
from numpy.random import Generator

CorruptorFunc = Callable[[list[str]], list[str]]
PhoneticFlag = Literal["start", "end", "middle"]


class PhoneticReplacementRule(NamedTuple):
    pattern: str
    replacement: str
    flags: list[PhoneticFlag]


# todo can this be decided on the fly?
_kb_map_max_rows = 5
_kb_map_max_cols = 15


def _check_probability_in_bounds(p: float):
    if p < 0 or p > 1:
        raise ValueError("probability is out of range, must be between 0 and 1")


def decode_iso_kb_pos(iso_kb_pos: str) -> (int, int):
    kb_row_dict = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}

    kb_row = kb_row_dict.get(iso_kb_pos[0])
    kb_col = int(iso_kb_pos[1:])

    return kb_row, kb_col


@dataclass(frozen=True)
class KeyMutation:
    row: list[str] = field(default_factory=list)
    col: list[str] = field(default_factory=list)


def with_cldr_keymap_file(
    cldr_path: Path, p: float = 0.1, rng: Optional[Generator] = None
):
    _check_probability_in_bounds(p)

    if rng is None:
        rng = np.random.default_rng()

    with cldr_path.open(mode="r", encoding="utf-8") as f:
        tree = etree.parse(f)

    # create keymap with all fields set to an empty string at first
    kb_map = np.chararray(
        shape=(
            _kb_map_max_rows,
            _kb_map_max_cols,
        ),
        itemsize=1,
        unicode=True,
    )
    kb_map[:] = ""

    # this dict records the row and column for each character (lowercase only so far)
    kb_char_to_idx_dict: dict[str, (int, int)] = {}

    root = tree.getroot()
    keymap_list = root.findall("keyMap")

    for keymap in keymap_list:
        keymap_mod = keymap.get("modifiers")

        # todo account for shift modifier in the future
        if keymap_mod is not None:
            continue

        for key in keymap.findall("map"):
            # determine the row and column for a character on the specified keyboard,
            # store in keymap and in lookup dict
            key_pos, key_char = key.get("iso"), html.unescape(key.get("to"))
            key_idx = decode_iso_kb_pos(key_pos)

            kb_map[key_idx[0], key_idx[1]] = key_char
            kb_char_to_idx_dict[key_char] = (key_idx[0], key_idx[1])

    # now construct all possible mutations for each character
    kb_char_to_mut_dict: dict[str, KeyMutation] = {}

    for kb_char, kb_idx in kb_char_to_idx_dict.items():
        kb_row, kb_col = kb_idx
        km = KeyMutation()

        # check if there's a row above the current character
        if kb_row > 0:
            kb_top = kb_map[kb_row - 1, kb_col]

            if kb_top != "":
                km.row.append(str(kb_top))

        # check if there's a row beneath the current character
        if kb_row < _kb_map_max_rows - 1:
            kb_bottom = kb_map[kb_row + 1, kb_col]

            if kb_bottom != "":
                km.row.append(str(kb_bottom))

        # check if there's a column to the left of the current character
        if kb_col > 0:
            kb_left = kb_map[kb_row, kb_col - 1]

            if kb_left != "":
                km.col.append(str(kb_left))

        # check if there's a column to the right of the current character
        if kb_col < _kb_map_max_cols - 1:
            kb_right = kb_map[kb_row, kb_col + 1]

            if kb_right != "":
                km.col.append(str(kb_right))

        kb_char_to_mut_dict[kb_char] = km

    def _corrupt(str_in_list: list[str]) -> list[str]:
        def _corrupt_single(str_in: str) -> str:
            # deconstruct the string into its single characters
            str_out = list(str_in)

            # construct a mask where `False` marks a character for mutation and `True` keeps a character as-is
            chr_mask = rng.choice([False, True], len(str_out), p=[p, 1 - p])
            str_mask = np.ma.array(str_out, mask=chr_mask)

            # this iterates over all unmasked chars, meaning the ones marked for mutation
            for c_idx, c in np.ma.ndenumerate(str_mask):
                c_idx = c_idx[0]  # c_idx would be a tuple here

                # if char cannot be mutated, skip
                if c not in kb_char_to_mut_dict:
                    continue

                # get the list of characters that this char can be mutated to
                mut = kb_char_to_mut_dict[c]
                mut_chars = mut.row + mut.col

                # this shouldn't happen but better to be safe than sorry
                if len(mut_chars) != 0:
                    continue

                # draw random character
                str_out[c_idx] = rng.choice(mut_chars)

            return "".join(str_out)

        # mutate every string one by one
        return [_corrupt_single(s) for s in str_in_list]

    return _corrupt


def with_phonetic_replacement_table(
    csv_file_path: Path,
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

    def _corrupt(str_in_list: list[str]) -> list[str]:
        def _corrupt_single(str_in: str) -> str:
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

        return [_corrupt_single(s) for s in str_in_list]

    return _corrupt


def with_replacement_table(
    csv_file_path: Path,
    header: bool = False,
    encoding: str = "utf-8",
    delimiter: str = ",",
    rng: Optional[Generator] = None,
    p: float = 0.1,
) -> CorruptorFunc:
    _check_probability_in_bounds(p)

    if rng is None:
        rng = np.random.default_rng()

    mut_dict: dict[str, list[str]] = {}

    with csv_file_path.open(mode="r", encoding=encoding, newline="") as f:
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

    def _corrupt(str_in_list: list[str]) -> list[str]:
        # keep track of strs that have already been mutated
        mutated_mask = np.full(len(str_in_list), False)
        # find() returns -1 when a substring wasn't found, so create an array to quickly compare against
        not_found_mask = np.full(len(str_in_list), -1)
        # create randomized mask s.t. every string has a probability of `p` of being mutated
        rand_mask = rng.choice([False, True], len(str_in_list), p=[p, 1 - p])
        # create copy of input list
        str_out_list = str_in_list[:]

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
                str_out_list[idx] = s.replace(
                    mutable_str, rng.choice(mut_dict[mutable_str]), 1
                )
                # mark string as mutated
                mutated_mask[idx] = True

        return str_out_list

    return _corrupt


def _corrupt_all_from_value(value: str) -> CorruptorFunc:
    def _corrupt_list(str_in_list: list[str]) -> list[str]:
        return [value for _ in str_in_list]

    return _corrupt_list


def _corrupt_only_empty_from_value(value: str) -> CorruptorFunc:
    def _corrupt_list(str_in_list: list[str]) -> list[str]:
        return [str_in or value for str_in in str_in_list]

    return _corrupt_list


def _corrupt_only_blank_from_value(value: str) -> CorruptorFunc:
    def _corrupt_list(str_in_list: list[str]) -> list[str]:
        return [str_in or value for str_in in np.char.strip(str_in_list)]

    return _corrupt_list


def with_missing_value(
    value: str = "", strategy: Literal["all", "blank", "empty"] = "all"
) -> CorruptorFunc:
    if strategy == "all":
        return _corrupt_all_from_value(value)
    elif strategy == "blank":
        return _corrupt_only_blank_from_value(value)
    elif strategy == "empty":
        return _corrupt_only_empty_from_value(value)
    else:
        raise ValueError(f"unrecognized replacement strategy: {strategy}")


def with_edit(
    p_insert: float = 0,
    p_delete: float = 0,
    p_substitute: float = 0,
    p_transpose: float = 0,
    rng: Optional[Generator] = None,
    charset: str = string.ascii_letters,
) -> CorruptorFunc:
    if rng is None:
        rng = np.random.default_rng()

    edit_ops = ["ins", "del", "sub", "trs"]
    edit_ops_prob = [p_insert, p_delete, p_substitute, p_transpose]

    for p in edit_ops_prob:
        _check_probability_in_bounds(p)

    charset_lst = list(charset)

    try:
        # sanity check
        rng.choice(edit_ops, p=edit_ops_prob)
    except ValueError:
        raise ValueError("probabilities must sum up to 1.0")

    def _corrupt_single_insert(str_in: str) -> str:
        if str_in == "":
            return str_in

        c = rng.choice(charset_lst)
        i = rng.choice(len(str_in))

        return str_in[:i] + c + str_in[i:]

    def _corrupt_single_delete(str_in: str) -> str:
        if str_in == "":
            return str_in

        i = rng.choice(len(str_in))
        return str_in[:i] + str_in[i + 1 :]

    def _corrupt_single_substitute(str_in: str) -> str:
        if str_in == "":
            return str_in

        # select random character to substitute
        i = rng.choice(len(str_in))
        c = str_in[i]
        # draw a random character from the charset minus the character to be substituted
        x = rng.choice(list(charset.replace(c, "")))
        return str_in[:i] + x + str_in[i + 1 :]

    def _corrupt_single_transpose(str_in: str) -> str:
        if len(str_in) < 2:
            return str_in

        # trivial case
        if len(str_in) == 2:
            return str_in[1] + str_in[0]

        # collect all indices 0...n-2 because n-1 may need to be selected later
        idx_shuffle = np.arange(len(str_in) - 1)
        rng.shuffle(idx_shuffle)

        for idx_0 in idx_shuffle:
            chr_0 = str_in[idx_0]

            # get neighboring character
            idx_1 = idx_0 + 1
            chr_1 = str_in[idx_1]

            # check if the characters are distinct from one another
            if chr_0 != chr_1:
                return str_in[:idx_0] + chr_1 + chr_0 + str_in[idx_1 + 1 :]

        # otherwise the string is composed of one character only and there's nothing to transpose
        return str_in

    def _corrupt_list(str_in_list: list[str]) -> list[str]:
        str_in_edit_ops = rng.choice(edit_ops, size=len(str_in_list), p=edit_ops_prob)
        str_out_list = str_in_list[:]

        for e_idx, edit_op in np.ndenumerate(str_in_edit_ops):
            idx = e_idx[0]
            str_in = str_in_list[idx]

            if edit_op == "ins":
                str_out_list[idx] = _corrupt_single_insert(str_in)
            elif edit_op == "del":
                str_out_list[idx] = _corrupt_single_delete(str_in)
            elif edit_op == "sub":
                str_out_list[idx] = _corrupt_single_substitute(str_in)
            elif edit_op == "trs":
                str_out_list[idx] = _corrupt_single_transpose(str_in)
            else:
                raise ValueError(f"unsupported edit operation: {edit_op}")

        return str_out_list

    return _corrupt_list


def with_categorical_values(
    csv_file_path: Path,
    header: bool = False,
    value_column: Union[str, int] = 0,
    encoding: str = "utf-8",
    delimiter: str = ",",
    rng: Optional[Generator] = None,
) -> CorruptorFunc:
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

    def _map_value(x):
        # select from all values without the one that's supposed to be mapped
        value_options = unique_values[~unique_values.isin([x])]
        # randomly pick a value from the resulting list
        return rng.choice(value_options)

    def _corrupt_list(str_in_list: list[str]) -> list[str]:
        nonlocal unique_values
        str_in_srs = pd.Series(str_in_list)
        return list(str_in_srs.map(_map_value))

    return _corrupt_list
