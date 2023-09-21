import csv
import html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from lxml import etree
from numpy.random import Generator

CorruptorFunc = Callable[[list[str]], list[str]]

_cldr_kb_base_path = Path(__file__).parent.parent / "vendor" / "cldr" / "keyboards"

# todo can this be decided on the fly?
_kb_map_max_rows = 5
_kb_map_max_cols = 15


def decode_iso_kb_pos(iso_kb_pos: str) -> (int, int):
    kb_row_dict = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}

    kb_row = kb_row_dict.get(iso_kb_pos[0])
    kb_col = int(iso_kb_pos[1:])

    return kb_row, kb_col


@dataclass(frozen=True)
class KeyMutation:
    row: list[str] = field(default_factory=list)
    col: list[str] = field(default_factory=list)


def from_cldr(
        cldr_path: Path,
        probability: float = 0.1,
        rng: Generator | None = None
):
    if probability < 0 or probability > 1:
        raise ValueError("probability is out of range, must be between 0 and 1")

    if rng is None:
        rng = np.random.default_rng()

    # get path to cldr keyboard definition file
    full_cldr_path = _cldr_kb_base_path / cldr_path

    with full_cldr_path.open(mode="r", encoding="utf-8") as f:
        tree = etree.parse(f)

    # create keymap with all fields set to an empty string at first
    kb_map = np.chararray(shape=(_kb_map_max_rows, _kb_map_max_cols,), itemsize=1, unicode=True)
    kb_map[:] = ''

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
            chr_mask = rng.choice([False, True], len(str_out), p=[probability, 1 - probability])
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


def from_replacement_table(
        csv_file_path: Path,
        header: bool = False,
        encoding: str = "utf-8",
        delimiter: str = ",",
        rng: Generator | None = None,
        probability: float = 0.1
) -> CorruptorFunc:
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
            # create randomized mask
            rand_mask = rng.choice([False, True], len(str_in_list), p=[probability, 1-probability])
            mutable_str_mask = mutable_str_mask | rand_mask
            # now mask the input list s.t. we get the elements that are supposed to be mutated
            str_in_list_masked = np.ma.array(str_in_list, mask=mutable_str_mask)

            for s_idx, s in np.ma.ndenumerate(str_in_list_masked):
                idx = s_idx[0]
                # perform replacement
                str_out_list[idx] = s.replace(mutable_str, rng.choice(mut_dict[mutable_str]), 1)
                # mark string as mutated
                mutated_mask[idx] = True

        return str_out_list

    return _corrupt
