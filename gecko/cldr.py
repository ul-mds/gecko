import html
import re
from functools import lru_cache


def uppercase_char_to_index(char: str) -> int:
    return ord(char) - 65  # ord('A') = 65, so ord('A') - 65 = 0


def unescape_kb_char(char: str) -> str:
    char = html.unescape(char)  # take care of html-escaped chars

    # in cldr files, unicode entities are formatted like \u{22}. however python can only decode them if they're
    # formatted like \u0022. the regex below scans for any cldr-formatted unicode entities and puts a capture
    # group around the hex codepoint. the codepoint is extracted, padded and returned as a unicode entity that
    # python can work with.
    def _replace_unicode_entities(match_obj: re.Match) -> str:
        padded_unicode = f"0000{match_obj.group(1)}"
        return f"\\u{padded_unicode[-4:]}"

    # but we're not done yet. python can only decode unicode entities if they come from a latin-1 encoded sequence.
    # so latin-1 -> unicode_escape to handle all unicode entities, and then latin-1 -> utf-8 to revert back to
    # python's default utf-8 charset for strings. fun!
    if char.startswith("\\u"):
        return (
            re.sub(r"\\u\{([0-9a-fA-F]+)\}", _replace_unicode_entities, char)
            .encode("latin-1")
            .decode("unicode_escape")
            .encode("latin-1")
            .decode("utf-8")
        )

    return char


@lru_cache  # i've only seen everything from A00 to E15 for now so caching this seems sensible
def decode_iso_kb_pos(iso_kb_pos_str: str) -> (int, int):
    kb_row_str = iso_kb_pos_str[0]
    kb_col_str = iso_kb_pos_str[1:]

    return uppercase_char_to_index(kb_row_str), int(kb_col_str)


def get_neighbor_kb_pos_for(
    kb_pos: tuple[int, int, int], max_row: int, max_col: int
) -> list[tuple[int, int, int]]:
    kb_neighbors: list[tuple[int, int, int]] = []
    _kb_row, _kb_col, _kb_mod = kb_pos

    if _kb_row > 0:
        kb_neighbors.append((_kb_row - 1, _kb_col, _kb_mod))

    if _kb_col > 0:
        kb_neighbors.append((_kb_row, _kb_col - 1, _kb_mod))

    if _kb_row < max_row:
        kb_neighbors.append((_kb_row + 1, _kb_col, _kb_mod))

    if _kb_col < max_col:
        kb_neighbors.append((_kb_row, _kb_col + 1, _kb_mod))

    # flip the kb modifier, can only be 0 or 1 anyway
    kb_neighbors.append((_kb_row, _kb_col, _kb_mod ^ 1))

    return kb_neighbors
