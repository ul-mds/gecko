import pytest

from gecko import _cldr


def test_uppercase_char_to_index():
    assert _cldr.uppercase_char_to_index("A") == 0
    assert _cldr.uppercase_char_to_index("B") == 1  # and so on ...
    assert _cldr.uppercase_char_to_index("Z") == 25


def test_decode_iso_kb_pos():
    assert _cldr.decode_iso_kb_pos("A00") == (0, 0)
    assert _cldr.decode_iso_kb_pos("B00") == (1, 0)
    assert _cldr.decode_iso_kb_pos("A01") == (0, 1)
    assert _cldr.decode_iso_kb_pos("B15") == (1, 15)


def test_unescape_kb_char():
    assert _cldr.unescape_kb_char("A") == "A"  # stays the same
    assert _cldr.unescape_kb_char("&amp;") == "&"  # html entities should be decoded
    assert _cldr.unescape_kb_char("\\u{22}") == '"'  # unicode entities should be decoded
    # this case caused the char.startswith("\\u") check because there's some funny
    # stuff that happens when you run special chars through many encodings
    assert _cldr.unescape_kb_char("ß") == "ß"


@pytest.mark.parametrize(
    "kb_pos,max_row,max_col,expected_kb_pos_list",
    [
        # horizontal and vertical neighbor, shift off
        ((1, 1, 0), 5, 14, [(1, 1, 1), (0, 1, 0), (2, 1, 0), (1, 0, 0), (1, 2, 0)]),
        # horizontal and vertical neighbor, shift on
        ((1, 1, 1), 5, 14, [(1, 1, 0), (0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1)]),
        # no left neighbor, shift off
        ((1, 0, 0), 5, 14, [(1, 0, 1), (0, 0, 0), (2, 0, 0), (1, 1, 0)]),
        # no top neighbor, shift off
        ((0, 1, 0), 5, 14, [(0, 1, 1), (0, 0, 0), (0, 2, 0), (1, 1, 0)]),
        # no right neighbor, shift off
        ((1, 14, 0), 5, 14, [(1, 14, 1), (0, 14, 0), (2, 14, 0), (1, 13, 0)]),
        # no bottom neighbor, shift off
        ((5, 13, 0), 5, 14, [(5, 13, 1), (5, 14, 0), (5, 12, 0), (4, 13, 0)]),
    ],
    ids=[
        # t = top, r = right, b = bottom, l = left
        "trbl-noshift",
        "trbl-shift",
        "trb-noshift",
        "rbl-noshift",
        "blt-noshift",
        "ltb-noshift",
    ],
)
def test_get_neighbor_kb_pos_for(kb_pos, max_row, max_col, expected_kb_pos_list):
    actual_kb_pos_list = _cldr.get_neighbor_kb_pos_for(kb_pos, max_row, max_col)

    assert len(actual_kb_pos_list) == len(expected_kb_pos_list)

    for expected_kb_pos in expected_kb_pos_list:
        assert expected_kb_pos in actual_kb_pos_list
