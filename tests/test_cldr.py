from gecko.cldr import (
    uppercase_char_to_index,
    decode_iso_kb_pos,
    unescape_kb_char,
    get_neighbor_kb_pos_for,
)


def test_uppercase_char_to_index():
    assert uppercase_char_to_index("A") == 0
    assert uppercase_char_to_index("B") == 1  # and so on ...
    assert uppercase_char_to_index("Z") == 25


def test_decode_iso_kb_pos():
    assert decode_iso_kb_pos("A00") == (0, 0)
    assert decode_iso_kb_pos("B00") == (1, 0)
    assert decode_iso_kb_pos("A01") == (0, 1)
    assert decode_iso_kb_pos("B15") == (1, 15)


def test_unescape_kb_char():
    assert unescape_kb_char("A") == "A"  # stays the same
    assert unescape_kb_char("&amp;") == "&"  # html entities should be decoded
    assert unescape_kb_char("\\u{22}") == '"'  # unicode entities should be decoded
    # this case caused the char.startswith("\\u") check because there's some funny
    # stuff that happens when you run special chars through many encodings
    assert unescape_kb_char("ß") == "ß"


def test_get_neighbor_kb_pos_for():
    _3i = tuple[int, int, int]
    # (kb_pos), max_row, max_col, [(kb_pos_neighbor)...]
    test_cases: list[tuple[_3i, int, int, list[_3i]]] = [
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
    ]

    for case in test_cases:
        kb_pos, max_row, max_col, expected_kb_pos_list = case
        actual_kb_pos_list = get_neighbor_kb_pos_for(kb_pos, max_row, max_col)

        assert len(actual_kb_pos_list) == len(expected_kb_pos_list)

        for expected_kb_pos in expected_kb_pos_list:
            assert expected_kb_pos in actual_kb_pos_list
