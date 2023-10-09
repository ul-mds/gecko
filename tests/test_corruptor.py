from geco.corruptor import with_missing_value, ReplacementStrategy, with_edit, with_categorical_values
from tests.helpers import get_asset_path


def test_with_value_replace_all():
    x = ["foo", "   ", ""]
    corr = with_missing_value("bar", ReplacementStrategy.ALL)
    assert corr(x) == ["bar", "bar", "bar"]


def test_with_value_replace_empty():
    x = ["foo", "   ", ""]
    corr = with_missing_value("bar", ReplacementStrategy.ONLY_EMPTY)
    assert corr(x) == ["foo", "   ", "bar"]


def test_with_value_replace_blank():
    x = ["foo", "   ", ""]
    corr = with_missing_value("bar", ReplacementStrategy.ONLY_BLANK)
    assert corr(x) == ["foo", "bar", "bar"]


def test_with_edit_insert(rng):
    x = ["foobar"]
    corr = with_edit(p_insert=1, rng=rng, charset="0")
    x_corr = corr(x)

    assert x != x_corr
    # check that a character has been inserted
    assert len(x[0]) == len(x_corr[0]) - 1
    # check that the character from the input charset has been added
    assert "0" in x_corr[0]


def test_with_edit_delete(rng):
    x = ["foobar"]
    corr = with_edit(p_delete=1, rng=rng)
    x_corr = corr(x)

    assert x != x_corr
    # check that a character has been deleted
    assert len(x[0]) == len(x_corr[0]) + 1


def test_with_edit_substitute(rng):
    x = ["foobar"]
    corr = with_edit(p_insert=0, p_delete=0, p_substitute=1, p_transpose=0, rng=rng, charset="0")
    x_corr = corr(x)

    assert x != x_corr
    # check that the length is the same
    assert len(x[0]) == len(x_corr[0])
    # check that the character from the input charset has been selected
    assert "0" in x_corr[0]


def test_with_edit_substitute_overlap(rng):
    x = ["foobar"]
    # all characters of "foobar" are in the charset, but a character should never be substituted with itself
    corr = with_edit(p_substitute=1, rng=rng, charset="fobar")

    # after about 1000 times we can be pretty sure this worked
    for _ in range(1000):
        x_corr = corr(x)
        assert x != x_corr


def test_with_edit_transpose(rng):
    x = ["foobar"]
    corr = with_edit(p_transpose=1, rng=rng)
    x_corr = corr(x)

    assert x != x_corr
    assert len(x[0]) == len(x_corr[0])


def test_with_categorical_values(rng):
    def _generate_gender_list():
        nonlocal rng
        return rng.choice(["m", "f", "d", "x"], size=1000)

    corr = with_categorical_values(
        get_asset_path("freq_table_gender.csv"),
        header=True, value_column="gender"
    )

    x = _generate_gender_list()
    x_corr = corr(x)

    assert len(x) == len(x_corr)

    for i in range(len(x)):
        assert x[i] != x_corr[i]
