from geco.corruptor import with_missing_value, ReplacementStrategy


def test_from_value_replace_all():
    x = ["foo", "   ", ""]
    corr = with_missing_value("bar", ReplacementStrategy.ALL)
    assert corr(x) == ["bar", "bar", "bar"]


def test_from_value_replace_empty():
    x = ["foo", "   ", ""]
    corr = with_missing_value("bar", ReplacementStrategy.ONLY_EMPTY)
    assert corr(x) == ["foo", "   ", "bar"]


def test_from_value_replace_blank():
    x = ["foo", "   ", ""]
    corr = with_missing_value("bar", ReplacementStrategy.ONLY_BLANK)
    assert corr(x) == ["foo", "bar", "bar"]
