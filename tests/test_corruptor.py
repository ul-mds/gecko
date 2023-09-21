from geco.corruptor import from_value, ReplacementStrategy


def test_from_value_replace_all():
    x = ["foo", "   ", ""]
    corr = from_value("bar", ReplacementStrategy.ALL)
    assert corr(x) == ["bar", "bar", "bar"]


def test_from_value_replace_empty():
    x = ["foo", "   ", ""]
    corr = from_value("bar", ReplacementStrategy.ONLY_EMPTY)
    assert corr(x) == ["foo", "   ", "bar"]


def test_from_value_replace_blank():
    x = ["foo", "   ", ""]
    corr = from_value("bar", ReplacementStrategy.ONLY_BLANK)
    assert corr(x) == ["foo", "bar", "bar"]
