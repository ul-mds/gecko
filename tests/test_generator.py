from geco import generator
from tests.helpers import get_asset_path


def test_from_function():
    flag = False

    def _generator() -> str:
        nonlocal flag
        flag = not flag

        return "foo" if flag else "bar"

    generate_foobar = generator.from_function(_generator)
    foobar_list = generate_foobar(4)

    assert foobar_list == [["foo", "bar", "foo", "bar"]]


def test_from_frequency_table_no_header(rng, foobar_freq_head):
    generate_tab = generator.from_frequency_table(get_asset_path("freq_table_no_header.csv"), rng=rng)
    h = generate_tab(len(foobar_freq_head))[0]
    assert (h == foobar_freq_head).all()


def test_from_frequency_table_with_header(rng, foobar_freq_head):
    generate_tab = generator.from_frequency_table(get_asset_path("freq_table_header.csv"), rng=rng, header=True)
    h = generate_tab(len(foobar_freq_head))[0]
    assert (h == foobar_freq_head).all()


def test_from_frequency_table_tsv(rng, foobar_freq_head):
    generate_tab = generator.from_frequency_table(get_asset_path("freq_table_no_header.tsv"), rng=rng, delimiter="\t")
    h = generate_tab(len(foobar_freq_head))[0]
    assert (h == foobar_freq_head).all()
