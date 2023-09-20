import numpy as np
import pytest


@pytest.fixture()
def rng():
    return np.random.default_rng(727)


@pytest.fixture(scope="session")
def foobar_freq_head():
    return ["foo", "bar", "foo", "foo", "foo", "bar", "bar", "foo", "foo", "foo"]
