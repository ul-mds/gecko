import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng_factory():
    def _supply_rng():
        return np.random.default_rng(727)

    return _supply_rng


@pytest.fixture()
def rng(rng_factory):
    return rng_factory()
