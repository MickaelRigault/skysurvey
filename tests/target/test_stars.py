import pytest
import numpy as np
from skysurvey.target.stars import StableTarget, Star

# tests for the class StableTarget
def test_stabletarget_instantiation():

    stable = StableTarget()

    assert stable.__class__.__name__ == "StableTarget"

def test_stabletarget_attributes():

    stable = StableTarget()

    assert hasattr(stable, "_KIND")
    assert stable._KIND == "stable"

    assert "_MODEL" in dir(stable)
    assert "radec" in stable._MODEL
    assert "magobs" in stable._MODEL

sizes = [None, 5]
@pytest.mark.parametrize("size", sizes)
def test_stabletarget_random_magobs_ouput(size):
    magobs = StableTarget.random_magobs(size=size, zpmax=22.5, scale=3, rng=None)

    if size is None:
        assert np.isscalar(magobs)
    else:
        assert len(magobs) == size
        assert isinstance(magobs, np.ndarray)

    if size is None:
        assert magobs <= 22.5
    else:
        assert np.all(magobs <= 22.5)

@pytest.mark.parametrize("size", sizes)
def test_stabletarget_random_magobs_seed(size):
    rng_seed = 23
    magobs1 = StableTarget.random_magobs(size=size, zpmax=22.5, scale=3, rng=rng_seed)
    magobs2 = StableTarget.random_magobs(size=size, zpmax=22.5, scale=3, rng=rng_seed)

    if size is None:
        assert np.isclose(magobs1, magobs2)
    else:
        assert np.all(np.isclose(magobs1, magobs2))

#tests for the class Star
def test_star_instantiation():

    star = Star()

    assert star.__class__.__name__ == "Star"

def test_star_attributes():

    star = Star()

    assert hasattr(star, "_KIND")
    assert star._KIND == "star"