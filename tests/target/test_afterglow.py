import pytest
import sncosmo

# tests for the class Afterglow
def test_afterglow_instantiation():
    pytest.importorskip("afterglowpy")
    from skysurvey.target.afterglow import Afterglow

    ag = Afterglow()

    assert ag.__class__.__name__ == "Afterglow"

def test_afterglow_attributes():
    pytest.importorskip("afterglowpy")
    from skysurvey.target.afterglow import Afterglow

    ag = Afterglow()

    assert hasattr(ag, "_KIND")
    assert ag._KIND == "afterglow"

    assert hasattr(ag, "_RATE")
    assert ag._RATE == 20

    assert "_MODEL" in dir(ag)
    assert "redshift" in ag._MODEL
    assert "t0" in ag._MODEL

def test_afterglow_template():
    pytest.importorskip("afterglowpy")
    from skysurvey.target.afterglow import Afterglow
    
    ag = Afterglow()

    assert isinstance(ag._TEMPLATE, sncosmo.Model)
    flux = ag._TEMPLATE.flux(ag._TEMPLATE.source._phase[0], [3600])
    assert flux.shape[0] == 1