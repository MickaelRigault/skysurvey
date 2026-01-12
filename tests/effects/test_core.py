import pytest
import sncosmo
from skysurvey.effects.core import Effect
from skysurvey.effects import milkyway 
from skysurvey.effects import hostdust
from skysurvey.effects import scatter

# tests for the class Effect
def test_effect_default_instantiation():
    effect = Effect()

    assert effect.effect is None
    assert effect.name is None
    assert effect.frame is None
    assert effect.model == {}

def test_effect_instantiation():
    dummy = object
    model = {"a":1}

    effect = Effect(effect=dummy, name="mw", frame="obs", model=model)

    assert effect.effect is dummy
    assert effect.name == "mw"
    assert effect.frame == "obs"
    assert effect.model == model

# tests for the func from_sncosmo
def test_from_cosmo():
    dummy = object
    model = {"a":1}

    effect = Effect.from_sncosmo(effect=dummy, name="mw", frame="obs", model=model)

    assert isinstance(effect, Effect)
    assert effect.effect is dummy
    assert effect.name == "mw"
    assert effect.frame == "obs"
    assert effect.model == model

# tests for the func from_name
def test_from_name_mw():
    effect = Effect.from_name("mw", which="ccm89")

    assert isinstance(effect, Effect)
    assert isinstance(effect.effect, sncosmo.CCM89Dust)

    assert effect.name == "mw"
    assert effect.frame == "obs"
    assert effect.model == milkyway.mwebv_model

def test_from_name_mw_invalid():
    with pytest.raises(NotImplementedError):
        Effect.from_name("mw", which="unknown")

def test_from_name_hostdust():
    effect = Effect.from_name("hostdust")

    assert isinstance(effect, Effect)
    assert isinstance(effect.effect, sncosmo.CCM89Dust)

    assert effect.name == "host"
    assert effect.frame == "rest"
    assert effect.model == hostdust.dust_model

def test_from_name_hostdust_invalid():
    with pytest.raises(NotImplementedError):
        Effect.from_name("hostdust", which="unknown")

def test_from_name_scatter_g10():
    effect = Effect.from_name("scatter", which="g10")

    assert isinstance(effect, Effect)
    assert isinstance(effect.effect, scatter.ColorScatter_G10)

    assert effect.name == "colorscatter"
    assert effect.frame == "rest"
    assert effect.model == {}

def test_from_name_scatter_c11():
    effect = Effect.from_name("scatter", which="c11")

    assert isinstance(effect, Effect)
    assert isinstance(effect.effect, scatter.ColorScatter_C11)

    assert effect.name == "colorscatter"
    assert effect.frame == "rest"
    assert effect.model == {}

def test_from_name_scatter_invalid():
    with pytest.raises(ValueError):
        Effect.from_name("scatter", which="unknown")

def test_from_name_invalid():
    with pytest.raises(NotImplementedError):
        Effect.from_name("unknown")

# tests for the funcs __repr__ and __str__ 
def test_repr_and_str():
    effect = Effect()
    assert isinstance(str(effect), str)
    assert isinstance(repr(effect), str)