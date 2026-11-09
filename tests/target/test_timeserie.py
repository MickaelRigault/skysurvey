import numpy as np
import pandas
import pytest
import sncosmo

from astropy.cosmology import Planck18

from skysurvey.template import Template, TemplateCollection
from skysurvey.target.timeserie import TSTransient, MultiTemplateTSTransient
from skysurvey.tools.stats import skewed_gaussian_pdf


# Tiny hand-built sncosmo time-series sources: flat spectrum, gaussian in phase
# peaking at phase 0. No download needed.
PHASES = np.linspace(-20, 50, 15)
WAVES = np.linspace(2000, 10000, 50)

def make_source(name):
    flux = np.exp(-0.5 * (PHASES[:, None] / 10)**2) * np.ones((len(PHASES), len(WAVES))) * 1e-15
    return sncosmo.TimeSeriesSource(PHASES, WAVES, flux, name=name)

class OtherTimeSeriesSource( sncosmo.TimeSeriesSource ):
    """ same as TimeSeriesSource but a different class """


@pytest.fixture
def tstransient():
    return TSTransient.from_draw(size=5, template=make_source("fakeA"), magabs=[-18, 0.1])

@pytest.fixture
def multitransient():
    return MultiTemplateTSTransient.from_draw(size=20, template=[make_source("fakeA"), make_source("fakeB")],
                                              magabs=[-18, 0.1])


# tests for the class TSTransient
def test_tstransient_default_magabs():
    transient = TSTransient()
    # _MAGABS is None: loc must be given by the user (nan by default)
    assert np.isnan(transient.get_model_parameter("magabs", "loc"))
    assert transient.rate == 0.001

def test_tstransient_init_template():
    transient = TSTransient(template=make_source("fakeA"))
    assert isinstance(transient.template, Template)
    assert transient.template_source.name == "fakeA"

def test_tstransient_init_magabs():
    transient = TSTransient(magabs=[-17, 0.5])
    assert transient.get_model_parameter("magabs", "loc") == -17
    assert transient.get_model_parameter("magabs", "scale") == 0.5

def test_tstransient_class_magabs():
    class FakeTS( TSTransient ):
        _MAGABS = (-16, 0.3)

    transient = FakeTS()
    assert transient.get_model_parameter("magabs", "loc") == -16
    assert transient.get_model_parameter("magabs", "scale") == 0.3

def test_tstransient_model_has_template_parameters(tstransient):
    for key in ["z", "t0", "magabs", "magobs", "ra", "dec"]:
        assert key in tstransient.data, f"{key} missing"

# tests for the func set_magabs
def test_set_magabs_gaussian():
    transient = TSTransient()
    transient.set_magabs([-18, 0.2])
    assert transient.get_model_parameter("magabs", "loc") == -18
    assert transient.get_model_parameter("magabs", "scale") == 0.2

def test_set_magabs_skewed():
    transient = TSTransient()
    transient.set_magabs([-18, 0.2, 0.5])

    assert transient.model.model["magabs"]["func"] is skewed_gaussian_pdf
    assert transient.get_model_parameter("magabs", "scale_low") == 0.2
    assert transient.get_model_parameter("magabs", "scale_high") == 0.5
    # pdf evaluated from loc - 10 scale_low to loc + 10 scale_high
    assert transient.get_model_parameter("magabs", "xx") == "-20.0:-13.0:10000j"

def test_set_magabs_skewed_draw():
    transient = TSTransient.from_draw(size=50, template=make_source("fakeA"), magabs=[-18, 0.2, 0.5])
    assert (transient.data["magabs"] >= -20).all() and (transient.data["magabs"] <= -13).all()

def test_set_magabs_none():
    transient = TSTransient(magabs=[-17, 0.5])
    transient.set_magabs(None)
    assert transient.get_model_parameter("magabs", "loc") == -17, "None does nothing"

# tests for the func _parse_init_kwargs_
def test_parse_init_kwargs():
    init_kwargs, kwargs = TSTransient._parse_init_kwargs_(magabs=[-18, 1], t0={"low": 0})
    assert init_kwargs == {"magabs": [-18, 1]}
    assert kwargs == {"t0": {"low": 0}}

def test_parse_init_kwargs_no_magabs():
    init_kwargs, kwargs = TSTransient._parse_init_kwargs_()
    assert init_kwargs == {"magabs": None}
    assert kwargs == {}

# tests for the func from_sncosmo
def test_from_sncosmo():
    transient = TSTransient.from_sncosmo(make_source("fakeA"), rate=500, rate_H0=60, magabs=[-15, 0.4])

    assert transient.template_source.name == "fakeA"
    assert transient.rate == 500
    assert transient._rateh0 == 60
    assert transient.get_model_parameter("magabs", "loc") == -15

def test_from_sncosmo_model():
    model = {"t0": {"func": np.random.uniform, "kwargs": {"low": 0, "high": 1}}}
    transient = TSTransient.from_sncosmo(make_source("fakeA"), model=model)
    assert transient.get_model_parameter("t0", "high") == 1

# tests for the func from_draw
def test_from_draw(tstransient):
    data = tstransient.data
    assert len(data) == 5
    assert (data["template"] == "fakeA").all()
    # default zmax=0.05
    assert (data["z"] > 0).all() and (data["z"] <= 0.05).all()
    assert (data["t0"] >= 56_000).all() and (data["t0"] <= 56_200).all()

def test_from_draw_independent_instances():
    # two instances must not share the random state (t0 and magabs scatter).
    data_1 = TSTransient.from_draw(size=5, template=make_source("fakeA"), magabs=[-18, 0.1]).data
    data_2 = TSTransient.from_draw(size=5, template=make_source("fakeA"), magabs=[-18, 0.1]).data
    assert not np.array_equal(data_1["t0"], data_2["t0"]), "same t0 drawn for different instances"
    assert not np.array_equal(data_1["magabs"], data_2["magabs"]), "same magabs drawn for different instances"

def test_draw_rng_reproducible_t0_magabs():
    transient = TSTransient(template=make_source("fakeA"), magabs=[-18, 0.1])
    data_1 = transient.draw(size=5, rng=1)
    data_2 = transient.draw(size=5, rng=1)
    assert np.array_equal(data_1["t0"], data_2["t0"])
    assert np.array_equal(data_1["magabs"], data_2["magabs"])

def test_from_draw_magobs(tstransient):
    data = tstransient.data
    # observed magnitude = absolute magnitude + distance modulus
    distmod = Planck18.distmod(data["z"].values).value
    assert np.allclose(data["magobs"] - data["magabs"], distmod, atol=1e-3)

def test_get_template_set_magabs(tstransient):
    model = tstransient.get_template(index=0, as_model=True, set_magabs=True)
    assert np.isclose(model.source_peakmag("bessellb", "ab"), tstransient.data.loc[0, "magobs"], atol=1e-4)


# tests for the class MultiTemplateTSTransient
def test_multi_set_template():
    transient = MultiTemplateTSTransient()
    transient.set_template([make_source("fakeA"), make_source("fakeB")])

    assert isinstance(transient.template, TemplateCollection)
    assert transient.template.ntemplates == 2
    assert transient.template.names == ["fakeA", "fakeB"]

def test_multi_set_template_multiple_types():
    transient = MultiTemplateTSTransient()
    other = OtherTimeSeriesSource(PHASES, WAVES, np.ones((len(PHASES), len(WAVES))), name="other")
    with pytest.raises(ValueError):
        transient.set_template([make_source("fakeA"), other])

def test_multi_set_template_multiple_types_allowed():
    transient = MultiTemplateTSTransient()
    other = OtherTimeSeriesSource(PHASES, WAVES, np.ones((len(PHASES), len(WAVES))), name="other")
    transient.set_template([make_source("fakeA"), other], force_uniquetype=False)
    assert transient.template.ntemplates == 2

def test_multi_model_has_template_entry():
    transient = MultiTemplateTSTransient()
    assert transient.model.model["template"]["func"] == "draw_template"
    assert transient.model.model["template"]["kwargs"] == {"redshift": "@z"}

# tests for the func set_rate (MultiTemplateTSTransient)
def test_multi_set_rate_single():
    transient = MultiTemplateTSTransient()
    transient.set_rate(1e3)
    assert transient.rate == 1e3
    assert transient._rateh0 == 70, "H0=None falls back to _RATE_H0"
    assert not transient.has_multirates

def test_multi_set_rate_callable():
    transient = MultiTemplateTSTransient()
    func = lambda z: 10 * np.ones_like(z)
    transient.set_rate(func, H0=68)
    assert transient.rate is func
    assert transient._rateh0 == 68

def test_multi_set_rate_list():
    transient = MultiTemplateTSTransient()
    transient.set_template([make_source("fakeA"), make_source("fakeB")])
    transient.set_rate([1e3, 2e3])
    # one rate per template, as a column
    assert transient.rate.shape == (2, 1)
    assert transient.has_multirates

def test_multi_set_rate_list_same_values():
    transient = MultiTemplateTSTransient()
    transient.set_template([make_source("fakeA"), make_source("fakeB")])
    transient.set_rate([1e3, 1e3])
    assert not transient.has_multirates, "identical rates are not 'multi' rates"

# tests for the func draw_template
def test_draw_template_uniform():
    transient = MultiTemplateTSTransient()
    transient.set_template([make_source("fakeA"), make_source("fakeB")])
    names = transient.draw_template(redshift=np.full(200, 0.1), rng=1)

    assert len(names) == 200
    assert set(names) == {"fakeA", "fakeB"}

def test_draw_template_rate_zero():
    transient = MultiTemplateTSTransient()
    transient.set_template([make_source("fakeA"), make_source("fakeB")])
    transient.set_rate([1e3, 0.])
    names = transient.draw_template(redshift=np.full(50, 0.1), rng=1)
    assert (names == "fakeA").all(), "a template with rate=0 is never drawn"

# tests for the func draw_redshift (MultiTemplateTSTransient)
def test_multi_draw_redshift():
    transient = MultiTemplateTSTransient()
    redshifts = transient.draw_redshift(zmax=0.1, zmin=0.05, size=30)
    assert len(redshifts) == 30
    assert (redshifts >= 0.05).all() and (redshifts <= 0.1).all()

# tests for the func from_draw (MultiTemplateTSTransient)
def test_multi_from_draw(multitransient):
    data = multitransient.data
    assert len(data) == 20
    assert set(data["template"]).issubset({"fakeA", "fakeB"})

# tests for the func get_template (MultiTemplateTSTransient)
def test_multi_get_template_matches_data(multitransient):
    data = multitransient.data
    for index in [0, 5, 10]:
        model = multitransient.get_template(index=index, as_model=True)
        assert model.source.name == data.loc[index, "template"], "the target's own template is used"
        assert np.isclose(model.get("z"), data.loc[index, "z"])
        assert np.isclose(model.get("t0"), data.loc[index, "t0"])

def test_multi_get_template_default_index(multitransient):
    template = multitransient.get_template()
    assert isinstance(template, Template)
    assert np.isclose(template.sncosmo_model.get("z"), multitransient.data.loc[0, "z"]), "index=None => index 0"

def test_multi_get_template_kwargs(multitransient):
    model = multitransient.get_template(index=0, as_model=True, z=0.3)
    assert model.get("z") == 0.3

def test_multi_get_template_set_magabs_magobs(multitransient):
    model = multitransient.get_template(index=1, as_model=True, set_magabs=True)
    assert np.isclose(model.source_peakmag("bessellb", "ab"), multitransient.data.loc[1, "magobs"], atol=1e-4)

def test_multi_get_template_set_magabs_magabs():
    transient = MultiTemplateTSTransient()
    transient.set_template([make_source("fakeA"), make_source("fakeB")])
    data = pandas.DataFrame({"z": [0.01, 0.02], "t0": [0., 1.], "magabs": [-18., -19.],
                             "template": ["fakeA", "fakeB"]})
    transient.set_data(data)
    model = transient.get_template(index=1, as_model=True, set_magabs=True)
    assert model.source.name == "fakeB"
    assert np.isclose(model.source_peakabsmag("bessellb", "ab", cosmo=Planck18), -19., atol=1e-4)

# tests for the func as_targets
def test_multi_as_targets(multitransient):
    targets = multitransient.as_targets()
    counts = multitransient.data["template"].value_counts()

    assert len(targets) == len(counts)
    for target in targets:
        assert isinstance(target, TSTransient)
        name = target.template_source.name
        assert len(target.data) == counts[name]
        assert (target.data["template"] == name).all()

def test_multi_as_targets_no_template_column():
    transient = MultiTemplateTSTransient()
    transient.set_template([make_source("fakeA"), make_source("fakeB")])
    transient.set_data(pandas.DataFrame({"z": [0.1]}), incl_template=False)
    with pytest.raises(AttributeError):
        transient.as_targets()
