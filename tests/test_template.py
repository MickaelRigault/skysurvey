import numpy as np
import pandas
import pytest
import sncosmo

from skysurvey.template import (Template, TemplateCollection,
                                get_sncosmo_model, parse_template,
                                sncosmoresult_to_pandas)
from skysurvey.effects.core import Effect


# small, hand-built and local sources (no download needed)
PHASES = np.linspace(-10, 40, 51)
WAVES = np.linspace(3000, 9000, 61)
FLAT_FLUX = 1e-15 # erg/s/cm2/A
FLAT_NAME = "skysurvey_test_flat"


def get_flat_source(name=FLAT_NAME):
    """ source with a flux constant in time and wavelength """
    return sncosmo.TimeSeriesSource(PHASES, WAVES, np.full((len(PHASES), len(WAVES)), FLAT_FLUX),
                                    name=name)

def get_gaussian_source(name="skysurvey_test_gauss"):
    """ source peaking at phase 0 with a 10 days gaussian width (flat in wavelength) """
    timeprofile = np.exp(-0.5 * (PHASES / 10)**2)
    return sncosmo.TimeSeriesSource(PHASES, WAVES, FLAT_FLUX * timeprofile[:, None] * np.ones(len(WAVES)),
                                    name=name)

def get_stretch_source(name="skysurvey_test_stretch"):
    """ a source of another sncosmo class (to test non-unique collections) """
    return sncosmo.StretchSource(PHASES, WAVES, np.full((len(PHASES), len(WAVES)), FLAT_FLUX),
                                 name=name)


# register the flat source so that it can be accessed by name
sncosmo.register(get_flat_source(), FLAT_NAME, force=True)


@pytest.fixture
def flat_template():
    return Template.from_sncosmo(get_flat_source())

@pytest.fixture
def gaussian_template():
    return Template.from_sncosmo(get_gaussian_source())


# tests for the func get_sncosmo_model
def test_get_sncosmo_model_from_source():
    source = get_flat_source()
    model = get_sncosmo_model(source, z=0.1, t0=10, amplitude=2)

    assert isinstance(model, sncosmo.Model)
    assert model.source.name == FLAT_NAME # (sncosmo.Model copies the source)
    assert model.get("z") == 0.1
    assert model.get("t0") == 10
    assert model.get("amplitude") == 2

def test_get_sncosmo_model_from_name():
    model = get_sncosmo_model(FLAT_NAME)
    assert model.source.name == FLAT_NAME

def test_get_sncosmo_model_zero_before():
    # zero_before=True => no flux before the first phase of the source.
    model = get_sncosmo_model(get_flat_source(), zero_before=True)
    assert model.source._zero_before
    assert model.flux(-20, [5000.])[0] == 0, "flux before the source minimum phase should be zero"

    model = get_sncosmo_model(get_flat_source(), zero_before=False)
    assert not model.source._zero_before
    assert model.flux(-20, [5000.])[0] > 0, "sncosmo extrapolates the first phase flux by default"


# tests for the func sncosmoresult_to_pandas
def test_sncosmoresult_to_pandas():
    # sncosmo.Result is a dict with attribute access, a dict is enough here.
    result = {"param_names": ["z", "t0", "amplitude"],
              "parameters": np.array([0.1, 10., 2.]),
              "errors": {"t0": 0.5, "amplitude": 0.1},
              "covariance": np.array([[0.25, 0.01], [0.01, 0.01]]),
              "success": True, "ncall": 42, "chisq": 12., "ndof": 4}

    fit_res, fit_meta = sncosmoresult_to_pandas(result)

    assert list(fit_res.index) == ["z", "t0", "amplitude"]
    assert list(fit_res.columns) == ["value", "error", "cov_t0", "cov_amplitude"]
    np.testing.assert_allclose(fit_res["value"], [0.1, 10., 2.])
    assert np.isnan(fit_res.loc["z", "error"]), "z is not fitted, so it has no error"
    assert fit_res.loc["t0", "error"] == 0.5
    assert fit_res.loc["t0", "cov_amplitude"] == 0.01
    assert fit_res.loc["amplitude", "cov_amplitude"] == 0.01

    assert fit_meta["success"]
    assert fit_meta["ncall"] == 42
    assert fit_meta["chi2dof"] == 3. # 12 / 4


# tests for the func parse_template
def test_parse_template_name():
    template = parse_template(FLAT_NAME)
    assert isinstance(template, Template)
    assert template.source.name == FLAT_NAME

def test_parse_template_source():
    source = get_flat_source()
    template = parse_template(source)
    assert isinstance(template, Template)
    assert template.source.name == FLAT_NAME # (sncosmo.Model copies the source)

def test_parse_template_model():
    model = sncosmo.Model(get_flat_source())
    template = parse_template(model)
    assert isinstance(template, Template)
    assert template.sncosmo_model is model

def test_parse_template_template(flat_template):
    # a Template is returned as is
    assert parse_template(flat_template) is flat_template

def test_parse_template_list(flat_template):
    templates = parse_template([flat_template, get_gaussian_source()])
    assert len(templates) == 2
    assert templates[0] is flat_template
    assert isinstance(templates[1], Template)
    assert templates[1].source.name == "skysurvey_test_gauss"


# tests for the class Template
def test_from_sncosmo_source():
    source = get_flat_source()
    template = Template.from_sncosmo(source)
    assert isinstance(template.sncosmo_model, sncosmo.Model)
    assert template.source.name == FLAT_NAME # (sncosmo.Model copies the source)
    assert template.source._zero_before, "zero_before is True by default"

def test_from_sncosmo_kwargs():
    template = Template.from_sncosmo(get_flat_source(), zero_before=False, amplitude=3)
    assert not template.source._zero_before
    assert template.sncosmo_model.get("amplitude") == 3

def test_from_sncosmo_model():
    model = sncosmo.Model(get_flat_source())
    template = Template.from_sncosmo(model)
    assert template.sncosmo_model is model

def test_parameters(flat_template):
    assert flat_template.parameters == ["z", "t0", "amplitude"]
    assert flat_template.core_parameters == ["amplitude"]
    assert flat_template.effect_parameters == []

# add_effect
def test_add_effect(flat_template):
    effect = Effect.from_sncosmo(sncosmo.CCM89Dust(), name="mw", frame="obs")
    flat_template.add_effect(effect)

    assert flat_template.effect_parameters == ["mw"]
    assert flat_template.parameters == ["z", "t0", "amplitude", "mwebv", "mwr_v"]

def test_add_effect_list(flat_template):
    effects = [Effect.from_sncosmo(sncosmo.CCM89Dust(), name="mw", frame="obs"),
               Effect.from_sncosmo(sncosmo.CCM89Dust(), name="host", frame="rest")]
    flat_template.add_effect(effects)

    assert flat_template.effect_parameters == ["mw", "host"]

def test_add_effect_changes_flux(flat_template):
    effect = Effect.from_sncosmo(sncosmo.CCM89Dust(), name="mw", frame="obs")
    flat_template.add_effect(effect)

    flux_nodust = flat_template.get(mwebv=0).flux(0, [5000.])[0]
    flux_dust = flat_template.get(mwebv=0.1).flux(0, [5000.])[0]
    assert flux_nodust == pytest.approx(FLAT_FLUX), "no dust => source flux"
    assert flux_dust < flux_nodust, "dust extinction should lower the flux"

# get
def test_get_is_a_copy(flat_template):
    model = flat_template.get()
    assert model is not flat_template.sncosmo_model
    assert list(model.parameters) == list(flat_template.sncosmo_model.parameters)

    # changing the copy does not change the template.
    model.set(amplitude=10)
    assert flat_template.sncosmo_model.get("amplitude") == 1

def test_get_kwargs(flat_template):
    model = flat_template.get(z=0.1, t0=100, amplitude=3)
    assert model.get("z") == 0.1
    assert model.get("t0") == 100
    assert model.get("amplitude") == 3
    # template unchanged
    assert flat_template.sncosmo_model.get("t0") == 0

# get_lightcurve
def test_get_lightcurve_flux(flat_template):
    times = np.array([0., 5., 10.])
    params = dict(z=0.1, t0=0, amplitude=2)
    lc = flat_template.get_lightcurve(["bessellb", "bessellv"], times, zp=25, zpsys="ab", **params)

    assert lc.shape == (2, 3)
    model = sncosmo.Model(get_flat_source())
    model.set(**params)
    np.testing.assert_allclose(lc[0], model.bandflux("bessellb", times, zp=25, zpsys="ab"))
    np.testing.assert_allclose(lc[1], model.bandflux("bessellv", times, zp=25, zpsys="ab"))

def test_get_lightcurve_mag(flat_template):
    times = np.array([0., 5., 10.])
    params = dict(z=0.1, t0=0, amplitude=2)
    flux = flat_template.get_lightcurve(["bessellb", "bessellv"], times, zp=25, zpsys="ab", **params)
    mag = flat_template.get_lightcurve(["bessellb", "bessellv"], times, zpsys="ab", in_mag=True, **params)

    # mag = -2.5 log10(flux) + zp
    np.testing.assert_allclose(mag, -2.5 * np.log10(flux) + 25)

def test_get_lightcurve_sncosmo_model(flat_template):
    # if a sncosmo_model is given, it is used (kwargs are ignored)
    model = flat_template.get(amplitude=4)
    times = np.array([0., 5.])
    lc = flat_template.get_lightcurve(["bessellb"], times, sncosmo_model=model, amplitude=1)
    np.testing.assert_allclose(lc[0], model.bandflux("bessellb", times, zp=25, zpsys="ab"))

def test_get_lightcurve_amplitude_scaling(flat_template):
    times = np.array([0., 5.])
    lc_1 = flat_template.get_lightcurve(["bessellb"], times, amplitude=1)
    lc_3 = flat_template.get_lightcurve(["bessellb"], times, amplitude=3)
    np.testing.assert_allclose(lc_3, 3 * lc_1)

# get_spectrum
def test_get_spectrum_as_phase(flat_template):
    lbdas = np.array([4000., 5000.])
    flux = flat_template.get_spectrum(5., lbdas, z=0.1, t0=100, amplitude=2)
    model = flat_template.get(z=0.1, t0=100, amplitude=2)
    # phase 5 => time = 5*(1+z) + t0
    np.testing.assert_allclose(flux, model.flux(5 * 1.1 + 100, lbdas))

def test_get_spectrum_as_time(flat_template):
    lbdas = np.array([4000., 5000.])
    flux = flat_template.get_spectrum(105., lbdas, as_phase=False, z=0.1, t0=100)
    model = flat_template.get(z=0.1, t0=100)
    np.testing.assert_allclose(flux, model.flux(105., lbdas))

def test_get_spectrum_sncosmo_model(flat_template):
    # if a sncosmo_model is given, it is used (kwargs are ignored)
    model = flat_template.get(amplitude=4)
    flux = flat_template.get_spectrum(0., np.array([5000.]), sncosmo_model=model, amplitude=1)
    assert flux[0] == pytest.approx(4 * FLAT_FLUX)

def test_get_spectrum_flat_value(flat_template):
    # at z=0 and amplitude=1, the flat source flux is FLAT_FLUX
    flux = flat_template.get_spectrum(0., np.array([4000., 5000.]))
    np.testing.assert_allclose(flux, FLAT_FLUX)

def test_get_spectrum_outside_wavelength(flat_template):
    # model defined on ]3000, 9000[ (at z=0) => 0 outside.
    flux = flat_template.get_spectrum(0., np.array([2000., 5000., 10000.]))
    assert flux[0] == 0
    assert flux[1] == pytest.approx(FLAT_FLUX)
    assert flux[2] == 0

def test_get_spectrum_no_wavelength_warns(flat_template):
    with pytest.warns(UserWarning):
        flux = flat_template.get_spectrum(0., np.array([1000., 2000.]))
    assert (flux == 0).all()

# fit_data
def test_fit_data(gaussian_template):
    truth = dict(z=0.1, t0=100., amplitude=2.)
    times = np.linspace(80, 140, 16)
    bands = np.resize(["bessellb", "bessellv"], len(times))
    model = gaussian_template.get(**truth)
    flux = model.bandflux(bands, times, zp=25, zpsys="ab")
    # noise-free data with a 1% error
    data = pandas.DataFrame({"time": times, "band": bands, "flux": flux,
                             "fluxerr": 0.01 * flux.max(), "zp": 25., "zpsys": "ab"})

    fit_res, fit_meta = gaussian_template.fit_data(data,
                                                   guessparams={"t0": 98., "amplitude": 1.5},
                                                   fixedparams={"z": 0.1})

    assert fit_res.loc["z", "value"] == 0.1, "z is fixed"
    assert fit_res.loc["t0", "value"] == pytest.approx(100., abs=1e-3)
    assert fit_res.loc["amplitude", "value"] == pytest.approx(2., rel=1e-4)
    assert fit_meta["success"]
    assert fit_meta["ndof"] == len(data) - 2 # t0, amplitude fitted
    assert fit_meta["chisq"] < 1e-4, "noise-free data should be perfectly fitted"


# tests for the class TemplateCollection
@pytest.fixture
def collection():
    return TemplateCollection.from_sncosmo([get_flat_source(), get_gaussian_source()])

def test_collection_from_sncosmo(collection):
    assert collection.ntemplates == 2
    assert all(isinstance(t, Template) for t in collection.templates)
    assert collection.names == [FLAT_NAME, "skysurvey_test_gauss"]

def test_collection_from_list(flat_template):
    collection = TemplateCollection.from_list([flat_template, get_gaussian_source()])
    assert collection.ntemplates == 2
    assert collection.templates[0] is flat_template

def test_collection_is_uniquetype(collection):
    assert collection.is_uniquetype # both TimeSeriesSource

    mixed = TemplateCollection.from_sncosmo([get_flat_source(), get_stretch_source()])
    assert not mixed.is_uniquetype

def test_collection_parameters(collection):
    # unique type => parameters of the first template
    assert collection.parameters == ["z", "t0", "amplitude"]
    assert collection.template_parameters == ["z", "t0", "amplitude"]

def test_collection_parameters_mixed():
    mixed = TemplateCollection.from_sncosmo([get_flat_source(), get_stretch_source()])
    # non-unique type => one list per template
    assert mixed.parameters == [["z", "t0", "amplitude"], ["z", "t0", "amplitude", "s"]]

def test_collection_call_down(collection):
    assert collection.call_down("parameters") == [["z", "t0", "amplitude"]] * 2
    # callable are called with kwargs
    models = collection.call_down("get", amplitude=3)
    assert [m.get("amplitude") for m in models] == [3, 3]
    # or returned as is if not allow_call
    methods = collection.call_down("get", allow_call=False)
    assert all(callable(m) for m in methods)

def test_collection_call_down_source(collection):
    assert collection.call_down_source("name") == [FLAT_NAME, "skysurvey_test_gauss"]
    assert collection.call_down_source("minphase") == [-10, -10]

def test_collection_add_effect(collection):
    effect = Effect.from_sncosmo(sncosmo.CCM89Dust(), name="mw", frame="obs")
    collection.add_effect(effect)
    assert collection.call_down("effect_parameters") == [["mw"], ["mw"]]

def test_collection_nameorindex_to_index(collection):
    assert collection.nameorindex_to_index("skysurvey_test_gauss") == 1
    assert collection.nameorindex_to_index(1) == 1

def test_collection_get(collection):
    model = collection.get(ref_index=1, amplitude=3)
    assert model.source.name == "skysurvey_test_gauss"
    assert model.get("amplitude") == 3

def test_collection_get_mixed_raises():
    mixed = TemplateCollection.from_sncosmo([get_flat_source(), get_stretch_source()])
    with pytest.raises(NotImplementedError):
        mixed.get()

def test_collection_get_lightcurve_raises(collection):
    with pytest.raises(ValueError):
        collection.get_lightcurve(["bessellb"], [0., 1.])
