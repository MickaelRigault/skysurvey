import warnings
import numpy as np
import pandas
import pytest
import sncosmo

from astropy import time
from astropy.cosmology import Planck18, FlatLambdaCDM
from shapely import geometry

from skysurvey.effects import Effect
from skysurvey.template import Template
from skysurvey.target.core import Target, Transient
from skysurvey.target.rates import get_ntargets
from skysurvey.tools.utils import random_radec


# A tiny hand-built sncosmo time-series source: flat spectrum, gaussian in phase
# peaking at phase 0 (sigma=10 days). No download needed.
PHASES = np.linspace(-20, 50, 15) # 5-day steps, includes phase 0
WAVES = np.linspace(2000, 10000, 50)

def make_source(name="fake"):
    flux = np.exp(-0.5 * (PHASES[:, None] / 10)**2) * np.ones((len(PHASES), len(WAVES))) * 1e-15
    return sncosmo.TimeSeriesSource(PHASES, WAVES, flux, name=name)


def constant(value, size=None):
    """ model func returning `size` times `value` """
    return np.full(size, value)

def constant_like(ref, value):
    """ model func returning `value` for each entry of `ref` (e.g. '@z') """
    return np.full(len(ref), value)

def rate_entry(rate, size=None):
    """ model func accepting a 'rate' argument (see Target._update_rate_in_model_) """
    return np.full(size, rate)


class FakeTransient( Transient ):
    _KIND = "fake"
    _RATE = 1e3
    _TEMPLATE = make_source()
    _MODEL = dict(redshift={"kwargs": {"zmax": 0.05}, "as": "z"},
                  t0={"func": np.random.uniform, "kwargs": {"low": 100, "high": 200}},
                  magabs={"func": np.random.normal, "kwargs": {"loc": -18, "scale": 0.1}},
                  magobs={"func": "magabs_to_magobs", "kwargs": {"z": "@z", "magabs": "@magabs"}},
                  radec={"func": random_radec, "as": ["ra", "dec"]},
                  )

class FakeTransientNoRadec( FakeTransient ):
    _MODEL = dict(redshift={"kwargs": {"zmax": 0.05}, "as": "z"},
                  t0={"func": np.random.uniform, "kwargs": {"low": 100, "high": 200}},
                  )


@pytest.fixture
def transient():
    return FakeTransient.from_draw(size=5)

@pytest.fixture
def simple_data():
    return pandas.DataFrame({"z": [0.01, 0.02, 0.03],
                             "t0": [100., 110., 120.],
                             "magabs": [-18., -18.5, -19.],
                             "foo": [1, 2, 3]})


# tests for the class Target (default attributes)
def test_target_defaults():
    target = FakeTransient()
    assert target.kind == "fake"
    assert target.magsys == "ab"
    assert target.peak_absmag_band == "bessellb"
    assert target.amplitude_name == "amplitude"
    assert target.cosmology is Planck18

def test_class_properties_not_inherited_from_parent_read():
    # reading the parent class first must not change what the subclass returns.
    class FakeSubTransient( FakeTransient ):
        _KIND = "fakesub"
        _MAGSYS = "vega"
        _PEAK_ABSMAG_BAND = "bessellv"
        _AMPLITUDE_NAME = "x0"

    parent = (FakeTransient.kind, FakeTransient.magsys,
              FakeTransient.peak_absmag_band, FakeTransient.amplitude_name)
    child = (FakeSubTransient.kind, FakeSubTransient.magsys,
             FakeSubTransient.peak_absmag_band, FakeSubTransient.amplitude_name)
    assert parent == ("fake", "ab", "bessellb", "amplitude")
    assert child == ("fakesub", "vega", "bessellv", "x0")
    assert FakeSubTransient().kind == "fakesub", "instance must return its class value"

def test_target_no_model_no_template():
    target = Target()
    assert target.data is None
    assert target.template is None, "no _TEMPLATE => no template"

def test_set_cosmology():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    target = FakeTransient()
    target.set_cosmology(cosmo)
    assert target.cosmology is cosmo

def test_model_is_a_copy_of_class_model():
    target = FakeTransient()
    target.update_model_parameter(t0={"low": 0})

    assert target.get_model_parameter("t0", "low") == 0
    assert FakeTransient._MODEL["t0"]["kwargs"]["low"] == 100, "the class _MODEL must not be changed"

# tests for the func from_setting
def test_from_setting_not_implemented():
    with pytest.raises(NotImplementedError):
        Target.from_setting({})

# tests for the func set_data
def test_set_data_adds_template_name(simple_data):
    target = FakeTransient()
    target.set_data(simple_data)
    assert (target.data["template"] == "fake").all(), "template column filled with the source name"

def test_set_data_unknown_template(simple_data):
    target = Target()
    target.set_data(simple_data)
    assert (target.data["template"] == "unknown").all()

def test_set_data_keeps_existing_template(simple_data):
    simple_data["template"] = "other"
    target = FakeTransient()
    target.set_data(simple_data)
    assert (target.data["template"] == "other").all()

def test_set_data_without_template(simple_data):
    target = FakeTransient()
    target.set_data(simple_data, incl_template=False)
    assert "template" not in target.data

# tests for the func from_data
def test_from_data(simple_data):
    target = FakeTransient.from_data(simple_data)

    assert len(target.data) == 3
    assert list(target.data["z"]) == [0.01, 0.02, 0.03]
    assert target.template_source.name == "fake"

def test_from_data_template_and_model(simple_data):
    target = FakeTransient.from_data(simple_data, template=make_source("other"),
                                     model={"t0": {"func": np.random.uniform, "kwargs": {"low": 0, "high": 1}}})

    assert target.template_source.name == "other"
    assert target.get_model_parameter("t0", "high") == 1

# tests for the func set_template
def test_set_template_from_source():
    target = FakeTransient()
    target.set_template(make_source("newsource"))

    assert isinstance(target.template, Template)
    assert target.template_source.name == "newsource"
    assert list(target.template_parameters) == ["z", "t0", "amplitude"]
    assert list(target.template_effect_parameters) == []

def test_set_template_rate_update_warns():
    target = FakeTransient()
    with pytest.warns(UserWarning):
        target.set_template(make_source(), rate_update=True)

# tests for the func set_rate / rate
def test_rate_default():
    target = FakeTransient()
    assert target.rate == 1e3
    assert target._rateh0 == 70

def test_set_rate_float():
    target = FakeTransient()
    target.set_rate(5, H0=68)
    assert target.rate == 5.0
    assert isinstance(target.rate, float)
    assert target._rateh0 == 68.0

def test_set_rate_h0_default():
    target = FakeTransient()
    target.set_rate(5)
    assert target._rateh0 == 70, "H0=None falls back to _RATE_H0"

def test_set_rate_callable():
    target = FakeTransient()
    func = lambda z: 10 * (1 + z)
    target.set_rate(func)
    assert target.rate is func

# tests for the func get_rate
def test_get_rate_same_h0():
    target = FakeTransient()
    target.set_cosmology(FlatLambdaCDM(H0=70, Om0=0.3))
    # constant volumetric rate seen in the observer frame: rate / (1+z)
    assert np.isclose(target.get_rate(0.1), 1e3 / 1.1)

def test_get_rate_h0_rescaling():
    target = FakeTransient()
    target.set_cosmology(FlatLambdaCDM(H0=35, Om0=0.3))
    # volumes scale as H0^-3 so rates per Gpc3 as H0^3: (35/70)**3 = 1/8
    assert np.isclose(target.get_rate(0.), 1e3 / 8)

def test_get_rate_callable_kwargs():
    target = FakeTransient()
    target.set_rate(lambda z, factor=1: factor * np.ones_like(z))
    target.set_cosmology(FlatLambdaCDM(H0=70, Om0=0.3))
    assert np.isclose(target.get_rate(1., factor=4), 4 / 2)

# tests for the func draw_redshift
def test_draw_redshift_range():
    target = FakeTransient()
    redshifts = target.draw_redshift(zmax=0.2, zmin=0.1, size=100)

    assert len(redshifts) == 100
    assert (redshifts >= 0.1).all() and (redshifts <= 0.2).all()

# tests for the func magabs_to_magobs
def test_magabs_to_magobs():
    target = FakeTransient()
    z = 0.1
    # distance modulus: 5 log10(d_L / 10pc)
    dl_pc = Planck18.luminosity_distance(z).to("pc").value
    expected = 5 * np.log10(dl_pc / 10) - 19.
    assert np.isclose(target.magabs_to_magobs(z, -19.), expected, rtol=1e-6)

def test_magabs_to_magobs_cosmology():
    target = FakeTransient()
    cosmo = FlatLambdaCDM(H0=35, Om0=0.3)
    # halving H0 doubles distances => +5 log10(2) mag
    diff = target.magabs_to_magobs(0.1, -19., cosmology=cosmo) - target.magabs_to_magobs(0.1, -19., cosmology=FlatLambdaCDM(H0=70, Om0=0.3))
    assert np.isclose(diff, 5 * np.log10(2), rtol=1e-5)

def test_magabs_to_magobs_array():
    target = FakeTransient()
    magobs = target.magabs_to_magobs([0.05, 0.1], [-19., -19.])
    assert magobs.shape == (2,)
    assert magobs[1] > magobs[0], "further is fainter"

# tests for the func get_model / get_model_parameter / update_model_parameter / update_model
def test_get_model_parameter():
    target = FakeTransient()
    assert target.get_model_parameter("t0", "low") == 100
    assert target.get_model_parameter("t0", "unknown", default=-1) == -1

def test_get_model():
    target = FakeTransient()
    model = target.get_model()
    assert isinstance(model, dict)
    assert set(model.keys()) == {"redshift", "t0", "magabs", "magobs", "radec"}

def test_update_model_parameter():
    target = FakeTransient()
    target.update_model_parameter(t0={"high": 300})

    assert target.get_model_parameter("t0", "high") == 300
    assert target.get_model_parameter("t0", "low") == 100, "other kwargs are kept"

def test_update_model_new_entry():
    target = FakeTransient()
    target.update_model(foo={"func": np.random.uniform, "kwargs": {"low": 0, "high": 1}})

    assert "foo" in target.model.model
    assert "t0" in target.model.model, "existing entries kept"

def test_update_model_feeds_rate():
    target = FakeTransient()
    target.set_rate(42)
    target.update_model(withrate={"func": rate_entry})
    assert target.get_model_parameter("withrate", "rate") == 42, "rate fed into entries accepting 'rate'"

def test_update_model_warns_multiple_rate_entries():
    target = FakeTransient()
    with pytest.warns(UserWarning, match="more than 1"):
        target.update_model(rate_a={"func": rate_entry}, rate_b={"func": rate_entry})

# tests for the func from_draw / draw
def test_from_draw_size(transient):
    data = transient.data
    assert len(data) == 5
    for key in ["z", "t0", "magabs", "magobs", "ra", "dec", "template"]:
        assert key in data, f"{key} missing"

    assert (data["z"] > 0).all() and (data["z"] <= 0.05).all()
    assert (data["t0"] >= 100).all() and (data["t0"] <= 200).all()
    assert (data["template"] == "fake").all()

def test_from_draw_lower_precision(transient):
    # inplace draws are stored as float32 to save memory
    assert transient.data["z"].dtype == np.float32
    assert transient.data["t0"].dtype == np.float32

def test_from_draw_magobs(transient):
    data = transient.data
    expected = transient.magabs_to_magobs(data["z"], data["magabs"])
    assert np.allclose(data["magobs"], expected, atol=1e-4)

def test_from_draw_zmax():
    target = FakeTransient.from_draw(size=20, zmax=0.01)
    assert (target.data["z"] <= 0.01).all()
    assert target.get_model_parameter("redshift", "zmax") == 0.01, "inplace stores the draw kwargs"

def test_from_draw_options():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    target = FakeTransient.from_draw(size=3, cosmology=cosmo, rate=10, rate_H0=60,
                                     template=make_source("other"),
                                     model={"magabs": {"func": np.random.normal, "kwargs": {"loc": -10, "scale": 0.1}}},
                                     t0={"low": 0, "high": 1})
    assert target.cosmology is cosmo
    assert target.rate == 10 and target._rateh0 == 60
    assert (target.data["template"] == "other").all()
    assert (target.data["magabs"] > -11).all()
    assert (target.data["t0"] <= 1).all()

def test_draw_tstart_tstop_str():
    target = FakeTransient()
    data = target.draw(size=20, tstart="2020-01-01", tstop="2020-02-01")
    # 2020-01-01 is MJD 58849, 2020-02-01 is MJD 58880
    assert (data["t0"] >= 58849).all() and (data["t0"] <= 58880).all()

def test_draw_tstart_tstop_time():
    target = FakeTransient()
    data = target.draw(size=20, tstart=time.Time("2020-01-01"), tstop=time.Time("2020-02-01"))
    assert (data["t0"] >= 58849).all() and (data["t0"] <= 58880).all()

def test_draw_tstart_only_one_year():
    target = FakeTransient()
    data = target.draw(size=50, tstart=1000)
    # default: one year after tstart
    assert (data["t0"] >= 1000).all() and (data["t0"] <= 1000 + 365.25).all()

def test_draw_no_size_no_nyears():
    target = FakeTransient()
    with pytest.raises(ValueError):
        target.draw()

def test_draw_not_inplace():
    target = FakeTransient()
    data = target.draw(size=4)
    assert len(data) == 4
    assert target.data is None, "inplace=False does not attach data"

def test_draw_model_does_not_change_current_model():
    target = FakeTransient()
    data = target.draw(size=10, model={"t0": {"func": np.random.uniform, "kwargs": {"low": -2, "high": -1}}})
    assert (data["t0"] < 0).all()
    assert target.get_model_parameter("t0", "low") == 100

def test_draw_nyears_size():
    target = FakeTransient()
    target.set_cosmology(Planck18)
    nyears = 2
    data = target.draw(nyears=nyears, tstart=0)
    # number of targets expected up to zmax=0.05 full sky, times the number of years
    expected = int(get_ntargets(0.05, rate=1e3, zstep=1e-4, astype="float", cosmology=Planck18) * nyears)
    assert len(data) == expected
    assert (data["t0"] >= 0).all() and (data["t0"] <= nyears * 365.25).all()

def test_draw_nyears_tstop():
    target = FakeTransient()
    data = target.draw(nyears=1, tstop=1000)
    # tstart = tstop - nyears years
    assert (data["t0"] >= 1000 - 365.25).all() and (data["t0"] <= 1000).all()

def test_draw_nyears_skyarea_fraction():
    target = FakeTransient()
    # a 'ra in [0,360], dec in [0,90]' cap is half of the sky
    northern = geometry.box(0, 0, 360, 90)
    full = target.draw(nyears=10, tstart=0)
    half = target.draw(nyears=10, tstart=0, skyarea=northern)

    assert np.abs(len(half) - len(full) / 2) <= 1, "half the sky => half the targets"
    assert (half["dec"] >= 0).all()

def test_draw_skyarea_without_radec_warns():
    target = FakeTransientNoRadec()
    with pytest.warns(UserWarning, match="no model have skyarea"):
        target.draw(size=3, skyarea=geometry.box(0, 0, 10, 10))

def test_draw_set_amplitude():
    target = FakeTransient()
    data = target.draw(size=5, set_amplitude=True)
    # same template, so the amplitude scales as the flux: 10**(-0.4 * magobs)
    norm = data["amplitude"] * 10**(0.4 * data["magobs"])
    assert np.allclose(norm, norm[0], rtol=1e-4)

# tests for the func get_template_columns / get_template_parameters
def test_get_template_columns(transient):
    assert list(transient.get_template_columns()) == ["z", "t0"]

def test_get_template_parameters(transient):
    params = transient.get_template_parameters()
    assert list(params.columns) == ["z", "t0"]
    assert len(params) == 5

def test_get_template_parameters_index(transient):
    params = transient.get_template_parameters(2)
    assert isinstance(params, pandas.Series)
    assert params["z"] == transient.data.loc[2, "z"]

# tests for the func get_template
def test_get_template_default():
    target = FakeTransient()
    template = target.get_template()
    assert isinstance(template, Template)
    assert template.sncosmo_model.get("z") == 0
    assert template.sncosmo_model is not target.template.sncosmo_model, "a copy is returned"

def test_get_template_as_model():
    target = FakeTransient()
    model = target.get_template(as_model=True, z=0.2)
    assert isinstance(model, sncosmo.Model)
    assert model.get("z") == 0.2

def test_get_template_index(transient):
    model = transient.get_template(index=1, as_model=True)
    assert np.isclose(model.get("z"), transient.data.loc[1, "z"])
    assert np.isclose(model.get("t0"), transient.data.loc[1, "t0"])

def test_get_template_index_kwargs_override(transient):
    model = transient.get_template(index=1, as_model=True, z=0.3)
    assert model.get("z") == 0.3, "kwargs take precedence over the data"

def test_get_template_set_magabs_magobs(transient):
    model = transient.get_template(index=0, as_model=True, set_magabs=True)
    # the source peak magnitude is set to magobs in the peak_absmag_band
    assert np.isclose(model.source_peakmag("bessellb", "ab"), transient.data.loc[0, "magobs"], atol=1e-4)

def test_get_template_set_magabs_magabs(simple_data):
    target = FakeTransient.from_data(simple_data)
    model = target.get_template(index=2, as_model=True, set_magabs=True)
    # no magobs: peak absolute magnitude is set to magabs
    assert np.isclose(model.source_peakabsmag("bessellb", "ab", cosmo=Planck18), -19., atol=1e-4)

def test_get_target_template(transient):
    template = transient.get_target_template(3)
    assert isinstance(template, Template)
    assert np.isclose(template.sncosmo_model.get("t0"), transient.data.loc[3, "t0"])

# tests for the func get_target_flux / get_target_mag / get_target_peakmag
def test_get_target_flux_restframe(transient):
    z = transient.data.loc[0, "z"]
    # a restframe phase p is an observer-frame phase p*(1+z)
    restflux = transient.get_target_flux(0, "bessellb", 5., restframe=True)
    obsflux = transient.get_target_flux(0, "bessellb", 5. * (1 + z), restframe=False)
    assert np.isclose(restflux, obsflux)

def test_get_target_flux_peak(transient):
    # the source peaks at phase 0
    assert transient.get_target_flux(0, "bessellb", 0.) > transient.get_target_flux(0, "bessellb", 10.)

def test_get_target_peakmag(transient):
    # the fake source peaks at phase 0
    peakmag = transient.get_target_peakmag(0, "bessellb")
    assert np.isclose(peakmag, transient.get_target_mag(0, "bessellb", 0.), atol=1e-3)

def test_get_target_mag_fainter_later(transient):
    assert transient.get_target_mag(0, "bessellb", 10.) > transient.get_target_mag(0, "bessellb", 0.)

# tests for the func get_lightcurve
def test_get_lightcurve_mag(transient):
    t0 = transient.data.loc[0, "t0"]
    mag = transient.get_lightcurve(["bessellb"], [t0], index=0, in_mag=True)
    assert np.isclose(mag, transient.get_target_mag(0, "bessellb", 0.), atol=1e-4)

def test_get_lightcurve_flux_zp(transient):
    t0 = transient.data.loc[0, "t0"]
    mag = transient.get_lightcurve(["bessellb"], [t0], index=0, in_mag=True)
    flux = transient.get_lightcurve(["bessellb"], [t0], index=0, zp=25, zpsys="ab")
    # flux in a zp=25 system: mag = 25 - 2.5 log10(flux)
    assert np.isclose(flux, 10**(-0.4 * (mag - 25)), rtol=1e-4)

def test_get_lightcurve_shape(transient):
    flux = transient.get_lightcurve(["bessellb", "bessellv"], [100, 110, 120], index=0)
    assert flux.shape == (2, 3)

def test_get_lightcurve_given_model(transient):
    model = transient.get_template(index=0, as_model=True, set_magabs=True)
    t0 = transient.data.loc[0, "t0"]
    flux = transient.get_lightcurve(["bessellb"], [t0], sncosmo_model=model)
    assert np.isclose(flux, model.bandflux("bessellb", t0, zp=25, zpsys="ab"))

# tests for the func get_spectrum
def test_get_spectrum_outside_range(transient):
    # the source is defined in [2000, 10000]AA rest-frame, so 50_000AA is out.
    flux = transient.get_spectrum(0., [5000., 50_000.], index=0)
    assert flux[0] > 0
    assert flux[1] == 0

def test_get_spectrum_flat(transient):
    # flat spectrum
    flux = transient.get_spectrum(0., [4000., 6000.], index=0)
    assert np.isclose(flux[0], flux[1])

# tests for the func clone_target_change_entry
def test_clone_target_change_entry_dataframe(transient):
    data = transient.clone_target_change_entry(0, "magabs", [-17., -19.], as_dataframe=True)

    assert len(data) == 2, "one row per new value"
    assert list(data["magabs"]) == [-17., -19.]
    assert (data["z"] == transient.data.loc[0, "z"]).all(), "upstream entries are unchanged"

def test_clone_target_change_entry_instance(transient):
    clone = transient.clone_target_change_entry(0, "magabs", [-17., -19.])
    assert isinstance(clone, FakeTransient)
    assert len(clone.data) == 2

# tests for the func apply_gaussian_noise
def test_apply_gaussian_noise_dataframe(transient):
    errmodel = {"magobs": {"func": constant, "kwargs": {"value": 0.1}}}
    new_data = transient.apply_gaussian_noise(errmodel, data=transient.data)

    assert isinstance(new_data, pandas.DataFrame)
    assert np.allclose(new_data["magobs_err"], 0.1)
    assert np.allclose(new_data["magobs_true"], transient.data["magobs"]), "original stored as _true"
    assert not np.allclose(new_data["magobs"], new_data["magobs_true"]), "noise applied"

def test_apply_gaussian_noise_instance(transient):
    errmodel = {"magobs": {"func": constant, "kwargs": {"value": 0.1}}}
    noisy = transient.apply_gaussian_noise(errmodel)
    assert isinstance(noisy, FakeTransient)
    assert "magobs_true" in noisy.data
    assert "magobs_true" not in transient.data, "original instance unchanged"

# tests for the func add_effect
def dust_effect():
    return {"effect": sncosmo.CCM89Dust(), "name": "mw", "frame": "obs",
            "model": {"mwebv": {"func": constant_like, "kwargs": {"ref": "@z", "value": 0.1}}}}

def test_add_effect_no_data():
    target = FakeTransient()
    target.add_effect(dust_effect())

    assert "mwebv" in target.template_parameters
    assert "mw" in target.template_effect_parameters, "effect names"
    assert "mwebv" in target.model.model

def test_add_effect_redraws_data(transient):
    transient.add_effect(Effect(**dust_effect()))
    assert np.allclose(transient.data["mwebv"], 0.1)

def test_add_effect_no_overwrite(transient):
    transient.add_effect(dust_effect())
    effect = dust_effect()
    effect["model"]["mwebv"]["kwargs"]["value"] = 0.5
    with pytest.warns(UserWarning, match="overwrite"):
        transient.add_effect(effect)
    assert np.allclose(transient.data["mwebv"], 0.1), "data unchanged"

def test_add_effect_model_override(transient):
    model = {"mwebv": {"func": constant_like, "kwargs": {"ref": "@z", "value": 0.3}}}
    transient.add_effect(dust_effect(), model=model)
    assert np.allclose(transient.data["mwebv"], 0.3)

def test_add_effect_data_merge(transient):
    extra = pandas.DataFrame({"t0": transient.data["t0"], "mwebv": np.arange(5) * 0.1})
    transient.add_effect(dust_effect(), data=extra, on="t0")
    assert np.allclose(transient.data["mwebv"], np.arange(5) * 0.1)

def test_add_effect_data_without_current_data():
    target = FakeTransient()
    with pytest.warns(UserWarning, match="no current data"):
        target.add_effect(dust_effect(), data=pandas.DataFrame({"mwebv": [0.1]}))
    assert target.data is None

def test_from_draw_effect():
    target = FakeTransient.from_draw(size=3, effect=dust_effect())
    assert np.allclose(target.data["mwebv"], 0.1)
    assert "mwebv" in target.template_parameters
