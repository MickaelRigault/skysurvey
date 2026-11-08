import numpy as np
import pandas
import pytest
import sncosmo

from skysurvey.lightcurves import get_obsdata, realize_lightcurves, _get_obsdata_


# small, hand-built and local source (no download needed)
PHASES = np.linspace(-10, 40, 51)
WAVES = np.linspace(3000, 9000, 61)


@pytest.fixture
def model():
    """ source with a flux constant in time and wavelength """
    source = sncosmo.TimeSeriesSource(PHASES, WAVES, np.full((len(PHASES), len(WAVES)), 1e-15))
    return sncosmo.Model(source)

@pytest.fixture
def observations():
    return pandas.DataFrame({"band": ["bessellb", "bessellv", "bessellb", "bessellv"],
                             "mjd": [100., 101., 120., 160.],
                             "zp": [25., 25., 27., 27.],
                             "zpsys": "ab",
                             "gain": [1., 2., 1., 2.],
                             "skynoise": [10., 20., 30., 40.]})

@pytest.fixture
def parameters():
    return pandas.DataFrame({"z": [0.1, 0.2], "t0": [100., 110.], "amplitude": [1., 2.]},
                            index=pandas.Index([3, 7], name="index"))


def expected_flux(model, observations, param):
    """ noise-free flux, directly from sncosmo """
    model = sncosmo.Model(model.source)
    model.set(**param)
    return model.bandflux(observations["band"], observations["mjd"],
                          zp=observations["zp"], zpsys=observations["zpsys"])


# tests for the func realize_lightcurves
def test_realize_lightcurves_noscatter(model, observations, parameters):
    lcs = realize_lightcurves(observations, model, parameters, scatter=False)

    # one lightcurve per target, all observations.
    assert lcs.index.names == ["index", None]
    assert list(lcs.index.get_level_values(0).unique()) == [3, 7]
    assert len(lcs) == 2 * len(observations)

    for target_index, param in parameters.iterrows():
        lc = lcs.xs(target_index)
        flux = expected_flux(model, observations, param.to_dict())
        np.testing.assert_allclose(lc["flux"], flux)
        # fluxerr: skynoise and poisson noise in quadrature
        np.testing.assert_allclose(lc["fluxerr"],
                                   np.sqrt(observations["skynoise"]**2 + np.abs(flux) / observations["gain"]))
        # observation columns kept
        assert list(lc["band"]) == list(observations["band"])
        assert list(lc["mjd"]) == list(observations["mjd"])

def test_realize_lightcurves_input_model_unchanged(model, observations, parameters):
    _ = realize_lightcurves(observations, model, parameters, scatter=False)
    assert model.get("z") == 0, "the input model should not be modified"
    assert model.get("t0") == 0

def test_realize_lightcurves_scatter(model, observations, parameters):
    lcs = realize_lightcurves(observations, model, parameters.iloc[:1], scatter=True, rng=1234)
    flux = expected_flux(model, observations, parameters.iloc[0].to_dict())
    fluxerr = np.sqrt(observations["skynoise"]**2 + np.abs(flux) / observations["gain"])

    # scattered flux = normal draw centered on the flux with fluxerr sigma.
    expected = np.random.default_rng(1234).normal(flux, fluxerr)
    np.testing.assert_allclose(lcs["flux"], expected)
    np.testing.assert_allclose(lcs["fluxerr"], fluxerr)

def test_realize_lightcurves_scatter_reproducible(model, observations, parameters):
    lcs_1 = realize_lightcurves(observations, model, parameters, rng=42)
    lcs_2 = realize_lightcurves(observations, model, parameters, rng=42)
    pandas.testing.assert_frame_equal(lcs_1, lcs_2)

def test_realize_lightcurves_trim_observations(model, observations, parameters):
    # target 0: z=0.1, t0=100 => defined in [100-10*1.1, 100+40*1.1] = [89, 144]
    lcs = realize_lightcurves(observations, model, parameters.iloc[:1],
                              trim_observations=True, scatter=False)
    assert list(lcs["mjd"]) == [100., 101., 120.], "mjd=160 is outside the model time range"

def test_realize_lightcurves_phase_range(model, observations, parameters):
    # target 0: z=0.1, t0=100 => phase [0, 5] <=> mjd [100, 105.5]
    lcs = realize_lightcurves(observations, model, parameters.iloc[:1],
                              phase_range=[0, 5], scatter=False)
    assert list(lcs["mjd"]) == [100., 101.]

def test_realize_lightcurves_skip_without_observations(model, observations, parameters):
    # target 7 (t0=110, z=0.2) has no observations in phase [-10, -9] <=> mjd [98, 99.2]
    # target 3 (t0=100, z=0.1) => mjd [89, 90.1] neither.
    assert realize_lightcurves(observations, model, parameters,
                               phase_range=[-10, -9], scatter=False) is None

    # target 3 has phase 0 at mjd=100, target 7 has not (mjd 110 not observed)
    lcs = realize_lightcurves(observations, model, parameters,
                              phase_range=[0, 0.5], scatter=False)
    assert list(lcs.index.get_level_values(0).unique()) == [3]


# tests for the func get_obsdata
def test_get_obsdata(model, observations, parameters):
    data = get_obsdata(model, observations, parameters, incl_error=False)
    expected = realize_lightcurves(observations, model, parameters, scatter=False)
    pandas.testing.assert_frame_equal(data, expected)

def test_get_obsdata_default_zpsys(model, observations, parameters):
    observations = observations.drop(columns="zpsys")
    data = get_obsdata(model, observations, parameters, zpsys="vega", incl_error=False)
    assert (data["zpsys"] == "vega").all()

    # vega and ab zero points differ, so do fluxes.
    data_ab = get_obsdata(model, observations.drop(columns="zpsys"), parameters,
                          zpsys="ab", incl_error=False)
    assert not np.allclose(data["flux"], data_ab["flux"])

def test_get_obsdata_discard_bands(model, observations, parameters):
    # the model starts at 3000AA rest-frame and bessellb at 3600AA.
    # => bessellb only works for z < 3600/3000 - 1 = 0.2: a z=0.3 target is discarded in b.
    parameters.loc[7, "z"] = 0.3
    data = get_obsdata(model, observations, parameters, incl_error=False, discard_bands=True)

    assert set(data.xs(3)["band"]) == {"bessellb", "bessellv"}
    assert set(data.xs(7)["band"]) == {"bessellv"}

    expected_b = realize_lightcurves(observations[observations["band"] == "bessellb"],
                                     model, parameters.iloc[:1], scatter=False)
    np.testing.assert_allclose(data[data["band"] == "bessellb"]["flux"], expected_b["flux"])


# tests for the func _get_obsdata_
def test__get_obsdata_(model, observations, parameters):
    # single-input version of get_obsdata (for map)
    data = _get_obsdata_([model, observations, parameters], incl_error=False)
    expected = get_obsdata(model, observations, parameters, incl_error=False)
    pandas.testing.assert_frame_equal(data, expected)
