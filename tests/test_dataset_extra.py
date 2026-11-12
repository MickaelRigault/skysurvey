import warnings
import numpy as np
import pandas
import pytest
import sncosmo

import skysurvey
from skysurvey.dataset import DataSet

# in-memory bandpasses: no download needed.
# FakeModel below covers [3000, 9000] AA.
sncosmo.register(sncosmo.Bandpass([4000., 4500., 5000.], [1., 1., 1.], name="test_inside"), force=True)
sncosmo.register(sncosmo.Bandpass([2000., 2500., 3000.], [1., 1., 1.], name="test_bluer"), force=True)


class FakeModel:
    """ sncosmo-like model with constant bandflux, recording what it is given """
    def __init__(self):
        self.set_calls = []
        self.peakmag_calls = []
        self.peakabsmag_calls = []

    def minwave(self):
        return 3000.0

    def maxwave(self):
        return 9000.0

    def bandflux(self, band, mjd, **kwargs):
        return np.full(len(mjd), 100.0)

    def set(self, **kwargs):
        self.set_calls.append(kwargs)

    def set_source_peakabsmag(self, absmag, band, magsys, cosmo=None):
        self.peakabsmag_calls.append(absmag)

    def set_source_peakmag(self, m, band, magsys):
        self.peakmag_calls.append(m)


class FakeTemplate:
    def __init__(self):
        self.model = FakeModel()

    def get(self, **kwargs):
        return self.model


class FakeTargets:
    peak_absmag_band = "bessellb"
    magsys = "ab"
    cosmology = None

    def __init__(self, data=None):
        if data is None:
            data = pandas.DataFrame({"ra": [10., 20.], "dec": [10., 20.],
                                     "t0": [0., 0.], "z": [1., 1.],
                                     "magabs": [-19.3, -19.1]})
        self.data = data
        self.template = FakeTemplate()
        self.get_target_template_calls = []

    def get_template_columns(self):
        return ["t0", "z"]

    def get_target_template(self, index=None, as_model=True, set_magabs=True):
        self.get_target_template_calls.append(index)
        return FakeModel()


class FakeFieldIDs:
    names = ["fieldid"]

class FakeSurvey:
    def __init__(self, bands=["test_inside", "test_inside"]):
        self.fieldids = FakeFieldIDs()
        self.data = pandas.DataFrame({"mjd": [0., 10.],
                                      "band": bands,
                                      "skynoise": [1., 2.],
                                      "gain": [1., 4.],
                                      "zp": [25., 25.],
                                      "fieldid": [1, 1]})
        self.data.index.name = "index_obs"

    def radec_to_fieldid(self, coords):
        return pandas.DataFrame({"fieldid": np.ones(len(coords), dtype=int)}, index=coords.index)


# tests for the func lightcurve_from_targets_and_survey
def test_lightcurve_from_targets_and_survey_index():
    lc = DataSet.lightcurve_from_targets_and_survey(FakeTargets(), FakeSurvey(), incl_error=False,
                                                    phase_range=None)
    assert list(lc.index.names) == ["index", "index_obs"]
    # 2 targets x 2 observations
    assert sorted(lc.index) == [(0, 0), (0, 1), (1, 0), (1, 1)]

def test_lightcurve_from_targets_and_survey_flux_and_fluxerr():
    lc = DataSet.lightcurve_from_targets_and_survey(FakeTargets(), FakeSurvey(), incl_error=False,
                                                    phase_range=None)
    # no noise: flux is the model's bandflux
    assert np.all(lc["flux"] == 100.)
    # fluxerr = sqrt(skynoise**2 + flux/gain)
    obs0 = lc.xs(0, level="index_obs")
    obs1 = lc.xs(1, level="index_obs")
    np.testing.assert_allclose(obs0["fluxerr"], np.sqrt(1**2 + 100 / 1))
    np.testing.assert_allclose(obs1["fluxerr"], np.sqrt(2**2 + 100 / 4))

def test_lightcurve_from_targets_and_survey_seed_reproducible():
    lc1 = DataSet.lightcurve_from_targets_and_survey(FakeTargets(), FakeSurvey(), seed=3)
    lc2 = DataSet.lightcurve_from_targets_and_survey(FakeTargets(), FakeSurvey(), seed=3)
    np.testing.assert_array_equal(lc1["flux"], lc2["flux"])
    assert not np.all(lc1["flux"] == 100.), "noise should have been added"

def test_lightcurve_from_targets_and_survey_noise_statistics():
    """ (flux - true flux) / fluxerr is a unit normal """
    size = 5000
    data = pandas.DataFrame({"ra": np.zeros(size), "dec": np.zeros(size),
                             "t0": np.zeros(size), "z": np.full(size, 0.1),
                             "magabs": np.full(size, -19.3)})
    lc = DataSet.lightcurve_from_targets_and_survey(FakeTargets(data), FakeSurvey(), seed=1)
    pull = (lc["flux"] - 100.) / lc["fluxerr"]
    n = len(pull)
    assert np.abs(pull.mean()) < 5 / np.sqrt(n)
    assert np.abs(pull.std() - 1) < 5 / np.sqrt(2 * n)

def test_lightcurve_from_targets_and_survey_phase_range_restframe():
    """ the phase cut is rest-frame: (mjd - t0) / (1 + z) """
    targets = FakeTargets()  # t0=0, z=1
    # obs at mjd 0 and 10 => rest-frame phases 0 and 5
    lc = DataSet.lightcurve_from_targets_and_survey(targets, FakeSurvey(), incl_error=False,
                                                    phase_range=[-1, 5])
    assert len(lc) == 4, "phase 5 is at the (inclusive) edge"
    lc = DataSet.lightcurve_from_targets_and_survey(targets, FakeSurvey(), incl_error=False,
                                                    phase_range=[-1, 4.9])
    assert set(lc["mjd"]) == {0.}

def test_lightcurve_from_targets_and_survey_discard_bands():
    """ the bluer band is not covered by the model [3000, 9000]AA and is discarded """
    survey = FakeSurvey(bands=["test_inside", "test_bluer"])
    lc = DataSet.lightcurve_from_targets_and_survey(FakeTargets(), survey, incl_error=False,
                                                    phase_range=None, discard_bands=True)
    assert set(lc["band"]) == {"test_inside"}
    assert len(lc) == 2 # one per target

def test_lightcurve_from_targets_and_survey_discard_bands_all_kept():
    lc = DataSet.lightcurve_from_targets_and_survey(FakeTargets(), FakeSurvey(), incl_error=False,
                                                    phase_range=None, discard_bands=True)
    assert len(lc) == 4
    assert np.all(lc["flux"] == 100.)

def test_lightcurve_from_targets_and_survey_magabs():
    """ without magobs, the model amplitude is set from magabs """
    targets = FakeTargets()
    DataSet.lightcurve_from_targets_and_survey(targets, FakeSurvey(), incl_error=False, phase_range=None)
    model = targets.template.model
    assert model.peakabsmag_calls == [-19.3, -19.1]
    assert model.peakmag_calls == []

def test_lightcurve_from_targets_and_survey_magobs():
    """ magobs (if available) is used directly to set the model amplitude """
    targets = FakeTargets()
    targets.data["magobs"] = [24.1, 24.2]
    DataSet.lightcurve_from_targets_and_survey(targets, FakeSurvey(), incl_error=False, phase_range=None)
    model = targets.template.model
    assert model.peakmag_calls == [24.1, 24.2]
    assert model.peakabsmag_calls == []

def test_lightcurve_from_targets_and_survey_set_template_parameters():
    targets = FakeTargets()
    targets.data["t0"] = [3., 4.]
    DataSet.lightcurve_from_targets_and_survey(targets, FakeSurvey(), incl_error=False, phase_range=None)
    assert targets.template.model.set_calls == [{"t0": 3., "z": 1.}, {"t0": 4., "z": 1.}]

def test_lightcurve_from_targets_and_survey_not_single_model():
    """ single_model=False builds one model per target """
    targets = FakeTargets()
    lc = DataSet.lightcurve_from_targets_and_survey(targets, FakeSurvey(), incl_error=False,
                                                    phase_range=None, single_model=False)
    assert targets.get_target_template_calls == [0, 1]
    assert targets.template.model.set_calls == [], "the shared model should not be used"
    assert np.all(lc["flux"] == 100.)


# tests for the func from_targets_and_survey
@pytest.fixture
def flat_target():
    """ a real skysurvey.Target with a time- and wavelength-flat in-memory source """
    phase = np.linspace(-20, 50, 8)
    wave = np.linspace(2000, 10000, 50)
    source = sncosmo.TimeSeriesSource(phase, wave, np.ones((len(phase), len(wave))), name="flat")
    data = pandas.DataFrame({"ra": [10., 20.], "dec": [10., 20.],
                             "t0": [0., 0.], "z": [0.1, 0.1],
                             "magabs": [-19.3, -19.3]})
    return skysurvey.Target.from_data(data, template=sncosmo.Model(source))

def test_from_targets_and_survey_single_target(flat_target):
    dataset = DataSet.from_targets_and_survey(flat_target, FakeSurvey(), incl_error=False)

    assert dataset.targets is flat_target
    assert len(dataset.data) == 4 # 2 targets x 2 observations
    # flat source, same magabs and z: all fluxes are equal and positive
    flux = dataset.data["flux"]
    assert np.all(flux > 0)
    np.testing.assert_allclose(flux, flux.iloc[0])
    np.testing.assert_allclose(dataset.data["fluxerr"],
                               np.sqrt(dataset.data["skynoise"]**2 + flux / dataset.data["gain"]))


# tests for the func DataSet._coadd_observations_
def test_coadd_observations_weighted_mean():
    data = pandas.DataFrame({"key": [0, 0, 1], "mjd": [1., 2., 3.],
                             "flux": [10., 20., 5.], "fluxerr": [3., 4., 1.], "zp": 30.})
    coadded = DataSet._coadd_observations_(data, by=["key"], first_columns=["zp"]).set_index("key")
    w = np.array([1 / 9, 1 / 16]) # inverse variances
    assert np.isclose(coadded.loc[0, "flux"], (10 * w[0] + 20 * w[1]) / w.sum())
    assert np.isclose(coadded.loc[0, "fluxerr"], 1 / np.sqrt(w.sum()))
    assert np.isclose(coadded.loc[0, "mjd"], (1 * w[0] + 2 * w[1]) / w.sum())
    assert coadded.loc[0, "nvisits"] == 2 and coadded.loc[0, "zp"] == 30.
    # a single observation is unchanged
    assert (coadded.loc[1, "flux"], coadded.loc[1, "fluxerr"], coadded.loc[1, "nvisits"]) == (5., 1., 1)

def test_coadd_observations_signal_to_noise():
    # 10 observations at 4 sigma (equal errors) -> 4 x sqrt(10) sigma
    data = pandas.DataFrame({"key": 0, "mjd": np.arange(10.), "flux": 4., "fluxerr": 1.})
    coadded = DataSet._coadd_observations_(data, by=["key"])
    assert np.isclose(coadded["flux"].iloc[0] / coadded["fluxerr"].iloc[0], 4 * np.sqrt(10))

def test_coadd_observations_statistics():
    # co-added fluxes of noisy observations follow N(truth, fluxerr)
    rng = np.random.default_rng(1)
    ngroups, nobs = 20_000, 5
    fluxerr = rng.uniform(1, 3, size=ngroups * nobs)
    data = pandas.DataFrame({"key": np.repeat(np.arange(ngroups), nobs), "mjd": 0.,
                             "flux": 10. + rng.normal(0, fluxerr), "fluxerr": fluxerr})
    coadded = DataSet._coadd_observations_(data, by=["key"])
    pull = (coadded["flux"] - 10.) / coadded["fluxerr"]
    assert np.abs(pull.mean()) < 5 / np.sqrt(ngroups)
    assert np.abs(pull.std() - 1) < 5 / np.sqrt(2 * ngroups)


# tests for the func get_data (join_bandday)
@pytest.fixture
def dataset_bandday():
    # target 0: g-band obs at mjd 0.1, 0.6 and 1.3, r-band obs at mjd 0.2.
    # with nights changing at 0.9 (boundary): g at 0.1 and 0.6 in the same night, r in it too, g at 1.3 the next one.
    index = pandas.MultiIndex.from_tuples([(0, 0), (0, 1), (0, 2), (0, 3)], names=["index", "index_obs"])
    data = pandas.DataFrame({"mjd": [0.1, 0.6, 0.2, 1.3],
                             "band": ["g", "g", "r", "g"],
                             "flux": [10., 20., 5., 7.],
                             "fluxerr": [3., 4., 1., 2.],
                             "zp": 30.}, index=index)
    return DataSet(data)

def get_g_night0(data, boundary=0.9):
    """ the joined g-band observation of the night of mjd=0.1 """
    return data[(data["band"] == "g") & (data["night"] == np.floor(0.1 - boundary))].iloc[0]

def test_get_data_join_bandday_first(dataset_bandday):
    data = dataset_bandday.get_data(join_bandday=True, join_how="first", boundary=0.9)
    assert len(data) == 3 # (g, first night), (r, first night), (g, second night)
    assert data.index.names == ["index", "index_obs"]
    g_night0 = get_g_night0(data)
    assert (g_night0["flux"], g_night0["fluxerr"], g_night0["nvisits"]) == (10., 3., 2)

def test_get_data_join_bandday_last(dataset_bandday):
    g_night0 = get_g_night0(dataset_bandday.get_data(join_bandday=True, join_how="last", boundary=0.9))
    assert (g_night0["flux"], g_night0["fluxerr"]) == (20., 4.)

@pytest.mark.parametrize("join_how", ["coadd", "add", "sum"])
def test_get_data_join_bandday_coadd(dataset_bandday, join_how):
    data = dataset_bandday.get_data(join_bandday=True, join_how=join_how, boundary=0.9)
    w = np.array([1 / 9, 1 / 16]) # inverse variances of the two g-band obs of night 0
    g_night0 = get_g_night0(data)
    assert np.isclose(g_night0["flux"], (10 * w[0] + 20 * w[1]) / w.sum())
    assert np.isclose(g_night0["fluxerr"], 1 / np.sqrt(w.sum()))
    assert g_night0["nvisits"] == 2 and g_night0["zp"] == 30.
    r_night0 = data[data["band"] == "r"].iloc[0]
    assert (r_night0["flux"], r_night0["fluxerr"], r_night0["nvisits"]) == (5., 1., 1), "single obs unchanged"

def test_get_data_join_bandday_boundary(dataset_bandday):
    # with nights changing at 0.5 (12h UT): g at 0.1 alone, g at 0.6 and 1.3 in the same night
    data = dataset_bandday.get_data(join_bandday=True, join_how="coadd", boundary=0.5)
    g_data = data[data["band"] == "g"].set_index("night")
    assert sorted(g_data["nvisits"]) == [1, 2]
    w = np.array([1 / 16, 1 / 4]) # g at 0.6 (20 +/- 4) and 1.3 (7 +/- 2)
    assert np.isclose(g_data.loc[np.floor(0.6 - 0.5), "flux"], (20 * w[0] + 7 * w[1]) / w.sum())

def test_get_data_join_bandday_boundary_from_survey(dataset_bandday):
    class FakeSurveyNight:
        def get_night_boundary(self):
            return 0.5

    dataset_bandday.set_survey(FakeSurveyNight())
    data = dataset_bandday.get_data(join_bandday=True, join_how="coadd")
    expected = dataset_bandday.get_data(join_bandday=True, join_how="coadd", boundary=0.5)
    pandas.testing.assert_frame_equal(data, expected, obj="the survey boundary (0.5) should be used")

def test_get_data_join_bandday_no_survey_warns(dataset_bandday):
    with pytest.warns(UserWarning):
        data = dataset_bandday.get_data(join_bandday=True, join_how="first")
    assert len(data) == 3 # nights change at 0h UT

def test_get_data_join_bandday_coadd_zp(dataset_bandday):
    # one of the two g-band obs of night 0 at zp=27.5: 10 times fainter in flux units
    dataset_bandday.data.loc[(0, 1), ["flux", "fluxerr", "zp"]] = [2., 0.4, 27.5]
    with pytest.warns(UserWarning, match="zp=30"): # the zero point used is given
        data = dataset_bandday.get_data(join_bandday=True, join_how="coadd", boundary=0.9)
    w = np.array([1 / 9, 1 / 16]) # converted to the zp of the first obs (30): back to 20 +/- 4
    assert np.isclose(get_g_night0(data)["flux"], (10 * w[0] + 20 * w[1]) / w.sum())
    assert (data["zp"] == 30).all()
    # or explicitly at zp=30, without warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data = dataset_bandday.get_data(join_bandday=True, join_how="coadd", boundary=0.9, zp=30)
    assert np.isclose(get_g_night0(data)["flux"], (10 * w[0] + 20 * w[1]) / w.sum())

def test_get_data_join_bandday_unknown_how(dataset_bandday):
    with pytest.raises(ValueError):
        dataset_bandday.get_data(join_bandday=True, join_how="mean", boundary=0.9)

def test_get_data_join_bandday_coadd_no_zp_column(dataset_bandday):
    # without zp column, the fluxes are assumed in the same unit and co-added as they are
    dataset_bandday.data.pop("zp")
    data = dataset_bandday.get_data(join_bandday=True, join_how="coadd", boundary=0.9)
    w = np.array([1 / 9, 1 / 16])
    assert np.isclose(get_g_night0(data)["flux"], (10 * w[0] + 20 * w[1]) / w.sum())
    assert "zp" not in data

def test_get_data_join_bandday_coadd_columns(dataset_bandday):
    # one row per (target, band, night), indexed by the first observation of the night;
    # zpsys (and zp) of the first observation are kept, per-observation columns are dropped.
    dataset_bandday.data["zpsys"] = "ab"
    dataset_bandday.data["skynoise"] = 1.
    data = dataset_bandday.get_data(join_bandday=True, join_how="coadd", boundary=0.9)
    assert data.index.names == ["index", "index_obs"]
    g_data = data[data["band"] == "g"]
    assert sorted(g_data.index.get_level_values("index_obs")) == [0, 3], "first obs of each g-band night"
    assert (data["zpsys"] == "ab").all()
    assert "skynoise" not in data
    assert set(data.columns) >= {"mjd", "band", "flux", "fluxerr", "zp", "night", "nvisits"}

def test_get_data_join_bandday_first_keeps_columns(dataset_bandday):
    dataset_bandday.data["skynoise"] = [1., 2., 3., 4.]
    g_night0 = get_g_night0(dataset_bandday.get_data(join_bandday=True, join_how="first", boundary=0.9))
    assert g_night0["skynoise"] == 1., "first: all the columns of the first observation"

def test_get_ndetection_join_how_default():
    # get_ndetection co-adds by default (join_how="coadd") when joining band and night
    index = pandas.MultiIndex.from_product([[0], range(10)], names=["index", "index_obs"])
    data = pandas.DataFrame({"mjd": 100.1 + np.arange(10) * 0.01, "band": "g",
                             "flux": 4., "fluxerr": 1., "zp": 30.}, index=index)
    dataset = DataSet(data)
    assert dataset.get_ndetection(join_bandday=True, boundary=0.9).loc[0] == 1
    # "first" keeps a single 4 sigma observation: no detection
    assert len(dataset.get_ndetection(join_bandday=True, join_how="first", boundary=0.9)) == 0

def test_get_ndetection_coadd():
    # 10 observations at 4 sigma in one night: no detection per visit, one once co-added
    index = pandas.MultiIndex.from_product([[0], range(10)], names=["index", "index_obs"])
    data = pandas.DataFrame({"mjd": 100.1 + np.arange(10) * 0.01, "band": "g",
                             "flux": 4., "fluxerr": 1., "zp": 30.}, index=index)
    dataset = DataSet(data)
    assert len(dataset.get_ndetection()) == 0
    assert dataset.get_ndetection(join_bandday=True, join_how="coadd", boundary=0.9).loc[0] == 1
