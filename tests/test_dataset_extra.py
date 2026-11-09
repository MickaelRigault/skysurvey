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


# tests for the func get_data (join_bandday)
@pytest.fixture
def dataset_bandday():
    # target 0: two g-band obs on day 0, one r-band obs on day 0, one g-band obs on day 1
    index = pandas.MultiIndex.from_tuples([(0, 0), (0, 1), (0, 2), (0, 3)], names=["index", "index_obs"])
    data = pandas.DataFrame({"mjd": [0.1, 0.6, 0.2, 1.3],
                             "band": ["g", "g", "r", "g"],
                             "flux": [10., 20., 5., 7.],
                             "fluxerr": [3., 4., 1., 2.]}, index=index)
    return DataSet(data)

def test_get_data_join_bandday_first(dataset_bandday):
    data = dataset_bandday.get_data(join_bandday=True, join_how="first")

    assert len(data) == 3 # (g, day0), (r, day0), (g, day1)
    g_day0 = data[(data["band"] == "g") & (data["mjd_date"] == 0)]
    assert g_day0["flux"].iloc[0] == 10.
    assert g_day0["fluxerr"].iloc[0] == 3.
    assert "fluxvar" not in data, "temporary fluxvar column should be removed"

def test_get_data_join_bandday_sum_flux(dataset_bandday):
    data = dataset_bandday.get_data(join_bandday=True, join_how="sum")
    g_day0 = data[(data["band"] == "g") & (data["mjd_date"] == 0)]
    assert g_day0["flux"].iloc[0] == 30.
    # variance adds up: sqrt(3**2 + 4**2) = 5
    assert g_day0["fluxerr"].iloc[0] == pytest.approx(5.)

def test_get_data_join_bandday_single_obs_unchanged(dataset_bandday):
    data = dataset_bandday.get_data(join_bandday=True, join_how="sum")
    r_day0 = data[data["band"] == "r"]
    assert r_day0["flux"].iloc[0] == 5.
    assert r_day0["fluxerr"].iloc[0] == pytest.approx(1.)

def test_get_data_join_bandday_mean_flux(dataset_bandday):
    data = dataset_bandday.get_data(join_bandday=True, join_how="mean")
    g_day0 = data[(data["band"] == "g") & (data["mjd_date"] == 0)]
    assert g_day0["flux"].iloc[0] == 15.

def test_get_data_join_bandday_keeps_existing_fluxvar(dataset_bandday):
    dataset_bandday.data["fluxvar"] = dataset_bandday.data["fluxerr"]**2
    data = dataset_bandday.get_data(join_bandday=True, join_how="sum")
    assert "fluxvar" in data
    g_day0 = data[(data["band"] == "g") & (data["mjd_date"] == 0)]
    assert g_day0["fluxvar"].iloc[0] == pytest.approx(25.)

def test_get_data_join_bandday_unknown_how(dataset_bandday):
    with pytest.raises(NotImplementedError):
        dataset_bandday.get_data(join_bandday=True, join_how="not_a_groupby_method")
