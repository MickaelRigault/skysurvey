import numpy as np
import pytest
import pandas
from unittest.mock import patch
from skysurvey.dataset import DataSet

# tests for the class DataSet
def test_dataset_default_initialization():
    index = pandas.MultiIndex.from_product([[0, 1], [0]], names=["index", "obs"])
    data = pandas.DataFrame({"mjd":[0., 10.],
        "band": ["sdssg", "sdssr"],
        "flux": [100.0, 1.0],
        "fluxerr": [10.0, 10.0],
        "zp": [25.0, 25.0],
        "zpsys": ["ab", "ab"]}, index=index)
    dataset = DataSet(data=data, targets="targets", survey="survey")

    assert dataset.data.equals(data)
    assert dataset.targets == "targets"
    assert dataset.survey == "survey"

    assert dataset._obs_index is None

    obs_idx = dataset.obs_index
    assert list(obs_idx) == [0, 1]

# tests for the func from_targets_and_survey
class FakeModel:
    def __init__(self):
        self.param_names = ["t0", "z"]
        self.parameters = [0., 1.]
    
    def minwave(self):
        return 3000.0
    
    def maxwave(self):
        return 9000.0
    
    def bandflux(self, band, mjd, **kwargs):
        return np.ones(len(mjd)) * 100.0


class FakeTargets:
    def __init__(self):
        self.data = pandas.DataFrame({
            "ra": [10., 20.],
            "dec": [10., 20.],
            "t0": [0.0, 0.0],
            "z": [1.0, 1.0]
        })
        self.data.index.name = "index"

    def get_target_template(self, index=None, as_model=True, set_magabs=True):
        return FakeModel()
    
    def show_lightcurve(self, *args, **kwargs):
        return kwargs["fig"]

@pytest.fixture
def targets():
    return FakeTargets()

class FakeFieldIDs1D:
    names = ["fieldid"]

class FakeSurvey1D:
    def __init__(self):
        self.fieldids = FakeFieldIDs1D()
        self.data = pandas.DataFrame({
            "mjd": [0., 10.],
            "band": ["sdssg", "sdssr"],
            "skynoise": [1.0, 1.0],
            "gain": [1.0 , 1.0],
            "zp": [25.0, 25.0],
            "fieldid": [1, 1]
        })
        self.data.index.name = "index_obs"
    
    def radec_to_fieldid(self, coords):
        return pandas.DataFrame({"fieldid": [1]}, index=coords.index)


@pytest.fixture
def survey():
    return FakeSurvey1D()
 
def test_from_targets_and_survey(targets, survey):
    dataset = DataSet.from_targets_and_survey(targets, survey, incl_error=False, phase_range=None, discard_bands=False)

    assert isinstance(dataset, DataSet)
    assert dataset.targets is targets
    assert dataset.survey is survey

def test_from_targets_and_survey_triggers_target_collection(monkeypatch, survey):
    triggered = {}

    class DummyTargetCollection:
        def __init__(self, targets):
            triggered["yes"] = True 
            self.data = pandas.DataFrame({
            "ra": [10.],
            "dec": [10.],
            "t0": [0.],
            "z": [1.]
        })
        def get_target_template(self, index=None, as_model=True, set_magabs=True):
            return FakeModel()

    monkeypatch.setattr("skysurvey.dataset.TargetCollection", DummyTargetCollection)
    DataSet.from_targets_and_survey(targets=[1, 2], survey=survey, discard_bands=False)

    assert "yes" in triggered

def test_from_targets_and_survey_time_cut_on_logs(targets, survey):
    phase_range = [-1, +1]

    dataset = DataSet.from_targets_and_survey(targets, survey, incl_error=False, phase_range=phase_range, discard_bands=False)

    assert len(dataset.data) == 2
    assert set(dataset.data["mjd"]) == {0}

def test_from_targets_and_survey_include_error(targets, survey):
    dataset_without_error = DataSet.from_targets_and_survey(targets, survey, incl_error=False, discard_bands=False)
    dataset_with_error = DataSet.from_targets_and_survey(targets, survey, incl_error=True, seed = 61, discard_bands=False)

    assert not np.allclose(dataset_without_error.data["flux"], dataset_with_error.data["flux"])

class FakeFieldIDs2D:
    names = ["fieldids1", "fieldids2"]

class FakeSurvey2D():
    def __init__(self):
        self.fieldids = FakeFieldIDs2D()
        self.data = pandas.DataFrame({
                "mjd": [0., 10.],
                "band": ["sdssg", "sdssr"],
                "skynoise": [1.0, 1.0],
                "gain": [1.0 , 1.0],
                "zp": [25.0, 25.0],
                "fieldids1": [1, 1],
                "fieldids2": [1, 1]
            })
        self.data.index.name = "index_obs"
        
    def radec_to_fieldid(self, coords):
        return pandas.DataFrame({"fieldids1": [1], "fieldids2": [1]}, index=coords.index)

def test_from_targets_and_survey_2d_fields():
    targets = FakeTargets()
    survey = FakeSurvey2D()

    with patch("skysurvey.tools.speedutils.isin_pair_elements", return_value= np.array([True]*len(targets.data))) as mock:
        DataSet.from_targets_and_survey(targets, survey, discard_bands=False)

        mock.assert_called_once()

class FakeFieldIDs3D:
    names = ["fieldids1", "fieldids2", "fieldids3"]

class FakeSurvey3D():
    def __init__(self):
        self.fieldids = FakeFieldIDs3D()
        self.data = pandas.DataFrame({
                "mjd": [0., 10.],
                "band": ["sdssg", "sdssr"],
                "skynoise": [1.0, 1.0],
                "gain": [1.0 , 1.0],
                "zp": [25.0, 25.0],
                "fieldids1": [1, 1],
                "fieldids2": [1, 1],
                "fieldids3": [1, 1],
            })
        self.data.index.name = "index_obs"
        
    def radec_to_fieldid(self, coords):
        return pandas.DataFrame({"fieldids1": [1], "fieldids2": [1], "fieldids3": [1]}, index=coords.index)

def test_from_targets_and_survey_fields_not_implemented():
    targets = FakeTargets()
    survey = FakeSurvey3D()

    with pytest.raises(NotImplementedError):
        DataSet.from_targets_and_survey(targets, survey, discard_bands=False)

def test_from_targets_and_survey_progress_bar(targets, survey):
    with patch("tqdm.tqdm") as mock_tqdm:
        mock_tqdm.side_effect = lambda x, **kwargs: x

        DataSet.from_targets_and_survey(targets, survey, progress_bar=True, discard_bands=False)

        mock_tqdm.assert_called_once()

def test_from_targets_and_survey_index_name_none(targets, survey):
    targets.data.index.name = None
    survey.data.index.name = None

    dataset = DataSet.from_targets_and_survey(targets, survey, discard_bands=False)

    assert isinstance(dataset, DataSet)
    assert len(dataset.data) > 0

# tests for the func read_parquet
def test_read_parquet(monkeypatch):
    index = pandas.MultiIndex.from_product([[0, 1], [0]], names=["index", "obs"])
    data = pandas.DataFrame({"mjd":[0., 10.],
        "band": ["sdssg", "sdssr"],
        "flux": [100.0, 1.0],
        "fluxerr": [10.0, 10.0],
        "zp": [25.0, 25.0],
        "zpsys": ["ab", "ab"]}, index=index)
    
    def pandas_read_parquet(path, **kwargs):
        assert path == "file.parquet"
        return data
    
    monkeypatch.setattr(pandas, "read_parquet", pandas_read_parquet)

    dataset = DataSet.read_parquet("file.parquet", survey="survey", targets="targets")

    assert isinstance(dataset, DataSet)
    assert dataset.data.equals(data)
    assert dataset.survey == "survey"
    assert dataset.targets == "targets"

# tests for the func read_from_directory 
def test_read_from_directory():
    with pytest.raises(NotImplementedError):
        DataSet.read_from_directory("dirname")

# tests for the func set_data
def test_set_data():
    dataset = DataSet(data=None)

    dataset._obs_index = [1,2,3]

    index = pandas.MultiIndex.from_product([[0, 1], [0]], names=["index", "obs"])
    data = pandas.DataFrame({"mjd":[0., 10.],
        "band": ["sdssg", "sdssr"],
        "flux": [100.0, 1.0],
        "fluxerr": [10.0, 10.0],
        "zp": [25.0, 25.0],
        "zpsys": ["ab", "ab"]}, index=index)

    dataset.set_data(data)

    assert dataset._data.equals(data)
    assert dataset._obs_index is None

# tests for the func set_targets
def test_set_targets(targets):
    dataset = DataSet(data=None)
    dataset.set_targets(targets)

    assert dataset.targets is targets

# tests for the func set_survey
def test_set_survey(survey):
    dataset = DataSet(data=None)
    dataset.set_survey(survey)

    assert dataset.survey is survey

# tests for the func get_data
@pytest.fixture
def dataset(targets, survey):
    index = pandas.MultiIndex.from_product([[0, 1], [0]], names=["index", "obs"])
    data = pandas.DataFrame({"mjd":[0., 10.],
        "band": ["sdssg", "sdssr"],
        "flux": [100.0, 1.0],
        "fluxerr": [10.0, 10.0],
        "zp": [25.0, 25.0],
        "zpsys": ["ab", "ab"]}, index=index)
    
    return DataSet(data=data, targets=targets, survey=survey)

def test_get_data(dataset):
    result = dataset.get_data()

    assert isinstance(result, pandas.DataFrame)
    assert len(result) == 2

def test_get_data_index_not_none(dataset):
    result = dataset.get_data(index=[0])

    assert len(result) == 1

def test_get_data_add_phase_true(dataset):
    result = dataset.get_data(add_phase=True)

    assert "phase" in result
    assert "phase_obs" in result

    assert np.allclose(result["phase_obs"], [0,10])
    assert np.allclose(result["phase"], [0,5])

def test_get_data_phase_range_not_none(dataset):
    result = dataset.get_data(phase_range=[-1, +1])

    assert "phase" in result
    assert "phase_obs" in result
    assert len(result) == 1

def test_get_data_detection_true(dataset):
    result = dataset.get_data(detection=True)

    assert len(result) == 1
    assert (result["flux"] / result["fluxerr"]).iloc[0] >= 5


def test_get_data_detection_false(dataset):
    result = dataset.get_data(detection=False)

    assert len(result) == 1
    assert (result["flux"] / result["fluxerr"]).iloc[0] < 5  

def test_get_data_zp_not_none(dataset):
    result = dataset.get_data(zp = 30.)
    coef = 10 ** (-(25. - 30.) / 2.5)

    assert np.all(result["zp"] == 30.)
    assert np.allclose(result["flux"], [100.*coef, 1.*coef])
    assert np.allclose(result["fluxerr"], [10.*coef, 10.*coef])

# tests for the func get_ndetection
def test_get_ndetection(dataset):
    with patch.object(dataset, "get_data") as mock_get:

        mock_get.return_value = dataset.data.iloc[:1]
        dataset.get_ndetection(join_bandday=True)

        mock_get.assert_called_once_with(phase_range=None, detection=True, join_bandday=True)

@pytest.fixture
def dataset_ndetection(targets, survey):
    index = pandas.MultiIndex.from_tuples([(0,0),(0,1),(1,0)],names=["index","obs"])
    data = pandas.DataFrame({"mjd":[0., 1., 2.], 
        "band":["sdssg", "sdssr", "sdssg"], 
        "flux":[100., 100., 100.],
        "fluxerr":[10., 10., 10.],
        "zp":[25., 25., 25.],
        "zpsys":["ab", "ab", "ab"]}, index=index)

    return DataSet(data=data, targets=targets, survey=survey)
        
def test_get_ndetection_per_band_true(dataset_ndetection):
    result = dataset_ndetection.get_ndetection(per_band=True)

    assert result.loc[(0,"sdssg")] == 1
    assert result.loc[(0,"sdssr")] == 1
    assert result.loc[(1,"sdssg")] == 1
    
def test_get_ndetection_per_band_false(dataset_ndetection):
    result = dataset_ndetection.get_ndetection(per_band=False)

    assert result.loc[0] == 2
    assert result.loc[1] == 1

# tests for the fuc get_target_lightcurve
def test_get_target_lightcurve(dataset):

    dataframe = pandas.DataFrame({"a":[1]})

    with patch.object(dataset, "get_data", return_value=dataframe) as mock_get:

        result = dataset.get_target_lightcurve(index=2, detection=True, phase_range=(-1,+1))

        mock_get.assert_called_once_with(index=2, detection=True, phase_range=(-1,+1))

        assert result is dataframe

# tests for the func show_target_lightcurve
def test_show_target_lightcurve(dataset):

    dataframe = dataset.get_target_lightcurve(index=0)

    with patch.object(dataset, "get_target_lightcurve", return_value=dataframe) as mock_lc:

        dataset.show_target_lightcurve(index=0, format_time=False, show_truth=False)

        mock_lc.assert_called_once()

@pytest.mark.filterwarnings("ignore:ERFA function.*dubious year")
def test_show_target_lightcurve_format_time_true(dataset, monkeypatch):
    fig = dataset.show_target_lightcurve(index=0, show_truth=False)

    assert fig is not None

def test_show_target_lightcurve_show_truth_true(dataset, monkeypatch):

    called = {}

    def show_fig(*args, **kwargs):
        called["yes"] = True
        return kwargs["fig"]

    monkeypatch.setattr(dataset.targets,"show_lightcurve", show_fig)

    fig = dataset.show_target_lightcurve(index=0, format_time=False)

    assert called["yes"]
    assert fig is not None

def test_show_target_lightcurve_bands_not_none(dataset, monkeypatch):
    fig = dataset.show_target_lightcurve(index=0, bands=["sdssg"], show_truth=False, format_time=False)

    assert fig is not None

def test_show_target_lightcurve_ax_not_none(dataset, monkeypatch):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    result = dataset.show_target_lightcurve(index=0, ax=ax, show_truth=False, format_time=False)

    assert result is fig

def test_show_target_lightcurve_axe_none_fig_not_none(dataset):

    import matplotlib.pyplot as plt

    fig = plt.figure()

    result = dataset.show_target_lightcurve(index=0, fig=fig, ax=None, show_truth=False, format_time=False)

    assert result is fig

def test_show_target_lightcurve_phase_window_not_none(dataset, monkeypatch):
    fig = dataset.show_target_lightcurve(show_truth=False, format_time=False, phase_window=[-1, 1])

    assert fig is not None

def test_show_target_lightcurve_color_not_none(dataset, monkeypatch):

    def colors(bands):
        return ["red"]

    monkeypatch.setattr("skysurvey.config.get_band_color", colors)

    fig = dataset.show_target_lightcurve(index=0, show_truth=False, format_time=False)

    assert fig is not None

# tests for the func _data_index
def test_data_index(dataset):
    dataset._hdata_index = "custom"

    value = dataset._data_index

    assert value == "custom"

def test_data_index_hdata_index_missing(dataset):
    if hasattr(dataset, "_hdata_index"):
        delattr(dataset, "_hdata_index")

    value = dataset._data_index

    assert value == "index"
    assert dataset._hdata_index == "index"

# tests for the func obs_index
def test_obs_index(dataset):
    dataset._obs_index = [0, 1]

    obs = dataset.obs_index

    assert obs == [0, 1]

def test_obs_index_obs_index_missing(dataset):
    if hasattr(dataset, "_obs_index"):
        delattr(dataset, "_obs_index")

    obs = dataset.obs_index

    assert list(obs) == [0,1]