import pandas
import pytest
import sncosmo
from unittest.mock import patch, Mock
import importlib
fit_salt_module = importlib.import_module("skysurvey.lcfit.fit_salt")

# tests for the func fit_salt
class FakeDataset:
    def __init__(self, obs_index):
        self.obs_index = obs_index

@pytest.fixture
def dataset():
    return FakeDataset(obs_index = [0, 1, 2])

def test_fit_salt_indexes_none(dataset):
    result = {"t0": 56000.0, "x0": 1e-4}

    with patch.object(fit_salt_module, "fit_salt_single", return_value=result) as mock_fit:
        output = fit_salt_module.fit_salt(dataset)
    
    assert mock_fit.call_count == 3 
    assert isinstance(output, pandas.DataFrame)
    assert set(output.index) == {0, 1, 2}
    assert set(output.columns) == {"t0", "x0"}
    assert output.loc[0, "t0"] == 56000.0

def test_fit_salt_indexes_not_none(dataset):
    result = {"t0": 56000.0, "x0": 1e-4}

    with patch.object(fit_salt_module, "fit_salt_single", return_value=result) as mock_fit:
        output = fit_salt_module.fit_salt(dataset, indexes=[1])
    
    assert mock_fit.call_count == 1
    assert set(output.index) == {1}

def test_fit_salt_drops_none(dataset):
    def drops_none_results(dataset, index, **kwargs):
        if index == 0:
            return {"t0": 56000.0, "x0": 1e-4}
        elif index == 1:
            return {"t0": 56000.0, "x0": None}
        elif index == 2:
            return {"t0": None, "x0": None}
        
    with patch.object(fit_salt_module, "fit_salt_single", side_effect=drops_none_results):
        output = fit_salt_module.fit_salt(dataset)

    assert set(output.index) == {0}

class FakeFuture:
    pass

class FakeClient:
    def submit_client(self, func,  **kwargs):
        return FakeFuture

def test_fit_salt_client_as_future_not_none(dataset):
    client = FakeClient()

    with patch.object(fit_salt_module, "fit_salt_single", return_value=FakeFuture()):
        output = fit_salt_module.fit_salt(dataset, client=client, as_future=True)

    assert isinstance(output, dict)
    assert all(isinstance(values_, FakeFuture) for values_ in output.values())

def test_fit_salt_gather_client(dataset):
    with patch.object(fit_salt_module, "fit_salt_single", return_value="future"):
        client = Mock()
        client.gather.return_value = {
            0: {"t0": 56000., "x0": 1e-4},
            1: {"t0": 56050, "x0": 2e-4},
            2: {"t0":56100, "x0": 3e-4}
            }
        output = fit_salt_module.fit_salt(dataset, client=client, as_future=False)

    client.gather.assert_called_once() 
    assert isinstance(output, pandas.DataFrame)

def test_fit_salt_progress_bar_true(dataset):
    result = {"t0": 56000.0, "x0": 1e-4}

    with patch("tqdm.tqdm", side_effect=lambda x: x) as mock_tqdm:
        with patch.object(fit_salt_module, "fit_salt_single", return_value=result):
            output = fit_salt_module.fit_salt(dataset, progress_bar=True)

    mock_tqdm.assert_called_once()
    assert isinstance(output, pandas.DataFrame)

# tests for the func fit_salt_single
class FakeTargetModel:
    def __init__(self):
        self.params = {"t0": 56000.0, "x0": 1e-4, "x1": 0.5, "c": 0.01,}

    def get(self, key):
        return self.params[key]
    
    def set(self, **kwargs):
        self.params.update(kwargs)

@pytest.fixture
def target_model():
    return FakeTargetModel()

def fake_target_data():
    return pandas.DataFrame({"flux": [10.0, 2.0], "fluxerr": [0.5, 0.6]})

def test_fit_salt_single_no_data_warn_true(target_model):
    empty_target_data = pandas.DataFrame()

    with patch.object(fit_salt_module, "_dataset_to_model_and_data_", return_value=(target_model, empty_target_data)):
        with pytest.warns(UserWarning, match="no data in the target lightcurves"):
            output = fit_salt_module.fit_salt_single(dataset=None, index=[0])

    assert output is None

def test_fit_salt_single_no_data_warn_false(target_model):
    empty_target_data = pandas.DataFrame()

    with patch.object(fit_salt_module, "_dataset_to_model_and_data_", return_value=(target_model, empty_target_data)):
        output = fit_salt_module.fit_salt_single(dataset=None, index=[0], warn=False)
    
    assert output is None

def test_fit_salt_single_no_detection_warn_true(target_model):
    noisy_target_data = pandas.DataFrame({"flux": [1.0, 2.0], "fluxerr": [1.0, 2.0]})

    with patch.object(fit_salt_module, "_dataset_to_model_and_data_", return_value=(target_model, noisy_target_data)):
        with pytest.warns(UserWarning, match="no detection >5 in the target lightcurves"):
            output = fit_salt_module.fit_salt_single(dataset=None, index=[0])

    assert output is None

def test_fit_salt_single_no_detection_warn_false(target_model):
    noisy_target_data = pandas.DataFrame({"flux": [1.0, 2.0], "fluxerr": [1.0, 2.0]})

    with patch.object(fit_salt_module, "_dataset_to_model_and_data_", return_value=(target_model, noisy_target_data)):
            output = fit_salt_module.fit_salt_single(dataset=None, index=[0], warn=False)

    assert output is None

def test_fit_salt_single(target_model):
    target_data = fake_target_data()

    with patch.object(fit_salt_module, "_dataset_to_model_and_data_", return_value=(target_model, target_data)), patch.object(fit_salt_module, "sncosmo_fit_single", return_value="results",) as mock_fit:
        output = fit_salt_module.fit_salt_single(dataset=None, index=[0])
    
    assert output == "results"
    mock_fit.assert_called_once()
    assert "bounds" in mock_fit.call_args.kwargs

def test_fit_salt_client_not_none(target_model):
    target_data = fake_target_data()
    client = Mock()
    client.submit.return_value = "future"

    with patch.object(fit_salt_module, "_dataset_to_model_and_data_", return_value=(target_model, target_data)):
        output = fit_salt_module.fit_salt_single(dataset=None, index=[0], client=client)

    client.submit.assert_called_once() 
    assert output == "future"
    
# tests for the func _dataset_to_model_and_data_
class FakeTargetTemplate:
    def __init__(self, with_effects=False):
        model = sncosmo.Model(source="salt2")
        model.set(z=0.1, t0=56000.0, x0=1e-4, x1=0.5, c=0.01)

        self.sncosmo_model = Mock()
        self.sncosmo_model.source = model.source
        
        if with_effects:
            dust = sncosmo.CCM89Dust()
            model.add_effect(dust, name="mw", frame="obs")
            model.set(mwebv=0.1) 
            self.sncosmo_model.effects = [dust]
        else:
            self.sncosmo_model.effects = []

        self.sncosmo_model.get.side_effect = model.get

class FakeTargets:
    def __init__(self, template):
        self._template = template

    def get_target_template(self, index):
        return self._template

class FakeDatasetToModelAndData:
    def __init__(self, template, time_column="time"):
        self.targets = FakeTargets(template)

        data = {
            "flux": [10.0, 2.0, 1.0],
            "fluxerr": [0.5, 0.5, 0.6],
            "band": ["sdssg", "sdssr", "sdssi"],
            "zp": [25.0, 25.0, 25.0],
            "zpsys": ["ab", "ab", "ab"],
            "index": [0, 0, 0],
        }

        if time_column is not None:
            data[time_column] = [56000, 56010, 56050]
            
        self.data = pandas.DataFrame(data).set_index("index")

@pytest.mark.remote_data 
def test_dataset_to_model_and_data_model_copy():
    this_template = FakeTargetTemplate(with_effects=False)
    dataset = FakeDatasetToModelAndData(this_template, time_column="time")
    this_model, this_data =  fit_salt_module._dataset_to_model_and_data_(dataset, index=0)

    assert this_model is not this_template.sncosmo_model
    assert this_model.source.name == this_template.sncosmo_model.source.name
    assert "mwebv" not in this_model.param_names

@pytest.mark.remote_data
def test_dataset_to_model_and_data_with_effects():
    this_template = FakeTargetTemplate(with_effects=True)
    dataset = FakeDatasetToModelAndData(this_template, time_column="time")
    this_model, this_data =  fit_salt_module._dataset_to_model_and_data_(dataset, index=0)
    assert "mwebv" in this_model.param_names

@pytest.mark.remote_data
def test_dataset_to_model_and_data_rename_mjd():
    this_template = FakeTargetTemplate(with_effects=False)
    dataset = FakeDatasetToModelAndData(this_template, time_column="mjd")
    this_model, this_data =  fit_salt_module._dataset_to_model_and_data_(dataset, index=0)

    assert "time" in this_data.columns
    assert "mjd" not in this_data.columns

@pytest.mark.remote_data
def test_dataset_to_model_and_data_rename_jd():
    this_template = FakeTargetTemplate(with_effects=False)
    dataset = FakeDatasetToModelAndData(this_template, time_column="jd")
    this_model, this_data =  fit_salt_module._dataset_to_model_and_data_(dataset, index=0)

    assert "time" in this_data.columns
    assert "jd" not in this_data.columns

@pytest.mark.remote_data
def test_dataset_to_model_and_data_no_time_key():
    this_template = FakeTargetTemplate(with_effects=False)
    dataset = FakeDatasetToModelAndData(this_template, time_column=None) 
    
    with pytest.raises(ValueError, match="cannot parse time entry from input dataset, provide time_key."):
        fit_salt_module._dataset_to_model_and_data_(dataset, index=0)

@pytest.mark.remote_data
def test_dataset_to_model_and_data_custom_time_key():
    this_template = FakeTargetTemplate(with_effects=False)
    dataset = FakeDatasetToModelAndData(this_template, time_column="custom_time_key")
    this_model, this_data =  fit_salt_module._dataset_to_model_and_data_(dataset, index=0, time_key="custom_time_key")

    assert "custom_time_key" in this_data.columns

@pytest.mark.remote_data
def test_dataset_to_model_and_data_phase_range():
    this_template = FakeTargetTemplate(with_effects=False)
    dataset = FakeDatasetToModelAndData(this_template, time_column="time")
    this_model, this_data =  fit_salt_module._dataset_to_model_and_data_(dataset, index=0, phase_range=[-10,10])

    this_t0 = this_model.get("t0")
    this_redshift = this_model.get("z")

    assert ((this_data["time"] - this_t0) / (1 + this_redshift)).between(-10, 10).all()
    assert len(this_data) == 2
    assert 56050 not in this_data["time"].values