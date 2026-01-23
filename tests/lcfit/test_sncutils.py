import numpy as np
import pytest
import pandas
from unittest.mock import patch
from skysurvey.lcfit.sncutils import sncosmo_results_to_dataframe, sncosmo_fit_single

# tests for the func sncosmo_results_to_dataframe
class FakeSncosmoResult:
    def __init__(self):

        self.param_names = ["t0", "x0", "x1", "c"]
        self.vparam_names = ["t0", "x0"]
        self.parameters = [55000., 1e-4, 0.5, 0.01]
        self._errors = {"t0":0.1, "x0":1e-6}
        self._covariance = [[0.01, 1.e-6],[1.e-6, 1.e-12],] 

    def __getitem__(self, key):
        if key == "errors":
            return self._errors
        if key == "covariance":
            return self._covariance
        raise KeyError(key)
    
@pytest.fixture
def result():
    return FakeSncosmoResult()

def test_sncosmo_results_to_dataframe_notflattened(result):
    dataframe = sncosmo_results_to_dataframe(result, flatten=False)

    assert set(dataframe.index) == {"t0", "x0", "x1", "c"}
    assert set(dataframe.columns) == {"values", "fitted", "errors", "cov_t0", "cov_x0"}

    assert dataframe.loc["t0", "values"] == 55000.
    assert dataframe.loc["x0", "values"] == 1e-4
    assert dataframe.loc["x1", "values"] == 0.5
    assert dataframe.loc["c", "values"] == 0.01

    assert dataframe.loc["t0", "fitted"] 
    assert dataframe.loc["x0", "fitted"]
    assert not dataframe.loc["x1", "fitted"] 
    assert not dataframe.loc["c", "fitted"] 

    assert dataframe.loc["t0", "errors"] == 0.1
    assert dataframe.loc["x0", "errors"] == 1e-6
    assert np.isnan(dataframe.loc["x1", "errors"])
    assert np.isnan(dataframe.loc["c", "errors"])

    assert dataframe.loc["t0", "cov_t0"] == 0.01
    assert dataframe.loc["t0", "cov_x0"] == 1.e-6
    assert dataframe.loc["x0", "cov_t0"] == 1.e-6
    assert dataframe.loc["x0", "cov_x0"] == 1.e-12

    assert np.isnan(dataframe.loc["x1", "cov_t0"])
    assert np.isnan(dataframe.loc["x1", "cov_x0"])
    assert np.isnan(dataframe.loc["c", "cov_t0"])
    assert np.isnan(dataframe.loc["c", "cov_x0"])

def test_sncosmo_results_to_dataframe_flattened(result):
    series = sncosmo_results_to_dataframe(result, flatten=True)

    keys =  {
        "t0",
        "t0_err",
        "cov_t0t0",
        "cov_x0t0",
        "x0",
        "x0_err",
        "cov_t0x0",
        "cov_x0x0",
        "x1",
        "c",
    }

    assert set(series.index) == keys
    
    assert series["t0"] == 55000.
    assert series["t0_err"] == 0.1
    assert series["x0"] == 1e-4
    assert series["x0_err"] == 1e-6
    assert series["x1"] == 0.5
    assert series["c"] == 0.01

    assert series["cov_t0t0"] == 0.01
    assert series["cov_t0x0"] == 1.e-6
    assert series["cov_x0t0"] == 1.e-6
    assert series["cov_x0x0"] == 1.e-12

# tests for the func sncosmo_fit_single
def test_sncosmo_fit_single(result):
    target_data = pandas.DataFrame({
        "time": [55070.0, 55072.0],
        "band": ["sdssg", "sdssr"],
        "flux": [1.0, 2.0],
        "fluxerr": [0.5, 0.6],
        "zp": [25.0, 25.0],
        "zpsys": ["ab", "ab"]
    })

    target_model = object()
    free_param = ["t0", "x0"]
    fitted_model = object()

    with patch("sncosmo.fit_lc", return_value=(result, fitted_model)) as mock_fit:
        output = sncosmo_fit_single(target_data, target_model, free_param, modelcov=False)

    args, kwargs = mock_fit.call_args

    lc_dict = args[0]
    assert set(lc_dict.keys()) == {"time", "band", "flux", "fluxerr","zp", "zpsys"}

    assert (lc_dict["time"] == target_data["time"].values).all()
    assert (lc_dict["band"] == target_data["band"].values).all()
    assert (lc_dict["flux"] == target_data["flux"].values).all()
    assert (lc_dict["fluxerr"] == target_data["fluxerr"].values).all()
    assert (lc_dict["zp"] == target_data["zp"].values).all()
    assert (lc_dict["zpsys"] == target_data["zpsys"].values).all()

    assert kwargs["model"] is target_model
    assert kwargs["vparam_names"] == free_param
    assert kwargs["modelcov"] is False

    expected_ouput = sncosmo_results_to_dataframe(result)
    pandas.testing.assert_series_equal(output, expected_ouput)

def test_sncosmo_fit_single_keymap(result):
    target_data = pandas.DataFrame({
        "time": [55070.0],
        "band": ["sdssg"],
        "flux": [1.0],
        "fluxerr_tot": [0.5],
        "zp": [25.0],
        "zpsys": ["ab"]
    })

    keymap = {"fluxerr": "fluxerr_tot"}
    target_model = object()
    free_param = ["t0", "x0"]
    fitted_model = object()

    with patch("sncosmo.fit_lc", return_value=(result, fitted_model)) as mock_fit:
        sncosmo_fit_single(target_data, target_model, free_param, modelcov=False, keymap=keymap)

    lc_dict = mock_fit.call_args[0][0]
    assert "fluxerr" in lc_dict
    assert (lc_dict["fluxerr"] == target_data["fluxerr_tot"].values).all()