import pytest
import numpy as np 
from unittest.mock import MagicMock
from skysurvey.effects.milkyway import get_mwebv
from skysurvey.effects.milkyway import mwebv_model

# tests for the func get_mwebv
def test_get_mwebv_planck_map(monkeypatch):
    fake_query_instance = MagicMock(return_value=np.array([0.1]))
    FakePlanckQuery = MagicMock(return_value=fake_query_instance)
    monkeypatch.setattr("dustmaps.planck.PlanckQuery", FakePlanckQuery)

    ebv_param = get_mwebv(17.45, -29.0, which="planck")

    FakePlanckQuery.assert_called_once()
    assert np.allclose(ebv_param, np.array([0.1]))

def test_get_mwebv_sfd_map(monkeypatch):
    fake_query_instance = MagicMock(return_value=np.array([0.1]))
    FakeSFDQuery = MagicMock(return_value=fake_query_instance)
    monkeypatch.setattr("dustmaps.sfd.SFDQuery", FakeSFDQuery) 

    ebv_param = get_mwebv(17.45, -29.0, which="sfd")
 
    FakeSFDQuery.assert_called_once()
    assert np.allclose(ebv_param, np.array([0.1]))

def test_get_mwebv_invalid_map(): 
    with pytest.raises(NotImplementedError):
        get_mwebv(17.45, -29.0, which="unknown")
        
# tests for the dict mwebv_model
def test_dust_model_keys():
    assert "mwebv" in mwebv_model

    for key in mwebv_model:
        entry = mwebv_model[key]
        assert "func" in entry
        assert callable(entry["func"])
        assert "kwargs" in entry
        assert isinstance(entry["kwargs"], dict)
