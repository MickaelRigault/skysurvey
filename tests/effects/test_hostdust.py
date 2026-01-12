import numpy as np
from skysurvey.effects.hostdust import ebv
from skysurvey.effects.hostdust import dust_model

# tests for the class ebv_distrib
def test_ebv_pdf_negative_values():
    x = np.linspace(-5, -0.1, 50)
    pdf = ebv.pdf(x)

    assert np.all(pdf == 0)

def test_ebv_pdf_both_values():
    x = np.linspace(-5, 5, 100)
    pdf = ebv.pdf(x)

    assert np.all(pdf >= 0)

def test_ebv_pdf_positive_values():
    x = np.linspace(0.1, 5, 50)
    pdf = ebv.pdf(x)

    assert np.all(pdf > 0)

# tests for the dict dust_model
def test_dust_model_keys():
    assert "hostebv" in dust_model
    assert "hostr_v" in dust_model

    for key in dust_model:
        entry = dust_model[key]
        assert "func" in entry
        assert callable(entry["func"])
        assert "kwargs" in entry
        assert isinstance(entry["kwargs"], dict)