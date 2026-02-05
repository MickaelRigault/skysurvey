import numpy as np
import pytest
from skysurvey.target.snia import SNeIaColor, SNeIaStretch, SNeIaMagnitude, SNeIa
from skysurvey.tools.utils import random_radec

# tests for the class SNeIaColor
# tests for the func color_rvs 
def test_color_rvs_output():
    x = SNeIaColor.color_rvs(1000)

    assert len(x) == 1000
    assert isinstance(x, np.ndarray)

def test_color_rvs_negative_size():
    with pytest.raises(ValueError):
        SNeIaColor.color_rvs(-1000)

# tests for the func asymetric_gaussian
def test_asymetric_gaussian_output():
    xx, pdf = SNeIaColor.asymetric_gaussian()

    assert isinstance(xx, np.ndarray)
    assert isinstance(pdf, np.ndarray)
    assert pdf.shape == xx.shape
    assert np.all(pdf >= 0)

def test_asymetric_gaussian_custom_xx():
    xx, pdf = SNeIaColor.asymetric_gaussian(xx="-0.2:2:0.002")

    assert xx.shape == pdf.shape
    assert len(xx) > 0
    assert np.all(pdf >= 0)

def test_asymetric_gaussian_xx_array():
    xx_input = np.linspace(-0.1, 1,  1000)
    xx, pdf = SNeIaColor.asymetric_gaussian(xx=xx_input)

    assert xx is xx_input  
    assert pdf.shape == xx_input.shape
    assert np.all(pdf >= 0)

# test for the func intrinsic_and_dust
def test_intrinsic_and_dust_output():
    xx, pdf = SNeIaColor.intrinsic_and_dust()

    assert isinstance(xx, np.ndarray)
    assert isinstance(pdf, np.ndarray)
    assert pdf.shape == xx.shape
    assert np.all(pdf >= 0)

def test_intrinsic_and_dust_custom_xx():
    xx, pdf = SNeIaColor.intrinsic_and_dust(xx="-0.2:2:0.002")

    assert xx.shape == pdf.shape
    assert len(xx) > 0
    assert np.all(pdf >= 0)

def test_intrinsic_and_dust_custom_xx_array():
    xx_input = np.linspace(-0.1, 1,  1000)
    xx, pdf = SNeIaColor.intrinsic_and_dust(xx=xx_input)

    assert xx is xx_input  
    assert pdf.shape == xx_input.shape
    assert np.all(pdf >= 0)

# tests for the class SNeIaStretch
# tests for the func nicolas2021
def test_nicolas2021_output():
    xx, pdf = SNeIaStretch.nicolas2021()

    assert isinstance(xx, np.ndarray)
    assert isinstance(pdf, np.ndarray)
    assert pdf.shape == xx.shape
    assert np.all(pdf >= 0)

def test_nicolas2021_custom_xx():
    xx, pdf = SNeIaStretch.nicolas2021(xx="-2:2:0.002")

    assert xx.shape == pdf.shape
    assert len(xx) > 0
    assert np.all(pdf >= 0)

def test_nicolas2021_custom_xx_array():
    xx_input = np.linspace(-1, 1,  1000)
    xx, pdf = SNeIaStretch.nicolas2021(xx=xx_input)

    assert xx is xx_input  
    assert pdf.shape == xx_input.shape
    assert np.all(pdf >= 0)   

def test_nicolas2021_fpromt_not_float():
    fprompt = [0.2, 0.5, 0.8]
    xx, pdf = SNeIaStretch.nicolas2021(fprompt = fprompt)

    assert isinstance(pdf, np.ndarray)
    assert pdf.ndim == 2
    assert pdf.shape[0] == len(fprompt)
    assert pdf.shape[1] == len(xx)

# tests for the class SNeIaMagnitude
# tests for the func tripp1998
def test_tripp1998_output():
    x1 = np.array([0.0, 0.5, -0.5])
    c = np.array([0.0, -0.2, 0.2])
    mabs_tripp1998 = SNeIaMagnitude.tripp1998(x1, c, rng=24)

    assert mabs_tripp1998.shape == x1.shape
    assert isinstance(mabs_tripp1998, np.ndarray)

def test_tripp1998_same_seed():
    x1 = np.array([0.0, 0.5, -0.5])
    c = np.array([0.0, -0.2, 0.2])
    mabs_tripp1998_1 = SNeIaMagnitude.tripp1998(x1, c, rng=24)
    mabs_tripp1998_2 = SNeIaMagnitude.tripp1998(x1, c, rng=24)

    assert np.allclose(mabs_tripp1998_1, mabs_tripp1998_2)

def test_tripp1998_different_seed():
    x1 = np.array([0.0, 0.5, -0.5])
    c = np.array([0.0, -0.2, 0.2])
    mabs_tripp1998_1 = SNeIaMagnitude.tripp1998(x1, c, rng=24)
    mabs_tripp1998_2 = SNeIaMagnitude.tripp1998(x1, c, rng=33)

    assert not np.allclose(mabs_tripp1998_1, mabs_tripp1998_2)

def test_trip1998_no_randomness():
    x1 = np.array([0.0, 0.5, -0.5])
    c = np.array([0.0, -0.2, 0.2])
    mabs_tripp1998 = SNeIaMagnitude.tripp1998(x1, c, sigmaint=0.0, rng=24)

    mabs_tripp1998_expected = -19.3 + (x1 * -0.14 + c * 3.15)

    assert np.allclose(mabs_tripp1998, mabs_tripp1998_expected)

# tests for the func tripp_and_step
def test_tripp_and_step_output():
    x1 = np.array([0.5, -0.5])
    c = np.array([-0.2, 0.2])
    isup = np.array([0, 1])
    mabs_tripp_and_step = SNeIaMagnitude.tripp_and_step(x1, c, isup)

    assert mabs_tripp_and_step.shape == isup.shape
    assert isinstance(mabs_tripp_and_step, np.ndarray)

def test_tripp_and_step_no_randomness():
    x1 = np.array([0.5, -0.5])
    c = np.array([-0.2, 0.2])
    isup = np.array([0, 1])
    gamma=0.1
    mabs_tripp_and_step = SNeIaMagnitude.tripp_and_step(x1, c, isup, sigmaint=0.0)

    mabs_trip1998 = -19.3 + (x1 * -0.14 + c * 3.15)
    mabs_tripp_and_step_expected = mabs_trip1998 + (isup - 0.5) * gamma

    assert np.allclose(mabs_tripp_and_step, mabs_tripp_and_step_expected)

# tests for the functripp_and_massstep
def test_tripp_and_massstep_output():
    x1 = np.array([0.5, -0.5])
    c = np.array([-0.2, 0.2])
    hostmass = np.array([10.0, 11.0])
    mabs_tripp_and_massstep = SNeIaMagnitude.tripp_and_massstep(x1, c, hostmass)

    assert mabs_tripp_and_massstep.shape == c.shape
    assert isinstance(mabs_tripp_and_massstep, np.ndarray)

def test_tripp_and_massstep_no_randomness():
    x1 = np.array([0.5, -0.5])
    c = np.array([-0.2, 0.2])
    hostmass = np.array([10.0, 11.0])
    split = 10.0
    gamma = 0.1

    mabs_tripp_and_massstep = SNeIaMagnitude.tripp_and_massstep(x1, c, hostmass, sigmaint=0.0)
    isup = np.asarray( hostmass>split, dtype=float)
    mabs_trip1998 = -19.3 + (x1 * -0.14 + c * 3.15)
    mabs_tripp_and_massstep_expected = mabs_trip1998 + (isup - 0.5) * gamma

    assert np.allclose(mabs_tripp_and_massstep, mabs_tripp_and_massstep_expected)

# tests for the class SNeIa
def test_sneia_instantiation():
    sneia = SNeIa()

    assert sneia.__class__.__name__ == "SNeIa"

def test_sneia_attributes():
    sneia = SNeIa()

    assert hasattr(sneia, "_KIND")
    assert sneia._KIND == "SNIa"

    assert hasattr(sneia, "_TEMPLATE")
    assert sneia._TEMPLATE == "salt2"

    assert hasattr(sneia, "_RATE")
    assert sneia._RATE == 2.35 * 10**4

    assert hasattr(sneia, "_AMPLITUDE_NAME")
    assert sneia._AMPLITUDE_NAME == "x0"

    assert "_MODEL" in dir(sneia)

def test_sneia_model_keys():
    sneia = SNeIa()
    model = sneia._MODEL

    model_keys = {"redshift", "x1", "c", "t0", "magabs", "magobs", "radec"}
    assert model_keys.issubset(model.keys())

    redshift = model["redshift"]
    assert "func" in redshift
    assert redshift["func"] == "draw_redshift"
    assert redshift["kwargs"] == {"zmax":0.2}
    assert redshift["as"] == "z"

    x1 = model["x1"]
    assert "func" in x1
    assert x1["func"] is SNeIaStretch.nicolas2021

    c = model["c"]
    assert "func" in c
    assert c["func"] is SNeIaColor.intrinsic_and_dust

    t0 = model["t0"]
    assert "func" in t0
    assert callable(t0["func"])
    assert t0["func"].__name__ == "uniform"     
    assert t0["kwargs"] == {"low":56_000, "high":56_200}

    magabs = model["magabs"]
    assert "func" in magabs
    assert magabs["func"] is SNeIaMagnitude.tripp1998
    assert magabs["kwargs"] ==  {"x1": "@x1", "c": "@c", "mabs":-19.3, "sigmaint":0.10}

    magobs = model["magobs"]
    assert "func" in magobs
    assert magobs["func"] == "magabs_to_magobs"
    assert magobs["kwargs"] == {"z":"@z", "magabs":"@magabs"}

    radec = model["radec"]
    assert "func" in radec
    assert radec["func"] == random_radec
    assert radec["kwargs"] == {}
    assert radec["as"] == ["ra","dec"]