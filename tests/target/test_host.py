import numpy as np
import pytest
from skysurvey.target.host import get_sfr_as_function_of_mass_and_redshift
from skysurvey.target.host import get_schechterpdf
from skysurvey.target.host import get_stellarmassfunction

# tests for the func get_sfr_as_function_of_mass_and_redshift
@pytest.mark.parametrize("mass, redshift",
    [
        (10.0, 0.1), 
        (np.array([8.5, 10.0, 11.5]), np.array([0.1, 0.2, 0.3])),  
    ],
)
def test_get_sfr_as_function_of_mass_and_redshift_output(mass, redshift):
    sfr = get_sfr_as_function_of_mass_and_redshift(mass, redshift)

    if np.isscalar(mass):
        assert np.isscalar(sfr)
    else:
        assert isinstance(sfr, np.ndarray)
        assert sfr.shape == mass.shape

    assert np.all(sfr >= 0)

# tests for the func get_schechterpdf
@pytest.mark.parametrize("mass, mstar, alpha, phi",
    [
        (10.0, 11.0, -1.2, 1e-4),  
        (np.array([8.5, 10.0, 11.5]), 11.0, -1.2, 1e-4),
    ],
)   
def test_get_schechterpdf_single(mass, mstar, alpha, phi):
    pdf = get_schechterpdf(mass, mstar, alpha, phi, alpha2=None, phi2=None)

    if np.isscalar(mass):
        assert np.isscalar(pdf)
    else:
        assert isinstance(pdf, np.ndarray)
        assert pdf.shape == mass.shape

    assert np.all(pdf >= 0)

@pytest.mark.parametrize("mass, mstar, alpha, phi, alpha2, phi2",
    [
        (10.0, 11.0, -1.2, 1e-4, -0.5, 1e-3),
        (np.array([8.5, 10.0, 11.5]), 11.0, -1.2, 1e-4, -0.5, 1e-3),
    ],
)   
def test_get_schechterpdf_double(mass, mstar, alpha, phi, alpha2, phi2):
    pdf_single = get_schechterpdf(mass, mstar, alpha, phi, alpha2=None, phi2=None)
    pdf_double = get_schechterpdf(mass, mstar, alpha, phi, alpha2, phi2)

    assert not np.allclose(pdf_single, pdf_double)
    assert np.all(pdf_double >= 0)

# tests for the func get_stellarmassfunction
@pytest.mark.parametrize("redshift", 
    [
    1.0, np.array([1.0, 1.5]),
    0.1, np.array([0.1, 0.5, 1.0, 1.5]),
    ])
def test_get_stellarmassfunction_redshift_shape(redshift):
    xx, pdf = get_stellarmassfunction(redshift)
    redshift_ = np.atleast_1d(redshift)

    assert pdf.shape == (len(redshift_), len(xx))
    assert np.all(pdf >= 0)

@pytest.mark.parametrize("which", ["all", "blue", "red"])
def test_get_stellarmassfunction_galaxy_type(which):
    xx, pdf = get_stellarmassfunction(1.0, which=which)

    assert pdf.shape == (1, len(xx))
    assert np.all(pdf >= 0)

def test_get_stellarmassfunction_custom_xx():
    xx, pdf = get_stellarmassfunction(1.0, xx="8:11:50j")

    assert pdf.shape == (1, len(xx))

def test_get_stellarmassfunction_xx_array():
    xx_input = np.linspace(8, 11, 50)
    xx, pdf = get_stellarmassfunction(1.0, xx=xx_input)

    assert xx is xx_input  
    assert pdf.shape == (1, len(xx_input))
