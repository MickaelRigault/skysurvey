import numpy as np
import pytest
from shapely import geometry
from astropy.cosmology import Planck18
from skysurvey.tools.utils import surface_of_skyarea
from skysurvey.target.rates import get_rate
from skysurvey.target.rates import get_redshift_pdf
from skysurvey.target.rates import get_volumetric_rate

#tests for the func get_rate()
def evolving_rate(z, r0=2.3e4, alpha=1.70):
        return r0 * (1 + z)**alpha

constant_rate = 1.0e4
z = np.array([0.1, 0.2, 0.3])
circle = geometry.Point(0,0).buffer(2)
def get_expected_rate_with_skyarea(n_per_gpc3, expected_skyarea):

    skyarea = surface_of_skyarea(expected_skyarea) 
    if skyarea is None:
        return n_per_gpc3

    full_sky = 4 * np.pi * (180 / np.pi)**2 

    return n_per_gpc3 * (skyarea / full_sky)

skyarea_cases = [
    (constant_rate, circle, 4*np.pi),
    (evolving_rate, circle, 4*np.pi),
    (constant_rate, None, 4 * np.pi * (180 / np.pi)**2 ), 
    (evolving_rate, None, 4 * np.pi * (180 / np.pi)**2 ),
    (constant_rate, 100.0, 100.0),
    (evolving_rate, 100.0, 100.0),
]

@pytest.mark.parametrize(
    "rate, skyarea_input, expected_skyarea", 
    skyarea_cases
)
def test_get_rate(rate, skyarea_input, expected_skyarea):

    if callable(rate):
        n_per_gpc3 = evolving_rate(z)
    else:
        n_per_gpc3 = constant_rate 

    expected_rate = get_expected_rate_with_skyarea(n_per_gpc3, expected_skyarea)
    actual_rate = get_rate(z=z, rate=rate, skyarea=skyarea_input)

    np.testing.assert_allclose(actual_rate, expected_rate, rtol=1e-02, atol = 1e-01)
    
    if callable(rate):
        assert actual_rate.shape == expected_rate.shape
    else:
        assert isinstance(actual_rate, (float, np.float64))

# tests for the func get_redshift_pdf
def evolving_rate(z, r0=2.3e4, alpha=1.70):
        return r0 * (1 + z)**alpha

@pytest.mark.parametrize("rate", [
    1.0e4,          # float case
    evolving_rate,  # callable case
])
def test_get_redshift_pdf_normalisation(rate):
    z = np.array([0.1, 0.2, 0.3])
    pdf = get_redshift_pdf(z=z, rate=rate, skyarea=None,
                           keepsize=True, cosmology=Planck18)
    assert np.isclose(pdf.sum(), 1.0)

@pytest.mark.parametrize("rate", [
    1.0e4,
    evolving_rate,
])
@pytest.mark.parametrize("keepsize, expected_len", [
    (True, 3), # len(z)
    (False, 2), # len(z) - 1
])
def test_get_redshift_pdf_keepsize(rate, keepsize, expected_len):
    z = np.array([0.1, 0.2, 0.3])

    pdf = get_redshift_pdf(z=z, rate=rate, skyarea=None,
                           keepsize=keepsize, cosmology=Planck18)

    assert len(pdf) == expected_len

# tests for the func volumetric_rate()
def test_volumetric_rate_float():
    z = 0.1
    n_per_gpc3_float = 10.0
    volume = Planck18.comoving_volume(z).to("Gpc**3").value
    z_rate_expected = volume * n_per_gpc3_float

    z_rate_actual = get_volumetric_rate(z,n_per_gpc3_float)

    np.testing.assert_allclose(z_rate_actual, z_rate_expected)

    assert z_rate_actual.shape == ()

def test_volumetric_rate_array():
    z = 0.1
    n_per_gpc3_array = np.array([5.0, 10.0, 15.0])
    volume = Planck18.comoving_volume(z).to("Gpc**3").value
    z_rate_expected = volume * n_per_gpc3_array

    z_rate_actual = get_volumetric_rate(z, n_per_gpc3_array)

    np.testing.assert_allclose(z_rate_actual, z_rate_expected)

    assert z_rate_actual.shape == n_per_gpc3_array.shape



def test_nyears_in_target():
    """ """
    import skysurvey
    rate_test = 3e4
    nyears = 0.1
    snia = skysurvey.SNeIa.from_draw(nyears=nyears, zmin=0, zmax=0.145, rate=rate_test, cosmology=Planck18)

    volume = Planck18.comoving_volume(0.145).to("Gpc^3").value
    expected_size = volume * rate_test * nyears
    assert np.isclose(len(snia.data), expected_size, rtol=0.01)
