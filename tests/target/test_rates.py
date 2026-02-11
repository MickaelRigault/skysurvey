import numpy as np
import pytest
from astropy.cosmology import Planck18
# from skysurvey.target.rates import get_rate
from skysurvey.target.rates import get_redshift_pdf
from skysurvey.target.rates import get_volumetric_rate


constant_rate = 1.0e4
z = np.array([0.1, 0.2, 0.3])


# tests for the func get_redshift_pdf
def evolving_rate(z, r0=2.3e4, alpha=1.70):
        return r0 * (1 + z)**alpha

@pytest.mark.parametrize("rate", [
    1.0e4,          # float case
    evolving_rate,  # callable case
])
def test_get_redshift_pdf_normalisation(rate):
    z = np.array([0.1, 0.2, 0.3])
    pdf = get_redshift_pdf(z=z, rate=rate, 
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

    pdf = get_redshift_pdf(z=z, rate=rate,
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


def test_nyears_and_radec():
    """ """
    import skysurvey
    snia_fullsky = skysurvey.SNeIa.from_draw(zmax=0.1, nyears=1., rate=1e3)
    snia_northsky = skysurvey.SNeIa.from_draw(zmax=0.1, nyears=1., rate=1e3, radec={"dec_range": [0, 90]})

    assert np.isclose( len(snia_northsky.data)/len(snia_fullsky.data), 0.5, rtol=0.01)
