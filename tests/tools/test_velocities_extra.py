import numpy as np
import pytest
from astropy import constants, units as u
from astropy.coordinates import SkyCoord

from skysurvey.tools import velocities as vpec

# Planck CMB dipole direction used by the module (galactic l, b) and the sun velocity
DIPOLE_L, DIPOLE_B = 264.021, 48.253
VSUN = 369.82e3 # m/s

def galactic_to_radec(l, b):
    icrs = SkyCoord(l=l*u.deg, b=b*u.deg, frame="galactic").icrs
    return icrs.ra.deg, icrs.dec.deg

def doppler_zsun(cos_theta):
    """ relativistic doppler shift for a velocity VSUN*cos_theta toward the target """
    x = VSUN * cos_theta / constants.c.value
    return np.sqrt((1 - x) / (1 + x)) - 1


# tests for the func zcmb_to_zhel
def test_zcmb_to_zhel_perpendicular_to_dipole():
    """ 90 deg away from the dipole, the sun velocity has no line-of-sight component """
    ra, dec = galactic_to_radec(DIPOLE_L, DIPOLE_B - 90)
    assert vpec.zcmb_to_zhel(0.1, ra, dec) == pytest.approx(0.1, abs=1e-12)

def test_zcmb_to_zhel_toward_dipole_is_blueshifted():
    """ the sun moves toward the dipole: targets there look closer (zhel < zcmb) """
    ra, dec = galactic_to_radec(DIPOLE_L, DIPOLE_B)
    assert vpec.zcmb_to_zhel(0.1, ra, dec) < 0.1

def test_zcmb_to_zhel_anti_dipole():
    ra, dec = galactic_to_radec(DIPOLE_L + 180, -DIPOLE_B)
    expected = (1 + 0.1) * (1 + doppler_zsun(-1)) - 1
    zhel = vpec.zcmb_to_zhel(0.1, ra, dec)
    assert zhel == pytest.approx(expected, rel=1e-8)
    assert zhel > 0.1

def test_zcmb_to_zhel_zero_redshift():
    """ at zcmb=0, zhel is the sun doppler shift alone: ~ -v/c toward the dipole """
    ra, dec = galactic_to_radec(DIPOLE_L, DIPOLE_B)
    zhel = vpec.zcmb_to_zhel(0., ra, dec)
    assert zhel == pytest.approx(doppler_zsun(1), rel=1e-8)
    assert zhel == pytest.approx(-VSUN / constants.c.value, rel=1e-2) # first order

def test_zcmb_to_zhel_array_shape():
    zcmb = np.array([0.01, 0.1, 1.0])
    zhel = vpec.zcmb_to_zhel(zcmb, [0., 100., 200.], [-30., 0., 30.])
    assert np.shape(zhel) == (3,)


# tests for the func zhel_to_zcmb
def test_zhel_to_zcmb_toward_dipole():
    ra, dec = galactic_to_radec(DIPOLE_L, DIPOLE_B)
    expected = (1 + 0.1) / (1 + doppler_zsun(1)) - 1
    assert vpec.zhel_to_zcmb(0.1, ra, dec) == pytest.approx(expected, rel=1e-8)

def test_zhel_to_zcmb_perpendicular_to_dipole():
    ra, dec = galactic_to_radec(DIPOLE_L, DIPOLE_B - 90)
    assert vpec.zhel_to_zcmb(0.1, ra, dec) == pytest.approx(0.1, abs=1e-12)

def test_zhel_to_zcmb_roundtrip():
    """ zcmb_to_zhel must undo zhel_to_zcmb """
    zhel = np.array([0.02, 0.3, 0.9])
    ra = np.array([45., 180., 330.])
    dec = np.array([10., -50., 75.])
    zcmb = vpec.zhel_to_zcmb(zhel, ra, dec)
    np.testing.assert_allclose(vpec.zcmb_to_zhel(zcmb, ra, dec), zhel, atol=1e-12)


# tests for the func magobs_to_magobshel
def test_magobs_to_magobshel_same_redshift():
    assert vpec.magobs_to_magobshel(20., 0.1, 0.1) == pytest.approx(20.)


# tests for the func magobshel_to_magobs
def test_magobshel_to_magobs_known_value():
    # (1 + 9) / (1 + 0) = 10 -> 5*log10(10) = 5 removed
    assert vpec.magobshel_to_magobs(25., 0., 9.) == pytest.approx(20.)

def test_magobshel_to_magobs_same_redshift():
    assert vpec.magobshel_to_magobs(20., 0.1, 0.1) == pytest.approx(20.)
