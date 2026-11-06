import numpy as np
import pytest
from astropy import constants, units as u
from astropy.coordinates import SkyCoord

from skysurvey.tools import velocities as vpec


def test_zcmb_to_zhel_toward_dipole():
    """Toward the dipole, cos(theta)=1, so zhel follows the Doppler formula directly."""
    icrs = SkyCoord(l=264.021*u.deg, b=48.253*u.deg, frame='galactic').icrs
    x = 369.82e3 / constants.c.value
    zsun = np.sqrt((1 - x) / (1 + x)) - 1
    expected = (1 + 0.1) * (1 + zsun) - 1
    assert vpec.zcmb_to_zhel(0.1, icrs.ra.deg, icrs.dec.deg) == pytest.approx(expected, rel=1e-8)


def test_roundtrip():
    """zhel_to_zcmb must undo zcmb_to_zhel."""
    zcmb = np.array([0.05, 0.5, 1.2])
    ra = np.array([10.0, 150.0, 300.0])
    dec = np.array([-30.0, 20.0, 60.0])
    zhel = vpec.zcmb_to_zhel(zcmb, ra, dec)
    np.testing.assert_allclose(vpec.zhel_to_zcmb(zhel, ra, dec), zcmb, atol=1e-12)

def test_magobs_to_magobshel_known_value():
    """A factor of 10 in (1+zhel)/(1+z) must shift the magnitude by exactly 5."""
    z = 0.0
    zhel = 9.0  # (1 + 9) / (1 + 0) = 10 -> 5*log10(10) = 5
    assert vpec.magobs_to_magobshel(20.0, z, zhel) == pytest.approx(25.0)


def test_roundtrip():
    """magobshel_to_magobs must undo magobs_to_magobshel."""
    magobs = np.array([18.5, 21.0, 24.3])
    z = np.array([0.05, 0.5, 1.2])
    zhel = np.array([0.051, 0.498, 1.203])
    magobshel = vpec.magobs_to_magobshel(magobs, z, zhel)
    np.testing.assert_allclose(vpec.magobshel_to_magobs(magobshel, z, zhel), magobs, atol=1e-12)