import numpy as np
import pytest
from unittest.mock import patch
from astropy import units as u
from skysurvey.tools.blackbody import get_blackbody_transient_source, get_blackbody_transient_flux, blackbody_nu, blackbody_lambda, get_wein_lbdamax

# tests for the func get_blackbody_transient_source
def test_get_blackbody_transient_source_str():
    phase = np.array([1.0, 2.0])
    temperature = np.array([5000, 6000])
    amplitude = np.array([1.0, 2.0])
    lbda="1_000:10_000:100j"
    expected_lbda = np.r_[1000:10000:100j]
    fake_fluxes = np.ones((2,len(expected_lbda)))

    with patch("skysurvey.tools.blackbody.get_blackbody_transient_flux", return_value=fake_fluxes) as mock_flux, \
         patch("sncosmo.TimeSeriesSource") as mock_tss:
        
        bb_source = get_blackbody_transient_source(phase, temperature, amplitude, lbda=lbda)

        mock_flux.assert_called_once()
        called_lbda = mock_flux.call_args[0][0]
        np.testing.assert_allclose(called_lbda, expected_lbda)
        assert mock_flux.call_args.kwargs["temperature"] is temperature
        assert mock_flux.call_args.kwargs["amplitude"] is amplitude

        mock_tss.assert_called_once()
        _, kwargs = mock_tss.call_args
        assert kwargs["zero_before"] is True
        assert kwargs["name"] == "bb_transient" 
        np.testing.assert_allclose(kwargs["phase"], phase)
        np.testing.assert_allclose(kwargs["wave"], expected_lbda)
        np.testing.assert_allclose(kwargs["flux"], fake_fluxes)

        assert bb_source is mock_tss.return_value

def test_get_blackbody_transient_source_array():
    phase = np.array([1.0, 2.0])
    temperature = np.array([5000, 6000])
    amplitude = np.array([1.0, 2.0])
    lbda = np.array([4000, 5000])
    fake_fluxes = np.ones((2,2))

    with patch("skysurvey.tools.blackbody.get_blackbody_transient_flux", return_value=fake_fluxes) as mock_flux, \
         patch("sncosmo.TimeSeriesSource") as mock_tss:
        
        bb_source = get_blackbody_transient_source(phase, temperature, amplitude, lbda=lbda)

        mock_flux.assert_called_once_with(lbda, temperature=temperature, amplitude=amplitude)
        mock_tss.assert_called_once_with(phase=phase, wave=lbda, flux=fake_fluxes, zero_before=True, name="bb_transient")

        assert bb_source is mock_tss.return_value

# tests for the func get_blackbody_transient_flux
def test_get_blackbody_transient_flux_no_unit():
    lbda = 5000
    temperature = 5800
    amplitude = 2.0
    flux = get_blackbody_transient_flux(lbda, temperature, amplitude, normed=True)

    assert isinstance(flux, np.ndarray)
    assert not hasattr(flux, "unit")
    assert flux.shape == (1, 1)

def test_get_blackbody_transient_flux_with_unit():
    lbda = 5000*u.AA
    temperature = 5800*u.K
    amplitude = 2.0 * (u.erg / (u.cm**2 * u.s * u.Hz))
    flux = get_blackbody_transient_flux(lbda, temperature, amplitude, normed=True)

    assert hasattr(flux, "unit")
    assert flux.unit.is_equivalent(amplitude.unit)
    assert flux.shape == (1, 1)

def test_get_blackbody_transient_flux_array():
    lbda = np.array([4000, 5000, 6000])
    temperature = np.array([5000, 6000])  
    amplitude = np.array([1.0, 2.0])
    flux = get_blackbody_transient_flux(lbda, temperature, amplitude, normed=True)

    assert flux.shape == (len(temperature), len(lbda))

def test_get_blackbody_transient_scaling():
    lbda = np.array([4000, 5000, 6000])
    temperature = np.array([5000, 5000])  
    amplitude = np.array([1.0, 2.0])
    flux = get_blackbody_transient_flux(lbda, temperature, amplitude, normed=True)

    np.testing.assert_allclose(flux[1], 2.0 * flux[0])

# tests for the func blackbody_nu
def test_blackbody_nu_no_unit():
    freq = 1e14
    temperature = 5800
    flux = blackbody_nu(freq, temperature)

    assert hasattr(flux, "unit")
    assert flux.unit.is_equivalent(u.erg / (u.cm**2 * u.s * u.Hz * u.sr))

def test_blackbody_nu_with_unit():
    freq = 1e14*u.Hz
    temperature = 5800*u.K
    flux1 = blackbody_nu(freq, temperature)
    flux2 = blackbody_nu(1e14, temperature)

    assert np.isclose(flux1.to_value(flux1.unit), flux2.to_value(flux2.unit))

def test_blackbody_nu_array():
    freq = np.array([1e14, 2e14])
    temperature = 5800
    flux = blackbody_nu(freq, temperature)

    assert flux.shape == freq.shape
    assert hasattr(flux, "unit")

def test_blackbody_nu_wavelenght_into_freq():
    lbda = 5000*u.AA
    temperature = 5800*u.K
    flux = blackbody_nu(lbda, temperature)

    assert hasattr(flux, "unit")

def test_blackbody_nu_negative_freq():
    freq = np.array([-1.0, 1e14])
    temperature = 5800

    with pytest.warns(UserWarning, match="freq contains invalid values"):
        blackbody_nu(freq, temperature)

def test_blackbody_nu_zero_freq():
    freq = np.array([0, 1e14])
    temperature = 5800

    with pytest.warns(UserWarning, match="freq contains invalid values"):
        with np.errstate(divide="ignore", invalid="ignore"):
            blackbody_nu(freq, temperature)

# tests for the func blackbody_lambda
def test_blackbody_lambda_no_unit():
    lbda = 5000
    temperature = 5800
    flux = blackbody_lambda(lbda, temperature, normed=False)

    assert hasattr(flux, "unit")
    assert flux.unit.is_equivalent(u.erg / (u.cm**2 * u.s * u.AA * u.sr))

def test_blackbody_lamba_with_unit():
    lbda = 5000*u.AA
    temperature = 5800
    flux1 = blackbody_lambda(lbda, temperature, normed=False)
    flux2 = blackbody_lambda(5000, temperature, normed=False)

    assert np.isclose(flux1.to_value(flux1.unit), flux2.to_value(flux2.unit))
    
def test_blackbody_lambda_array():
    lbda = np.linspace(1000, 10000, 100)
    temperature = 5800
    flux = blackbody_lambda(lbda, temperature, normed=False)

    assert flux.shape == lbda.shape
    assert hasattr(flux, "unit")

def test_blackbody_lambda_normed():
    lbda = np.linspace(1000, 10000, 100)
    temperature = 5800
    flux = blackbody_lambda(lbda, temperature, normed=True)

    assert isinstance(flux, np.ndarray)
    assert np.max(flux) == pytest.approx(1.0, rel=1e-3)
    assert not hasattr(flux, "unit")

# tests for the func get_wein_lbdamax
def test_get_wein_lbdamax_temperature_no_unit():
    lbda = get_wein_lbdamax(5800)

    assert hasattr(lbda, "unit")
    assert lbda.unit.is_equivalent(u.Angstrom)

def test_get_wein_lbdamax_temperature_with_unit():
    lbda1 = get_wein_lbdamax(5800)
    lbda2 = get_wein_lbdamax(5800*u.Kelvin)

    assert np.isclose(lbda1.to_value(u.Angstrom), lbda2.to_value(u.Angstrom))