import numpy as np
import pytest
from skysurvey.source.angular import AngularTimeSeriesSource
from scipy.interpolate import RectBivariateSpline

@pytest.fixture
def angular():
    phase = np.array([0.0, 10.0])
    wave = np.array([3000.0, 4000.0])
    cos_theta = np.array([-1.0, 1.0])
    flux = np.ones((2, 2, 2))

    return AngularTimeSeriesSource(phase=phase, wave=wave, cos_theta=cos_theta, flux=flux, zero_before=False,
        zero_after=False, name=None, version=None)

# tests for the class AngularTimeSeriesSource
def test_AngularTimeSeriesSource_initialization():
    phase = np.array([0.0, 10.0])
    wave = np.array([3000.0, 4000.0])
    cos_theta = np.array([-1.0, 1.0])
    flux = np.ones((2, 2, 2))

    angular = AngularTimeSeriesSource(phase=phase, wave=wave, cos_theta=cos_theta, flux=flux,
                     zero_before=False, zero_after=False, name="test_model",
                     version=1.0)

    assert angular.name == 'test_model'
    assert angular.version == 1.0
    assert angular._phase is phase
    assert angular._wave is wave
    assert angular._cos_theta is cos_theta
    assert angular._flux_array is flux
    assert angular._current_theta == 0.
    assert angular._zero_before is False
    assert angular._zero_after is False
    assert len(angular._parameters) == 2
    assert angular._param_names == ['amplitude', 'theta']

# tests for the func _set_theta
def test_set_theta(angular):
    assert isinstance(angular._model_flux, RectBivariateSpline)
    assert angular._current_theta == angular._parameters[1]

    phase = angular._phase
    wave = angular._wave
    model_flux = angular._model_flux(phase, wave)

    assert model_flux.shape == (len(phase), len(wave))

# tests for the func _flux
def test_flux_shape(angular):
    phase = angular._phase
    wave = angular._wave
    f = angular._flux(phase, wave)

    assert f.shape == (len(phase), len(wave))

def test_flux_amplitude_scaling(angular):
    phase = angular._phase
    wave = angular._wave

    f1 = angular._flux(phase, wave)
    angular._parameters[0] = 2.0
    f2 = angular._flux(phase, wave)

    assert np.allclose(2.0 * f1, f2)

def test_flux_zero_before():
    phase = np.array([-10.0, 0.0, 10.0])
    wave = np.array([3000.0, 4000.0])
    cos_theta = np.array([-1.0, 1.0])

    angular = AngularTimeSeriesSource(phase=np.array([0.0, 10.0]), wave=wave, cos_theta=cos_theta, flux=np.ones((2,2,2)),
                     zero_before=True, zero_after=False, name=None, version=None)
    
    f = angular._flux(phase, wave)
    assert np.all(f[0] == 0.0)
    assert np.all(f[1:] != 0.0)

def test_flux_zero_after():
    phase = np.array([0.0, 10.0, 20.0])
    wave = np.array([3000.0, 4000.0])
    cos_theta = np.array([-1.0, 1.0])

    angular = AngularTimeSeriesSource(phase=np.array([0.0, 10.0]), wave=wave, cos_theta=cos_theta, flux=np.ones((2,2,2)),
                     zero_before=False, zero_after=True, name=None, version=None)
    
    f = angular._flux(phase, wave)

    assert np.all(f[-1] == 0.0)
    assert np.all(f[:-1] != 0.0)

def test_flux_current_theta(angular):
    phase = angular._phase
    wave = angular._wave
    old_model_flux = angular._model_flux

    angular._parameters[1] = 45.0
    angular._flux(phase, wave)

    assert angular._current_theta == 45.0
    assert angular._model_flux is not old_model_flux