import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from skysurvey.effects.scatter import sine_interp
from skysurvey.effects.scatter import ColorScatter_G10, ColorScatter_C11 

# tests for the func sine_interp
@pytest.mark.parametrize("x_new, fun_x, fun_y",
    [
        (np.array([1.5]), np.array([1, 2, 3]), np.array([10 ,20])),
        (np.array([0.5]), np.array([1, 2, 3]), np.array([10, 20, 30])),
        (np.array([3.5]), np.array([1, 2, 3]), np.array([10, 20, 30])),
    ]
)
def test_sine_interp_value_error(x_new, fun_x, fun_y):
    with pytest.raises(ValueError):
        sine_interp(x_new, fun_x, fun_y)

def test_sine_interp():
    fun_x = np.array([1, 2, 3])
    fun_y = np.array([10, 20, 30])

    assert np.isclose(sine_interp(np.array([1]), fun_x, fun_y), 10)
    assert np.isclose(sine_interp(np.array([1.5]), fun_x, fun_y), 15)
    assert np.isclose(sine_interp(np.array([2]), fun_x, fun_y), 20)

    x_new = np.array([1.1, 1.2, 1.3, 1.4])
    values = sine_interp(x_new, fun_x, fun_y)
    
    assert values.shape == x_new.shape

# tests for the class ColorScatter_G10
@pytest.fixture
def fake_saltsource():
    fake_source = MagicMock()
    fake_source._colordisp = lambda x: np.ones_like(x) * 0.01
    fake_source.minwave.return_value = 3000
    fake_source.maxwave.return_value = 7000
    return fake_source

@pytest.fixture
def g10(fake_saltsource):
    return ColorScatter_G10(fake_saltsource)

def test_colorscatter_G10_initialization(g10):
    assert g10._minwave == 3000
    assert g10._maxwave == 7000
    assert callable(g10._colordisp)
    assert len(g10._parameters) == 4
    assert g10._param_names == ['L0', 'F0', 'F1', 'dL']

# tests for the func from_saltsource
def test_from_saltsource(fake_saltsource):

    with patch("sncosmo.get_source", return_value=fake_saltsource) as mock_get:
        g10 = ColorScatter_G10.from_saltsource(name="salt2", version=None)
    
    mock_get.assert_called_once_with("salt2", version=None)
    assert isinstance(g10, ColorScatter_G10)

# tests for the func compute_sigma_nodes  
def test_compute_sigma_nodes_output(g10):
    lam_nodes, siglam_values = g10.compute_sigma_nodes(rng=24)

    assert lam_nodes.shape == siglam_values.shape
    assert lam_nodes.min() >= g10._minwave
    assert lam_nodes.max() <= g10._maxwave
    assert np.all(np.diff(lam_nodes) > 0)

def test_compute_sigma_nodes_same_seed(g10):
    lam_1, siglam_1 = g10.compute_sigma_nodes(rng=24)
    lam_2, siglam_2 = g10.compute_sigma_nodes(rng=24)

    np.testing.assert_allclose(lam_1, lam_2)
    np.testing.assert_allclose(siglam_1, siglam_2)

def test_compute_sigma_nodes_different_seed(g10):
    lam_1, siglam_1 = g10.compute_sigma_nodes(rng=24)
    lam_2, siglam_2 = g10.compute_sigma_nodes(rng=33)

    assert not np.allclose(siglam_1, siglam_2)

def test_compute_sigma_nodes_edgecase(g10):
    lam_nodes_max = np.array([3000.0, 5000.0, 7000.0])
    
    with patch("skysurvey.effects.scatter.np.arange", return_value=lam_nodes_max):
        lam_nodes, siglam_values = g10.compute_sigma_nodes(rng=42)

        assert len(lam_nodes) == len(lam_nodes_max)
        assert lam_nodes[-1] == 7000

# tests for the func propagate
def test_propagate_G10(g10):
    wave = np.array([3000.0, 4000.0, 5000.0])
    flux = np.array([100.0, 200.0, 300.0])

    new_flux = g10.propagate(wave, flux)

    assert new_flux.shape == flux.shape

# tests for the class ColorScatter_C11
@pytest.fixture
def c11():
    return ColorScatter_C11()

def test_colorscatter_C11_initialization_param(c11):
    assert c11._minwave == 2000
    assert c11._maxwave == 11000
    assert len(c11._parameters) == 2
    assert c11._param_names == ["C_vU", 'S_f']

def test_colorscatter_C11_initialization_shape(c11):
    assert c11._lam_nodes.ndim == 1
    assert c11._siglam_values.shape == (len(c11._lam_nodes),)
    assert c11._corr_matrix.shape == (len(c11._lam_nodes), len(c11._lam_nodes))
    assert c11._cov_matrix.shape == c11._corr_matrix.shape

def test_colorscatter_C11_initialization_corr_matrix_symmetric(c11):
    np.testing.assert_allclose(c11._cov_matrix, c11._cov_matrix.T)

# tests for the func propagate
def test_propagate_C11(c11):
    wave = np.linspace(2000, 10000, 1000)
    flux = np.linspace(200, 1000, 1000) 

    new_flux = c11.propagate(wave, flux)

    assert new_flux.shape == flux.shape