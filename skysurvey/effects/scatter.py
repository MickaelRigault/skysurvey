""" module associated to dust effects (host or Milky way) """
import numpy as np
import sncosmo

# ================== #
#                    #
# Scatter Models     #
#                    #
# ================== #
def sine_interp(x_new, fun_x, fun_y):
    """ 
    Sinus interpolation for intrinsic scattering models.

    Parameters
    ----------
    x_new: array
        new x values.

    fun_x: array
        x values of the function to interpolate.

    fun_y: array
        y values of the function to interpolate.

    Returns
    -------
    array
        interpolated values.
    """
    
    if len(fun_x) != len(fun_y):
        raise ValueError('x and y must have the same len')
    if (x_new > fun_x[-1]).any() or (x_new < fun_x[0]).any():
        raise ValueError('x_new is out of range of fun_x')

    sup_bound = np.vstack([x_new >= x for x in fun_x])

    idx_inf = np.sum(sup_bound, axis=0) - 1
    idx_inf[idx_inf==len(fun_x) - 1] = -2

    x_inf = fun_x[idx_inf]
    x_sup = fun_x[idx_inf + 1]
    fun_y_inf = fun_y[idx_inf]
    fun_y_sup = fun_y[idx_inf + 1]

    sin_interp = np.sin(np.pi * (x_new - 0.5 * (x_inf + x_sup)) / (x_sup - x_inf))
    values = 0.5 * (fun_y_sup + fun_y_inf) + 0.5 * (fun_y_sup - fun_y_inf) * sin_interp
    return values


class ColorScatter_G10( sncosmo.PropagationEffect ):
    """Guy (2010) SNe Ia non-coherent scattering.
    
    Implementation is done following arxiv:1209.2482.
    """

    _param_names = ['L0', 'F0', 'F1', 'dL']
    param_names_latex = [r'\lambda_0', 'F_0', 'F_1', 'd_L']

    def __init__(self, saltsource):
        """
        Initialize G10 class.

        Parameters
        ----------
        saltsource: sncosmo.Source
            salt source to use.
        """
        self._parameters = np.array([2157.3, 0.0, 1.08e-4, 800])
        self._colordisp = saltsource._colordisp
        self._minwave = saltsource.minwave()
        self._maxwave = saltsource.maxwave()

    @classmethod
    def from_saltsource(cls, name="salt2", version=None):
        """ 
        Shortcut to directly load the color scatter from the salt2 source

        Parameters
        ----------
        name: str
            name of the salt source.

        version: str
            version of the salt source.

        Returns
        -------
        ColorScatter_G10
        """
        saltource = sncosmo.get_source(name, version=version)
        return cls(saltource)
        

    def compute_sigma_nodes(self):
        """
        Computes the sigma nodes.

        Returns
        -------
        (array, array)
            lambda nodes, sigma values
        """
        L0, F0, F1, dL = self._parameters
        
        lam_nodes = np.arange(self._minwave, self._maxwave, dL)
        if lam_nodes.max() < self._maxwave:
            lam_nodes = np.append(lam_nodes, self._maxwave)
            
        siglam_values = self._colordisp(lam_nodes) 

        siglam_values[lam_nodes < L0] *= 1 + (lam_nodes[lam_nodes < L0] - L0) * F0
        siglam_values[lam_nodes > L0] *= 1 + (lam_nodes[lam_nodes > L0] - L0) * F1
        siglam_values *= np.random.normal(size=len(lam_nodes))

        return lam_nodes, siglam_values

    def propagate(self, wave, flux):
        """
        Propagate the effect to the flux.

        Parameters
        ----------
        wave: array
            wavelengths.

        flux: array
            fluxes.

        Returns
        -------
        array
            propagated fluxes.
        """
        lam_nodes, siglam_values = self.compute_sigma_nodes()
        magscat = sine_interp(wave, lam_nodes, siglam_values)
        return flux * 10**(-0.4 * magscat)


class ColorScatter_C11( sncosmo.PropagationEffect ):
    """ C11 scattering effect for sncosmo.
    
    Use covariance matrix between the vUBVRI bands from N. Chotard thesis.

    Implementation is done following arxiv:1209.2482.
    """

    _param_names = ["C_vU", 'S_f']
    param_names_latex = ["\rho_\mathrm{vU}", 'S_f']
    _minwave = 2000
    _maxwave = 11000

    def __init__(self):
        """
        Initialise C11 class.
        """
        self._parameters = np.array([0., 1.3])

        # vUBVRI lambda eff
        self._lam_nodes = np.array([2500.0, 3560.0, 4390.0, 5490.0, 6545.0, 8045.0])

        # vUBVRI correlation matrix extract from SNANA, came from N.Chotard thesis
        self._corr_matrix = np.array(
            [
                [+1.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000],
                [ 0.000000, +1.000000, -0.118516, -0.768635, -0.908202, -0.219447],
                [ 0.000000, -0.118516, +1.000000, +0.570333, -0.238470, -0.888611],
                [ 0.000000, -0.768635, +0.570333, +1.000000, +0.530320, -0.399538],
                [ 0.000000, -0.908202, -0.238470, +0.530320, +1.000000, +0.490134],
                [ 0.000000, -0.219447, -0.888611, -0.399538, +0.490134, +1.000000]
            ]
            ) 

        self._corr_matrix[0, 1:] = self._parameters[0] * self._corr_matrix[1, 1:]
        self._corr_matrix[1:, 0] = self._parameters[0] * self._corr_matrix[1:, 1]

        # vUBVRI sigma
        self._siglam_values = np.array([0.5900, 0.06001, 0.040034, 0.050014, 0.040017, 0.080007])

        # Convert corr to cov
        self._cov_matrix = self._corr_matrix * np.outer(self._siglam_values, 
                                                        self._siglam_values) 
        # Rescale covariance as in arXiv:1209.2482
        self._cov_matrix *= self._parameters[1]

    def propagate(self, wave, flux):
        """Propagate the effect to the flux."""
        
        siglam_values = np.random.multivariate_normal(np.zeros(len(self._lam_nodes)),
                                                      self._cov_matrix)

        inf_mask = wave <= self._lam_nodes[0]
        sup_mask = wave >= self._lam_nodes[-1]

        magscat = np.zeros(len(wave))
        magscat[inf_mask] = siglam_values[0]
        magscat[sup_mask] = siglam_values[-1]
        magscat[~inf_mask & ~sup_mask] = sine_interp(wave[~inf_mask & ~sup_mask],
                                                     self._lam_nodes,
                                                     siglam_values)

        return flux * 10**(-0.4 * magscat)
