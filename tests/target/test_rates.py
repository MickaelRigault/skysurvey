import numpy as np

from astropy.cosmology import Planck18
from skysurvey.target.rates import get_ntargets_per_shell, get_rate, draw_redshift, get_ntargets


constant_rate = 1.0e4
z = np.array([0.1, 0.2, 0.3])


# tests for the func get_redshift_pdf
def evolving_rate(z, r0=2.3e4, alpha=1.70):
    return r0 * (1 + z)**alpha

def constante_rate(z, c=1e3):
    return c*np.ones(z.shape)



def test_get_ntargets_per_shell():
    """ """
    # test that it works as well as the returned format.
    rate_array = np.asarray([3, 4])
    zbins_float, pdf_float = get_ntargets_per_shell(0.145, zmin=0, zstep=1e-4, rate=rate_array[0])
    zbins_array, pdf_array = get_ntargets_per_shell(0.145, zmin=0, zstep=1e-4, rate=rate_array)

    assert (zbins_float == zbins_array).all()
    assert zbins_float.shape == (1449,)
    assert pdf_array.shape == (rate_array.shape[0], 1449)
    assert (pdf_float == pdf_array[0]).all()

def test_get_ntargets_constantrate():
    """ """
    # this tests get_ntargets for constante rate, either array or float, using the shell or not.
    
    redshift_volume_1Gyr= 0.145031 # for Planck18
    rate_float = 1e4

    # float array
    ntargets_constant = get_ntargets(redshift_volume_1Gyr, rate=rate_float, cosmology=Planck18, astype="int")
    assert np.isclose(ntargets_constant, rate_float, rtol=1e-3)
    # => use of shell or not.
    ntargets_constant_shell = get_ntargets(redshift_volume_1Gyr, rate=rate_float, cosmology=Planck18, astype="int", force_shell=True)
    assert np.isclose(ntargets_constant_shell, ntargets_constant)
    
    # list of array
    rates_float_array = np.asarray([rate_float, 1e5])
    ntargets_constant_array = get_ntargets(redshift_volume_1Gyr, rate=rates_float_array, cosmology=Planck18, astype="int")

    assert ntargets_constant_array.shape == rates_float_array.shape
    assert np.isclose(ntargets_constant, ntargets_constant_array[0]) # isclose to avoid rounding error
    assert np.isclose(ntargets_constant_array[1], rates_float_array[1], rtol=1e-3)

    # => use of shell
    ntargets_constant_array_shell = get_ntargets(redshift_volume_1Gyr, rate=rates_float_array, cosmology=Planck18, astype="int",
                                            force_shell=True)
    assert ntargets_constant_array_shell.shape == rates_float_array.shape
    assert np.isclose(ntargets_constant_array_shell, ntargets_constant_array, rtol=1e-3).all() # isclose to avoid rounding error

def test_get_ntargets_multirates():
    """ """
    # test cases with rate is list
    for ztest in [0.1, 0.2, 0.8]:
        vol_universe = Planck18.comoving_volume(ztest).to("Gpc^3").value
        ntargets_1, ntargets_2 = get_ntargets(ztest, rate=[1e3, 2e4])
        assert np.isclose(ntargets_1, 1e3*vol_universe, rtol=1e-3)
        assert np.isclose(ntargets_2, 2e4*vol_universe, rtol=1e-3)
    

def test_get_ntargets_volumeconsistancy():

    # assuming Planck18, z=0.145 => 1Gpc3 ; z=0.18457 => 2Gpc3

    rate_test = 1e3
    prop_test = dict( rate=rate_test, cosmology=Planck18, astype="float")

    for force_shell_ in [True, False]:
        prop_test["force_shell"] = force_shell_
        ntarget_vol1 = get_ntargets(zmax=0.145, **prop_test)
        ntarget_vol2 = get_ntargets(zmax=0.18457, **prop_test)
        ntarget_vol_21 = get_ntargets(zmax=0.18457, zmin=0.145, **prop_test)

        assert np.isclose(ntarget_vol1/ntarget_vol2, 0.5, rtol=0.01)
        assert np.isclose(ntarget_vol1/ntarget_vol_21, 1, rtol=0.01)

    
def test_draw_redshift_constantrate():
    """ """
    # ks test specify if two sample are drawn 
    # from the same underlying population.
    from scipy.stats import ks_2samp

    rate_float = 1e3
    redshifts_float = draw_redshift(int(1e4), rate=rate_float, zmax=0.145, 
                                    rng=1)
    # vary sample size and way rates are defined (constant but specified as constant per shell)
    redshifts_funcflat = draw_redshift(int(1e5), rate=constante_rate, zmax=0.145, rng=2,
                                           c=rate_float)
    
    kstest = ks_2samp(redshifts_float, redshifts_funcflat)
    assert kstest.pvalue>0.1

def test_draw_redshift_ndim2():
    """ """
    # ks test specify if two sample are drawn 
    # from the same underlying population.
    from scipy.stats import ks_2samp

    # constant rate should provide the same thing as the number of target is given. 
    # hence, as long as the rate(z) is the same at a constant, this cancels out.
    ntargets_1, ntargets_2 = draw_redshift(10_000, rate=[1, 2])
    kstest = ks_2samp(ntargets_1, ntargets_2)
    assert kstest.pvalue>0.1
    
# tests for the func volumetric_rate()
def test_volumetric_rate_float():
   z = 0.1
   n_per_gpc3_float = 10.0
   n_per_gpc3_returned = get_rate(z, n_per_gpc3_float)
   assert n_per_gpc3_returned == n_per_gpc3_float

def test_volumetric_rate_array():
   n_per_gpc3_array = np.asarray([10.0, 20.0, 30.0])
   n_per_gpc3_returned = get_rate(z, n_per_gpc3_array)
   assert np.all([n_per_gpc3_returned == n_per_gpc3_array])
   
def test_volumetric_rate_func():
    """ """
    redshifts = np.arange(0, 0.5, step=1e-3)
    constante = 3

    n_per_gpc3_returned = get_rate(redshifts, constante_rate, c=constante)
    assert np.all(n_per_gpc3_returned == constante)




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
