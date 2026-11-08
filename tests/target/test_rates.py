import numpy as np

from astropy.cosmology import Planck18
from skysurvey.target.rates import get_ntargets_per_shell, get_rate, draw_redshift, get_ntargets


constant_rate = 1.0e4
z = np.array([0.1, 0.2, 0.3])


def expected_ntargets(zmax, rate, zmin=0., cosmology=Planck18, rate_H0=70, npoints=200_000):
    """ independent expectation: N = rate x (H0/rate_H0)^3 x int dV / (1+z)

    the 1/(1+z) is the time dilation (observer-frame rate) and (H0/rate_H0)^3 the
    rescaling of a rate given for rate_H0 to the cosmology.
    """
    redshifts = np.linspace(zmin, zmax, npoints + 1)
    volume = cosmology.comoving_volume(redshifts).to("Gpc^3").value
    zmid = 0.5 * (redshifts[1:] + redshifts[:-1])
    return rate * (cosmology.H0.value / rate_H0) ** 3 * np.sum(np.diff(volume) / (1 + zmid))


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
    # 0.145 / 1e-4 = 1450 shells covering [0, 0.145] exactly
    assert zbins_float.shape == (1450,)
    assert pdf_array.shape == (rate_array.shape[0], 1450)
    assert (pdf_float == pdf_array[0]).all()

def test_get_ntargets_constantrate():
    """ """
    # this tests get_ntargets for constante rate, either array or float, using the shell or not.

    redshift_volume_1Gyr= 0.145031 # for Planck18 (comoving volume of 1 Gpc3)
    rate_float = 1e4

    # float array
    # (not rate_float: the count includes the time dilation and the H0 rescaling, see expected_ntargets)
    ntargets_constant = get_ntargets(redshift_volume_1Gyr, rate=rate_float, cosmology=Planck18, astype="int")
    assert np.isclose(ntargets_constant, expected_ntargets(redshift_volume_1Gyr, rate_float), rtol=1e-3)
    # => use of shell or not.
    ntargets_constant_shell = get_ntargets(redshift_volume_1Gyr, rate=rate_float, cosmology=Planck18, astype="int", force_shell=True)
    assert np.isclose(ntargets_constant_shell, ntargets_constant)

    # list of array
    rates_float_array = np.asarray([rate_float, 1e5])
    ntargets_constant_array = get_ntargets(redshift_volume_1Gyr, rate=rates_float_array, cosmology=Planck18, astype="int")

    assert ntargets_constant_array.shape == rates_float_array.shape
    assert np.isclose(ntargets_constant, ntargets_constant_array[0]) # isclose to avoid rounding error
    assert np.isclose(ntargets_constant_array[1], expected_ntargets(redshift_volume_1Gyr, rates_float_array[1]),
                      rtol=1e-3)

    # => use of shell
    ntargets_constant_array_shell = get_ntargets(redshift_volume_1Gyr, rate=rates_float_array, cosmology=Planck18, astype="int",
                                            force_shell=True)
    assert ntargets_constant_array_shell.shape == rates_float_array.shape
    assert np.isclose(ntargets_constant_array_shell, ntargets_constant_array, rtol=1e-3).all() # isclose to avoid rounding error

def test_get_ntargets_multirates():
    """ """
    # test cases with several rates (array)
    for ztest in [0.1, 0.2, 0.8]:
        # astype="float": compare the counts, not their integer truncation
        ntargets_1, ntargets_2 = get_ntargets(ztest, rate=np.asarray([1e3, 2e4]), astype="float")
        assert np.isclose(ntargets_1, expected_ntargets(ztest, 1e3), rtol=1e-3)
        assert np.isclose(ntargets_2, expected_ntargets(ztest, 2e4), rtol=1e-3)


def test_get_ntargets_rate_list():
    """ several rates given as a list: same as an array """
    ntargets_list = get_ntargets(0.2, rate=[1e3, 2e4])
    ntargets_array = get_ntargets(0.2, rate=np.asarray([1e3, 2e4]))
    assert np.allclose(ntargets_list, ntargets_array)


def test_get_ntargets_volumeconsistancy():

    # assuming Planck18, z=0.145 => 1Gpc3 ; z=0.18457 => 2Gpc3
    # (counts are not exactly proportional to the volume: time dilation, see expected_ntargets)

    rate_test = 1e3
    prop_test = dict( rate=rate_test, cosmology=Planck18, astype="float")

    for force_shell_ in [True, False]:
        prop_test["force_shell"] = force_shell_
        ntarget_vol1 = get_ntargets(zmax=0.145, **prop_test)
        ntarget_vol2 = get_ntargets(zmax=0.18457, **prop_test)
        ntarget_vol_21 = get_ntargets(zmax=0.18457, zmin=0.145, **prop_test)

        expected_1 = expected_ntargets(0.145, rate_test)
        expected_2 = expected_ntargets(0.18457, rate_test)
        expected_21 = expected_ntargets(0.18457, rate_test, zmin=0.145)
        assert np.isclose(ntarget_vol1/ntarget_vol2, expected_1/expected_2, rtol=1e-3)
        assert np.isclose(ntarget_vol1/ntarget_vol_21, expected_1/expected_21, rtol=1e-3)


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
    ntargets_1, ntargets_2 = draw_redshift(10_000, rate=[1, 2], rng = 2)
    kstest = ks_2samp(ntargets_1, ntargets_2)
    assert kstest.pvalue>0.1

# tests for the func get_rate()
# (the returned rate includes the 1/(1+z) frame change: observer-frame rate)
def test_volumetric_rate_float():
   z = 0.1
   n_per_gpc3_float = 10.0
   n_per_gpc3_returned = get_rate(z, n_per_gpc3_float, H0=70, rate_H0=70)
   assert np.isclose(n_per_gpc3_returned, n_per_gpc3_float / (1 + z))

def test_volumetric_rate_array():
   # several rates: one row per rate, one column per redshift
   n_per_gpc3_array = np.asarray([10.0, 20.0, 30.0])
   n_per_gpc3_returned = get_rate(z, n_per_gpc3_array, H0=70, rate_H0=70)
   assert n_per_gpc3_returned.shape == (len(n_per_gpc3_array), len(z))
   assert np.allclose(n_per_gpc3_returned, n_per_gpc3_array[:, None] / (1 + z))

def test_volumetric_rate_func():
    """ """
    redshifts = np.arange(0, 0.5, step=1e-3)
    constante = 3

    n_per_gpc3_returned = get_rate(redshifts, constante_rate, c=constante,
                                    H0=70, rate_H0=70)
    assert np.allclose(n_per_gpc3_returned, constante / (1 + redshifts))

def test_volumetric_rate_float_list_array():
    """ a rate can be a float, or several rates as a list or an array """
    rate_float = get_rate(z, 10., H0=70, rate_H0=70)
    rate_list = get_rate(z, [10., 20.], H0=70, rate_H0=70)
    rate_array = get_rate(z, np.asarray([10., 20.]), H0=70, rate_H0=70)
    assert rate_list.shape == rate_array.shape == (2, len(z))
    assert np.allclose(rate_list, rate_array)
    assert np.allclose(rate_list[0], rate_float)

    # same for the number of targets and the redshift draws
    ntargets_list = get_ntargets(0.2, rate=[10., 20.], astype="float")
    assert ntargets_list.shape == (2,)
    assert np.isclose(ntargets_list[0], get_ntargets(0.2, rate=10., astype="float"))
    redshifts = draw_redshift(100, rate=[10., 20.], zmax=0.2, rng=1)
    assert len(redshifts) == 2 and all(len(redshifts_) == 100 for redshifts_ in redshifts)

def test_volumetric_rate_h0():
    """ a rate given for rate_H0 is rescaled by (H0 / rate_H0)^3 (volume effect) """
    rate_70 = get_rate(0.1, 10., H0=70, rate_H0=70)
    rate_60 = get_rate(0.1, 10., H0=60, rate_H0=70)
    assert np.isclose(rate_60 / rate_70, (60 / 70) ** 3)




def test_nyears_in_target():
    """ """
    import skysurvey
    rate_test = 3e4
    nyears = 0.1
    snia = skysurvey.SNeIa.from_draw(nyears=nyears, zmin=0, zmax=0.145, rate=rate_test, cosmology=Planck18)

    # SNeIa rates are given for H0=70 (Target._RATE_H0)
    expected_size = expected_ntargets(0.145, rate_test) * nyears
    assert np.isclose(len(snia.data), expected_size, rtol=0.01)


def test_nyears_and_radec():
    """ """
    import skysurvey
    snia_fullsky = skysurvey.SNeIa.from_draw(zmax=0.1, nyears=1., rate=1e3)
    snia_northsky = skysurvey.SNeIa.from_draw(zmax=0.1, nyears=1., rate=1e3, radec={"dec_range": [0, 90]})

    assert np.isclose( len(snia_northsky.data)/len(snia_fullsky.data), 0.5, rtol=0.01)


# tests for the shells of get_ntargets_per_shell
def test_get_ntargets_per_shell_covers_zmax():
    # the outermost shell [zmax-zstep, zmax] must be included
    zbins, _ = get_ntargets_per_shell(0.05, rate=1e4, zmin=0, zstep=1e-4)
    assert len(zbins) == 500
    assert np.isclose(zbins[0], 0.5e-4), "first shell is [0, 1e-4]"
    assert np.isclose(zbins[-1], 0.05 - 0.5e-4), "last shell is [0.0499, 0.05]"

def test_get_ntargets_per_shell_zmax_not_multiple_of_zstep():
    # 0.12345 / 1e-4 = 1234.5 -> 1235 shells, slightly thinner than zstep, ending at zmax
    zbins, _ = get_ntargets_per_shell(0.12345, rate=1e4, zmin=0, zstep=1e-4)
    zstep_ = 0.12345 / 1235
    assert len(zbins) == 1235
    assert np.isclose(zbins[-1], 0.12345 - zstep_ / 2)

def test_get_ntargets_per_shell_float_noise():
    # (0.035 - 0.008) / 0.001 = 27.000000000000004 in float: still 27 shells
    zbins, _ = get_ntargets_per_shell(0.035, rate=1e4, zmin=0.008, zstep=1e-3)
    assert len(zbins) == 27
    assert np.allclose(np.diff(zbins), 1e-3)

def test_get_ntargets_volume_matches_comoving_volume():
    # constant rate, H0=rate_H0, no time dilation removed: sum of shells x (1+z)
    # is the comoving volume between zmin and zmax.
    zbins, ntargets = get_ntargets_per_shell(0.1, rate=1., zmin=0.02, zstep=1e-4,
                                             rate_H0=Planck18.H0.value, astype="float")
    volume = (Planck18.comoving_volume(0.1) - Planck18.comoving_volume(0.02)).to("Gpc^3").value
    assert np.isclose(np.sum(ntargets * (1 + zbins)), volume, rtol=1e-6)
