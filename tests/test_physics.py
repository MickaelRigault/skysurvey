""" First-principles tests of skysurvey.

These tests do not only check that the code runs, they check that what it
simulates is physically correct. Each expected value is derived independently
of skysurvey (astropy cosmology integrals, analytic K-corrections, Gaussian
statistics), so a failure means the simulation is wrong, not just different.

Statistical tests use fixed seeds and tolerances of 5 sigma (or p-values > 1e-3)
so they are deterministic and only fail for a real bias.
"""
import numpy as np
import pandas
import pytest
import sncosmo

from scipy import stats, integrate
from shapely import geometry
from astropy.cosmology import Planck18

import skysurvey
from skysurvey.target.rates import draw_redshift
from skysurvey.tools.utils import random_radec


SNIA_RATE = 2.35e4 # SNeIa._RATE in Gpc-3/yr, defined for H0=70
RATE_H0 = 70 # Transient._RATE_H0


def expected_ntargets_per_year(zmax, rate, zmin=0, cosmology=Planck18, rate_H0=RATE_H0):
    """ number of transients per observer-frame year in the full sky.

    N = rate x (H0/rate_H0)^3 x int_zmin^zmax dV/dz / (1+z) dz

    - dV/dz is the comoving volume element (full sky, Gpc^3)
    - 1/(1+z) is the cosmological time dilation: one year in the observer
      frame is 1/(1+z) year in the transient rest-frame.
    - (H0/rate_H0)^3 rescales a rate given in h70^3 Gpc^-3 to the cosmology.
    """
    def dn_dz(z):
        dvdz = cosmology.differential_comoving_volume(z).to("Gpc^3/sr").value * 4*np.pi
        return dvdz / (1+z)

    nper_rate = integrate.quad(dn_dz, zmin, zmax)[0]
    return rate * (cosmology.H0.value / rate_H0)**3 * nper_rate


def get_flatfnu_source(sigma_t=10.):
    """ SED flat in f_nu, with a gaussian time evolution of width sigma_t (days).

    For such a source, the observed AB magnitude is the same in any band and
    the K-correction is analytic: K(z) = -2.5 log10(1+z).
    """
    phase = np.linspace(-50, 100, 301) # rest-frame days
    wave = np.linspace(1000, 25_000, 500) # Angstrom
    flam = 2.998e18 / wave**2 # f_nu = 1 converted to f_lambda (arbitrary unit)
    lightcurve = np.exp(-0.5 * (phase/sigma_t)**2) # peak at phase 0
    return sncosmo.TimeSeriesSource(phase, wave, lightcurve[:,None] * flam[None,:],
                                    name="flatfnu")


def expected_flatfnu_mag(magabs, z, phase=0, sigma_t=10., cosmology=Planck18):
    """ observed AB magnitude of the flat f_nu source at the given rest-frame phase.

    m = M + mu(z) + K(z) - 2.5 log10( lightcurve(phase) )
      = M + mu(z) - 2.5 log10(1+z) + 2.5/ln(10) x phase^2 / (2 sigma_t^2)
    """
    distmod = cosmology.distmod(z).value
    kcorr = -2.5 * np.log10(1+z)
    decline = 2.5 / np.log(10) * 0.5 * (phase / sigma_t)**2
    return magabs + distmod + kcorr + decline


# ============== #
#  Fixtures      #
# ============== #
@pytest.fixture
def flatfnu_targets():
    """ three identical flat f_nu transients (M=-19) at increasing redshifts """
    data = pandas.DataFrame({"z": [0.05, 0.2, 0.5],
                             "t0": 59_000.,
                             "magabs": -19.,
                             "ra": 150., "dec": 0.})
    model = sncosmo.Model(get_flatfnu_source())
    return skysurvey.Transient.from_data(data, template=model)

@pytest.fixture
def survey_on_targets():
    """ 3000 observations in g, r, i all pointing at the targets location (ra=150, dec=0) """
    nobs = 3_000
    rng = np.random.default_rng(1)
    logs = {"mjd": rng.uniform(59_000 - 40, 59_000 + 60, size=nobs), # within the source phase range
            "band": rng.choice(["sdssg", "sdssr", "sdssi"], size=nobs),
            "ra": 150., "dec": 0.,
            "skynoise": 50., "gain": 2., "zp": 30.}
    footprint = geometry.Point(0, 0).buffer(1) # 1 deg radius camera
    return skysurvey.Survey.from_pointings(logs, footprint=footprint)


# ====================== #
#  Rates and volumes     #
# ====================== #
# tests for the number of targets: rate x volume x time
def test_snia_number_follows_rate_volume_and_time_dilation():
    nyears = 0.01
    zmax = 0.5
    data = skysurvey.SNeIa().draw(nyears=nyears, zmax=zmax, tstart=59_000)

    expected = expected_ntargets_per_year(zmax, SNIA_RATE) * nyears
    # 0.1% tolerance: this test targets the physics factors (time dilation is a
    # ~30% effect at z=0.5, the H0 rescaling ~10%). It absorbs:
    # - the int() rounding of the drawn size (1 target out of ~4800, 2e-4)
    # - the sky fraction, estimated by draw() from the convex hull of random
    #   (ra, sin dec) points (radecmodel_to_skysurface): ~1e-4 low for the full sky.
    assert np.isclose(len(data), expected, rtol=1e-3), f"{len(data)=} while {expected=:.1f}"
    # and the targets are spread over the nyears
    assert data["t0"].min() >= 59_000
    assert data["t0"].max() <= 59_000 + nyears * 365.25

def test_snia_number_low_redshift():
    # at low redshift N ~ zmax^3: missing the outermost shell would bias low
    # by 3 zstep/zmax (0.6% here, see get_ntargets_per_shell).
    nyears = 2
    zmax = 0.05
    data = skysurvey.SNeIa().draw(nyears=nyears, zmax=zmax, tstart=59_000)

    expected = expected_ntargets_per_year(zmax, SNIA_RATE) * nyears # ~1800 targets
    # 1 target out of 1800 is the int() rounding of the drawn size.
    assert np.isclose(len(data), expected, atol=1.), f"{len(data)=} while {expected=:.1f}"

@pytest.mark.parametrize("radec, skyfraction",
                         [({"dec_range": [-30, 90]}, (np.sin(np.radians(90)) - np.sin(np.radians(-30))) / 2),
                          ({"ra_range": [0, 90], "dec_range": [0, 90]}, 1/8),
                          ])
def test_snia_number_scales_with_sky_area(radec, skyfraction):
    # the area of a ra-dec box on the sphere is dra x (sin(dec_max) - sin(dec_min))
    nyears = 0.01
    zmax = 0.5
    snia = skysurvey.SNeIa()
    nfullsky = len(snia.draw(nyears=nyears, zmax=zmax, tstart=59_000))
    nbox = len(snia.draw(nyears=nyears, zmax=zmax, tstart=59_000, radec=radec))

    # 2 targets: int() rounding of both sizes.
    assert np.isclose(nbox, nfullsky * skyfraction, atol=2.), f"{nbox=} while {nfullsky*skyfraction=:.1f}"

# tests for the redshift distribution
def test_redshift_distribution_follows_volume_and_time_dilation():
    # for a constant rate, dN/dz is proportional to dV/dz / (1+z)
    size = 50_000
    redshifts = draw_redshift(size, rate=1e4, zmax=0.1, rng=4)

    edges = np.linspace(0, 0.1, 11)
    dn_dz = lambda z: Planck18.differential_comoving_volume(z).value / (1+z)
    weights = np.array([integrate.quad(dn_dz, zlow, zhigh)[0]
                        for zlow, zhigh in zip(edges[:-1], edges[1:])])
    expected_counts = weights / weights.sum() * size
    observed_counts, _ = np.histogram(redshifts, bins=edges)

    pvalue = stats.chisquare(observed_counts, expected_counts).pvalue
    assert pvalue > 1e-3, f"redshift histogram incompatible with dV/dz/(1+z) ({pvalue=})"


# ====================== #
#  Sky positions         #
# ====================== #
# tests for the func random_radec
def test_random_radec_is_isotropic():
    # isotropy on the sphere means ra and sin(dec) are uniform.
    ra, dec = random_radec(size=100_000, rng=5)

    pvalue_ra = stats.kstest(ra, stats.uniform(loc=0, scale=360).cdf).pvalue
    pvalue_dec = stats.kstest(np.sin(np.radians(dec)), stats.uniform(loc=-1, scale=2).cdf).pvalue
    assert pvalue_ra > 1e-3, "ra is not uniform in [0, 360]"
    assert pvalue_dec > 1e-3, "sin(dec) is not uniform in [-1, 1]: not isotropic"


# ====================== #
#  SN Ia standardisation #
# ====================== #
# tests for the SNeIa magnitude model (Tripp 1998)
def test_snia_tripp_relation_is_recovered():
    # the default model is magabs = -19.3 + alpha x1 + beta c + N(0, 0.10),
    # with alpha = -0.14 and beta = 3.15: a linear fit must recover them.
    size = 20_000
    data = skysurvey.SNeIa().draw(size, zmax=0.1, rng=3)

    design = np.vstack([np.ones(size), data["x1"], data["c"]]).T
    (mabs, alpha, beta), *_ = np.linalg.lstsq(design, data["magabs"], rcond=None)
    sigmaint = np.std(data["magabs"] - design @ [mabs, alpha, beta])

    # 1-sigma fit errors are sigmaint / (std(param) x sqrt(N)):
    # ~7e-4 for alpha (std(x1)~1), ~5e-3 for beta (std(c)~0.15). We use 5 sigma.
    assert np.isclose(alpha, -0.14, atol=5 * 0.10 / (data["x1"].std() * np.sqrt(size)))
    assert np.isclose(beta, 3.15, atol=5 * 0.10 / (data["c"].std() * np.sqrt(size)))
    assert np.isclose(mabs, -19.3, atol=5e-3)
    # std error on a std is sigma / sqrt(2N) ~ 5e-4
    assert np.isclose(sigmaint, 0.10, atol=5 * 0.10 / np.sqrt(2*size))


# ============================= #
#  Distances and K-corrections  #
# ============================= #
# tests for the target magnitudes
@pytest.mark.parametrize("band", ["sdssg", "sdssr", "sdssi"])
def test_target_mag_flatfnu_distance_and_kcorrection(flatfnu_targets, band):
    # m = M + mu(z) - 2.5 log10(1+z) at peak, whatever the band.
    for index, target_ in flatfnu_targets.data.iterrows():
        mag = flatfnu_targets.get_target_mag(index, band, phase=0., magsys="ab")
        expected = expected_flatfnu_mag(target_["magabs"], target_["z"])
        assert np.isclose(mag, expected, atol=1e-4), f"wrong peak mag at z={target_['z']} in {band}"

def test_target_flux_matches_target_mag(flatfnu_targets):
    # flux scaled to zp in the AB system is 10^(-0.4 (mag - zp))
    for index, target_ in flatfnu_targets.data.iterrows():
        flux = flatfnu_targets.get_target_flux(index, "sdssr", phase=0., zp=25, zpsys="ab")
        expected = 10**(-0.4 * (expected_flatfnu_mag(target_["magabs"], target_["z"]) - 25))
        assert np.isclose(flux, expected, rtol=1e-4)

def test_snia_amplitude_follows_inverse_square_law():
    # two identical SNe Ia (same M, x1, c) at z1 and z2: the template amplitude
    # x0 is a flux scale, it must follow 1/dL^2.
    data = pandas.DataFrame({"z": [0.05, 0.3], "x1": 0., "c": 0.,
                             "t0": 59_000., "magabs": -19.3,
                             "ra": 0., "dec": 0.})
    snia = skysurvey.SNeIa.from_data(data)
    x0 = [snia.get_target_template(index, as_model=True, set_magabs=True).get("x0")
          for index in data.index]

    dl = Planck18.luminosity_distance(data["z"]).value
    assert np.isclose(x0[0] / x0[1], (dl[1] / dl[0])**2, rtol=1e-5)


# ============================= #
#  Simulated lightcurves        #
# ============================= #
# tests for the func DataSet.from_targets_and_survey
def test_dataset_flux_distance_kcorrection_and_time_dilation(flatfnu_targets, survey_on_targets):
    # noiseless lightcurves: every observed point must follow
    # m(t) = M + mu(z) - 2.5 log10(1+z) + decline( (t - t0) / (1+z) )
    data = skysurvey.DataSet.from_targets_and_survey(flatfnu_targets, survey_on_targets,
                                                     incl_error=False).data
    assert len(data) == 3 * 3_000, "every target should be observed by every pointing"

    target_data = flatfnu_targets.data.loc[data.index.get_level_values("index")]
    z = target_data["z"].to_numpy()
    phase = (data["mjd"].to_numpy() - target_data["t0"].to_numpy()) / (1+z) # rest-frame
    expected = expected_flatfnu_mag(target_data["magabs"].to_numpy(), z, phase=phase)
    observed = -2.5 * np.log10(data["flux"]) + data["zp"]

    # 1e-3 mag: the template is linearly interpolated on a 0.5 day phase grid.
    np.testing.assert_allclose(observed, expected, atol=1e-3)

def test_dataset_noise_is_gaussian_with_sky_and_poisson_variance(flatfnu_targets, survey_on_targets):
    truth = skysurvey.DataSet.from_targets_and_survey(flatfnu_targets, survey_on_targets,
                                                      incl_error=False).data
    noisy = skysurvey.DataSet.from_targets_and_survey(flatfnu_targets, survey_on_targets,
                                                      incl_error=True, seed=2).data
    # variance = sky noise^2 + source Poisson noise (flux in ADU, gain in e-/ADU)
    expected_var = truth["skynoise"]**2 + truth["flux"] / truth["gain"]
    np.testing.assert_allclose(noisy["fluxerr"]**2, expected_var)

    # pulls must be N(0, 1)
    pull = (noisy["flux"] - truth["flux"]) / truth["fluxerr"]
    npoints = len(pull)
    assert np.abs(pull.mean()) < 5 / np.sqrt(npoints), "biased noise"
    assert np.abs(pull.std() - 1) < 5 / np.sqrt(2 * npoints), "wrong noise amplitude"
    assert stats.kstest(pull, stats.norm.cdf).pvalue > 1e-3, "noise is not gaussian"
