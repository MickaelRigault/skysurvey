import numpy as np
import pandas
import pytest
import shapely
from shapely import geometry

from skysurvey.tools import utils


# tests for the func get_skynoise_from_maglimit
def test_get_skynoise_from_maglimit_at_zp():
    # maglim == zp => 5-sigma flux is 1 => noise is 1/5
    assert utils.get_skynoise_from_maglimit(30, zp=30) == pytest.approx(0.2)

def test_get_skynoise_from_maglimit_brighter_by_2p5():
    # 2.5 mag brighter => flux x10 => noise 10/5
    assert utils.get_skynoise_from_maglimit(22.5, zp=25) == pytest.approx(2.0)

def test_get_skynoise_from_maglimit_array():
    maglim = np.array([30., 27.5, 25.])
    np.testing.assert_allclose(utils.get_skynoise_from_maglimit(maglim, zp=30), [0.2, 2.0, 20.0])


# tests for the func build_covariance
def test_build_covariance_diagonal():
    cov, names = utils.build_covariance(x1=0.1, c=0.05)

    assert list(names) == ["x1", "c"]
    np.testing.assert_allclose(cov, [[0.01, 0.], [0., 0.0025]])

def test_build_covariance_offdiagonal():
    cov, names = utils.build_covariance(x1=0.1, c=0.05, cov_x1c=0.001)
    np.testing.assert_allclose(cov, [[0.01, 0.001], [0.001, 0.0025]])

def test_build_covariance_offdiagonal_reversed_key():
    """ cov_{key2}{key1} is looked for as well as cov_{key1}{key2} """
    cov, names = utils.build_covariance(x1=0.1, c=0.05, cov_cx1=0.001)
    np.testing.assert_allclose(cov, [[0.01, 0.001], [0.001, 0.0025]])

def test_build_covariance_as_dataframe():
    cov, names = utils.build_covariance(as_dataframe=True, x1=0.1, c=0.05, cov_x1c=0.001)

    assert isinstance(cov, pandas.DataFrame)
    assert list(cov.index) == ["x1", "c"]
    assert list(cov.columns) == ["x1", "c"]
    assert cov.loc["c", "x1"] == pytest.approx(0.001)
    assert cov.loc["c", "c"] == pytest.approx(0.0025)


# tests for the func apply_gaussian_noise
class FakeTarget:
    """ minimal target: has data and the from_data classmethod """
    def __init__(self, data):
        self.data = data

    @classmethod
    def from_data(cls, data):
        return cls(data)

@pytest.fixture
def noise_target():
    size = 20_000
    data = pandas.DataFrame({"x1": np.zeros(size), "c": np.full(size, 0.1), "z": np.full(size, 0.05)})
    return FakeTarget(data)

def test_apply_gaussian_noise_returns_target(noise_target):
    noisy = utils.apply_gaussian_noise(noise_target, seed=1, x1=0.1, c=0.05)
    assert isinstance(noisy, FakeTarget)
    assert len(noisy.data) == len(noise_target.data)

def test_apply_gaussian_noise_truth_and_err_columns(noise_target):
    noisy = utils.apply_gaussian_noise(noise_target, seed=1, x1=0.1, c=0.05)

    np.testing.assert_array_equal(noisy.data["x1_true"], noise_target.data["x1"])
    np.testing.assert_array_equal(noisy.data["c_true"], noise_target.data["c"])
    assert np.all(noisy.data["x1_err"] == 0.1)
    assert np.all(noisy.data["c_err"] == 0.05)

def test_apply_gaussian_noise_untouched_columns(noise_target):
    noisy = utils.apply_gaussian_noise(noise_target, seed=1, x1=0.1, c=0.05)
    np.testing.assert_array_equal(noisy.data["z"], noise_target.data["z"])

def test_apply_gaussian_noise_input_not_modified(noise_target):
    _ = utils.apply_gaussian_noise(noise_target, seed=1, x1=0.1, c=0.05)
    assert np.all(noise_target.data["x1"] == 0)
    assert "x1_true" not in noise_target.data

def test_apply_gaussian_noise_seed_reproducible(noise_target):
    noisy1 = utils.apply_gaussian_noise(noise_target, seed=12, x1=0.1)
    noisy2 = utils.apply_gaussian_noise(noise_target, seed=12, x1=0.1)
    np.testing.assert_array_equal(noisy1.data["x1"], noisy2.data["x1"])

def test_apply_gaussian_noise_statistics(noise_target):
    noisy = utils.apply_gaussian_noise(noise_target, seed=1, x1=0.1, c=0.05, cov_x1c=0.003)
    n = len(noisy.data)
    dx1 = noisy.data["x1"] - noisy.data["x1_true"]
    dc = noisy.data["c"] - noisy.data["c_true"]

    # mean: error on the mean is sigma/sqrt(N); allow 5 sigma
    assert np.abs(dx1.mean()) < 5 * 0.1 / np.sqrt(n)
    assert np.abs(dc.mean()) < 5 * 0.05 / np.sqrt(n)
    # std: error on the std is ~ sigma/sqrt(2N); allow 5 sigma
    assert np.abs(dx1.std() - 0.1) < 5 * 0.1 / np.sqrt(2 * n)
    assert np.abs(dc.std() - 0.05) < 5 * 0.05 / np.sqrt(2 * n)
    # correlation: 0.003 / (0.1*0.05) = 0.6 ; error on rho ~ (1-rho**2)/sqrt(N)
    rho = np.corrcoef(dx1, dc)[0, 1]
    assert np.abs(rho - 0.6) < 5 * (1 - 0.6**2) / np.sqrt(n)


# tests for the func random_radec
def test_random_radec_size():
    ra, dec = utils.random_radec(size=100, rng=1)
    assert len(ra) == 100
    assert len(dec) == 100

def test_random_radec_ranges():
    ra, dec = utils.random_radec(size=1000, ra_range=[200, 250], dec_range=[-20, 10], rng=1)
    assert np.all((ra >= 200) & (ra <= 250))
    assert np.all((dec >= -20) & (dec <= 10))

def test_random_radec_seed_reproducible():
    ra1, dec1 = utils.random_radec(size=10, rng=42)
    ra2, dec2 = utils.random_radec(size=10, rng=42)
    np.testing.assert_array_equal(ra1, ra2)
    np.testing.assert_array_equal(dec1, dec2)

def test_random_radec_uniform_on_sphere():
    """ the fraction of the sphere above dec=30 is (1 - sin(30deg))/2 = 0.25 """
    n = 100_000
    ra, dec = utils.random_radec(size=n, rng=1)
    frac = np.mean(dec > 30)
    assert np.abs(frac - 0.25) < 5 * np.sqrt(0.25 * 0.75 / n)

def test_random_radec_uniform_in_ra():
    n = 100_000
    ra, dec = utils.random_radec(size=n, rng=1)
    frac = np.mean(ra < 90) # a quarter of [0, 360]
    assert np.abs(frac - 0.25) < 5 * np.sqrt(0.25 * 0.75 / n)

def test_random_radec_skyarea_polygon():
    skyarea = geometry.box(10, -5, 30, 5)
    ra, dec = utils.random_radec(size=500, skyarea=skyarea, rng=1)

    assert len(ra) == 500
    assert np.all((ra >= 10) & (ra <= 30))
    assert np.all((dec >= -5) & (dec <= 5))

def test_random_radec_skyarea_multipolygon():
    box1 = geometry.box(10, -5, 30, 5)
    box2 = geometry.box(100, 20, 120, 30)
    skyarea = geometry.MultiPolygon([box1, box2])
    ra, dec = utils.random_radec(size=500, skyarea=skyarea, rng=1)

    assert len(ra) == 500
    in_box1 = (ra >= 10) & (ra <= 30) & (dec >= -5) & (dec <= 5)
    in_box2 = (ra >= 100) & (ra <= 120) & (dec >= 20) & (dec <= 30)
    assert np.all(in_box1 | in_box2)
    assert np.any(in_box1) and np.any(in_box2), "both regions should be populated"
    assert len(np.unique(ra)) == 500, "the same position should never be drawn twice"

def test_random_radec_skyarea_multipolygon_follows_area():
    # a 10x10 deg and a 2x2 deg box on the equator: the small one has
    # 2 x 2 sin(1deg) / (10 x 2 sin(5deg) + 2 x 2 sin(1deg)) = 3.85% of the sky area.
    big, small = geometry.box(0, -5, 10, 5), geometry.box(100, -1, 102, 1)
    n = 100_000
    ra, dec = utils.random_radec(size=n, skyarea=geometry.MultiPolygon([big, small]), rng=1)

    sindeg = lambda deg: np.sin(np.radians(deg))
    expected = 2 * 2 * sindeg(1) / (10 * 2 * sindeg(5) + 2 * 2 * sindeg(1))
    frac = np.mean(ra > 50)
    assert np.abs(frac - expected) < 5 * np.sqrt(expected * (1 - expected) / n), \
        f"positions do not follow the sky area: {frac=:.4f} vs {expected=:.4f}"

def test_random_radec_skyarea_overlaps_count_once():
    # the same box given twice and another box of the same area: 50% each.
    box1, box2 = geometry.box(0, -5, 10, 5), geometry.box(50, -5, 60, 5)
    n = 50_000
    ra, dec = utils.random_radec(size=n, skyarea=geometry.MultiPolygon([box1, box1, box2]), rng=1)
    assert np.abs(np.mean(ra < 30) - 0.5) < 5 * np.sqrt(0.25 / n), "overlapping polygons counted twice"

def test_random_radec_skyarea_slanted_edges():
    # large triangle with a slanted edge reaching high dec: positions must be
    # uniform on the sphere within it, i.e. the same as a full-sky draw kept inside.
    triangle = geometry.Polygon([(0, 0), (60, 0), (0, 80)])
    small = geometry.box(200, -10, 230, 10)
    ra_sky, dec_sky = utils.random_radec(size=2_000_000, rng=2) # full sky
    in_triangle = shapely.contains_xy(triangle, ra_sky, dec_sky).sum()
    in_small = shapely.contains_xy(small, ra_sky, dec_sky).sum()
    expected = in_triangle / (in_triangle + in_small)

    n = 100_000
    ra, dec = utils.random_radec(size=n, skyarea=geometry.MultiPolygon([triangle, small]), rng=3)
    # errors from both samples
    sigma = np.sqrt(expected * (1 - expected) * (1 / n + 1 / (in_triangle + in_small)))
    assert np.abs(np.mean(ra < 100) - expected) < 5 * sigma

def test_random_radec_skyarea_thin_shape():
    # an annulus fills less than a quarter of its bounding box
    annulus = geometry.Point(50, 0).buffer(10).difference(geometry.Point(50, 0).buffer(9.5))
    ra, dec = utils.random_radec(size=5000, skyarea=annulus, rng=1)
    assert len(ra) == 5000
    assert shapely.contains_xy(annulus.buffer(1e-9), ra, dec).all()

def test_random_radec_skyarea_with_ranges():
    # the skyarea is cut by the ra and dec ranges
    skyarea = geometry.box(0, -10, 20, 10)
    ra, dec = utils.random_radec(size=1000, skyarea=skyarea, ra_range=[10, 360], dec_range=[0, 90], rng=1)
    assert np.all((ra >= 10) & (ra <= 20) & (dec >= 0) & (dec <= 10))

def test_random_radec_skyarea_size_none():
    ra, dec = utils.random_radec(skyarea=geometry.box(10, -5, 30, 5), rng=1)
    assert np.ndim(ra) == 0 and np.ndim(dec) == 0
    assert 10 <= ra <= 30 and -5 <= dec <= 5

def test_random_radec_skyarea_seed_reproducible():
    skyarea = geometry.MultiPolygon([geometry.box(0, -5, 10, 5), geometry.box(50, -5, 60, 5)])
    ra1, dec1 = utils.random_radec(size=10, skyarea=skyarea, rng=42)
    ra2, dec2 = utils.random_radec(size=10, skyarea=skyarea, rng=42)
    np.testing.assert_array_equal(ra1, ra2)
    np.testing.assert_array_equal(dec1, dec2)

def test_random_radec_skyarea_empty():
    with pytest.raises(ValueError):
        utils.random_radec(size=10, skyarea=geometry.box(0, 50, 10, 60), dec_range=[-90, 0])

def test_random_radec_skyarea_size_zero():
    ra, dec = utils.random_radec(size=0, skyarea=geometry.box(10, -5, 30, 5), rng=1)
    assert len(ra) == 0 and len(dec) == 0

def test_random_radec_no_skyarea_draw():
    # without skyarea: ra uniform in ra_range and sin(dec) uniform in sin(dec_range),
    # drawn in that order (same positions as before the skyarea rewrite for a given rng).
    ra, dec = utils.random_radec(size=10, ra_range=[200, 250], dec_range=[-20, 10], rng=42)

    rng = np.random.default_rng(42)
    expected_ra = rng.uniform(200, 250, size=10)
    expected_dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-20)), np.sin(np.radians(10)), size=10)))
    np.testing.assert_allclose(ra, expected_ra)
    np.testing.assert_allclose(dec, expected_dec)

def test_random_radec_isotropic():
    # isotropy on the sphere means ra and sin(dec) are uniform.
    from scipy import stats
    ra, dec = utils.random_radec(size=100_000, rng=5)
    assert stats.kstest(ra, stats.uniform(loc=0, scale=360).cdf).pvalue > 1e-3, "ra is not uniform"
    assert stats.kstest(np.sin(np.radians(dec)), stats.uniform(loc=-1, scale=2).cdf).pvalue > 1e-3, \
        "sin(dec) is not uniform"

def test_random_radec_dec_range_follows_sindec():
    # within dec_range=[0, 90], the fraction above dec=30 is (1 - sin(30deg)) / (1 - 0) = 0.5
    n = 100_000
    ra, dec = utils.random_radec(size=n, dec_range=[0, 90], rng=2)
    assert np.all(dec >= 0)
    assert np.abs(np.mean(dec > 30) - 0.5) < 5 * np.sqrt(0.25 / n)

def test_random_radec_skyarea_uniform_within_polygon():
    # inside a polygon far from the equator, ra and sin(dec) must still be uniform.
    from scipy import stats
    ra, dec = utils.random_radec(size=50_000, skyarea=geometry.box(10, 40, 50, 70), rng=3)
    sindec_low, sindec_high = np.sin(np.radians([40, 70]))
    assert stats.kstest(ra, stats.uniform(loc=10, scale=40).cdf).pvalue > 1e-3
    assert stats.kstest(np.sin(np.radians(dec)),
                        stats.uniform(loc=sindec_low, scale=sindec_high - sindec_low).cdf).pvalue > 1e-3

def test_random_radec_skyarea_dithered_pointings():
    # 5x5 dithered pointings: 2x2 deg boxes shifted by 0.5 deg, as survey.get_skyarea()
    # returns them (overlapping). Their union is the box [0, 4] x [-2, 2]: positions must
    # be uniform within it, not piled up where many pointings overlap.
    from scipy import stats
    offsets = np.arange(5) * 0.5
    pointings = [geometry.box(ra_, dec_ - 2, ra_ + 2, dec_) for ra_ in offsets for dec_ in offsets]
    n = 50_000
    ra, dec = utils.random_radec(size=n, skyarea=geometry.MultiPolygon(pointings), rng=4)

    sindec_low, sindec_high = np.sin(np.radians([-2, 2]))
    assert stats.kstest(ra, stats.uniform(loc=0, scale=4).cdf).pvalue > 1e-3, "ra is not uniform in the union"
    assert stats.kstest(np.sin(np.radians(dec)),
                        stats.uniform(loc=sindec_low, scale=sindec_high - sindec_low).cdf).pvalue > 1e-3, \
        "sin(dec) is not uniform in the union"

    # mean number of pointings covering a position: sum of the pointing areas / area of the union
    # (sky area of a ra-dec box: dra x (sin(dec_max) - sin(dec_min)))
    sky_area = lambda ramin, decmin, ramax, decmax: (ramax - ramin) * (np.sin(np.radians(decmax)) - np.sin(np.radians(decmin)))
    expected = np.sum([sky_area(*pointing_.bounds) for pointing_ in pointings]) / sky_area(0, -2, 4, 2)
    coverage = np.sum([shapely.contains_xy(pointing_, ra, dec) for pointing_ in pointings], axis=0)
    assert np.abs(coverage.mean() - expected) < 5 * coverage.std() / np.sqrt(n), \
        f"positions pile up where pointings overlap: {coverage.mean()=:.2f} vs {expected=:.2f}"


# tests for the func surface_of_skyarea
def test_surface_of_skyarea_str_not_full():
    assert utils.surface_of_skyarea("not_full") is None

def test_surface_of_skyarea_full():
    assert utils.surface_of_skyarea("full") == "full"

def test_surface_of_skyarea_float():
    assert utils.surface_of_skyarea(100.) == 100.

def test_surface_of_skyarea_no_projection():
    skyarea = geometry.box(0, -10, 20, 10)
    assert utils.surface_of_skyarea(skyarea, incl_projection=False) == pytest.approx(400.)

def test_surface_of_skyarea_with_projection():
    skyarea = geometry.box(0, -10, 20, 10)
    # true solid angle: delta_ra * delta_sin(dec), back to deg2
    expected = 20 * 2 * np.sin(np.deg2rad(10)) * 180 / np.pi
    assert utils.surface_of_skyarea(skyarea) == pytest.approx(expected)

def test_surface_of_skyarea_full_sky_polygon():
    skyarea = geometry.box(0, -90, 360, 90)
    # full sky is 4pi sr = 41253 deg2
    assert utils.surface_of_skyarea(skyarea) == pytest.approx(4 * np.pi * (180 / np.pi)**2)


# tests for the func parse_skyarea
def test_parse_skyarea_str_not_full():
    assert utils.parse_skyarea("not_full") is None

def test_parse_skyarea_full():
    assert utils.parse_skyarea("full") == "full"

def test_parse_skyarea_polygon():
    skyarea = geometry.box(0, 0, 1, 1)
    assert utils.parse_skyarea(skyarea) is skyarea


# tests for the func random_radecz_skymap
def test_random_radecz_skymap_no_ligo(monkeypatch):
    monkeypatch.setattr(utils, "LIGO_SKYMAP_IMPORTED", False)
    with pytest.raises(ImportError):
        utils.random_radecz_skymap(size=10)
