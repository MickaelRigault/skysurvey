import numpy as np
import pandas
import pytest
import geopandas
import shapely
from shapely import geometry

from skysurvey.tools import utils
from skysurvey.tools import projection


def angular_separation(ra1, dec1, ra2, dec2):
    """ great-circle distance in deg (haversine) """
    ra1, dec1, ra2, dec2 = [np.deg2rad(np.asarray(x, dtype="float")) for x in [ra1, dec1, ra2, dec2]]
    hav = np.sin((dec2 - dec1) / 2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2)**2
    return np.rad2deg(2 * np.arcsin(np.sqrt(hav)))


# tests for the func skyarea_to_skysurface
def test_skyarea_to_skysurface_full_sky_frac():
    full_sky = geometry.box(0, -90, 360, 90)
    # 2pi (ra) x 2 (sin dec) = 4pi => fraction 1
    assert projection.skyarea_to_skysurface(full_sky) == pytest.approx(1.)

def test_skyarea_to_skysurface_half_sky_steradian():
    north = geometry.box(0, 0, 360, 90)
    assert projection.skyarea_to_skysurface(north, frac=False, in_deg2=False) == pytest.approx(2 * np.pi)

def test_skyarea_to_skysurface_deg2_no_projection():
    skyarea = geometry.box(0, -10, 20, 10)
    surface = projection.skyarea_to_skysurface(skyarea, frac=False, in_deg2=True, projection_correction=False)
    assert surface == pytest.approx(400.) # flat 20 x 20 deg

def test_skyarea_to_skysurface_deg2_with_projection():
    skyarea = geometry.box(0, -10, 20, 10)
    # delta_ra (rad) * delta_sin(dec), back to deg2  (~398 deg2, see docstring)
    expected = np.deg2rad(20) * 2 * np.sin(np.deg2rad(10)) * (180 / np.pi)**2
    surface = projection.skyarea_to_skysurface(skyarea, frac=False, in_deg2=True)
    assert surface == pytest.approx(expected)
    assert surface < 400.

def test_skyarea_to_skysurface_overlap_counted_once():
    box = geometry.box(0, -10, 20, 10)
    single = projection.skyarea_to_skysurface(box)
    double = projection.skyarea_to_skysurface([box, box])
    assert double == pytest.approx(single)

def test_skyarea_to_skysurface_disjoint_add_up():
    box1 = geometry.box(0, -10, 20, 10)
    box2 = geometry.box(100, -10, 120, 10) # same size, elsewhere on the equator
    single = projection.skyarea_to_skysurface(box1)
    assert projection.skyarea_to_skysurface([box1, box2]) == pytest.approx(2 * single)


# tests for the func radecmodel_to_skysurface
def test_radecmodel_to_skysurface_favor_skyarea():
    """ with a skyarea, the analytic value of skyarea_to_skysurface is returned """
    skyarea = geometry.box(0, 0, 360, 90)
    radec = {"func": utils.random_radec, "kwargs": {"skyarea": skyarea}, "as": ["ra", "dec"]}

    assert projection.radecmodel_to_skysurface(radec, frac=True) == pytest.approx(0.5)
    assert projection.radecmodel_to_skysurface(radec, frac=False) == pytest.approx(2 * np.pi)

def test_radecmodel_to_skysurface_sampling_warns():
    skyarea = geometry.box(0, 0, 90, 30)
    radec = {"func": utils.random_radec, "kwargs": {"skyarea": skyarea}, "as": ["ra", "dec"]}
    with pytest.warns(UserWarning):
        projection.radecmodel_to_skysurface(radec, favor_skyarea=False, ntrial=1e3)

def test_radecmodel_to_skysurface_sampling_matches_analytic():
    """ a ra/dec box is a rectangle in (ra, sin dec): its convex hull is the true area """
    skyarea = geometry.box(0, 0, 90, 30)
    radec = {"func": utils.random_radec, "kwargs": {"skyarea": skyarea}, "as": ["ra", "dec"]}
    expected = (np.pi / 2) * np.sin(np.deg2rad(30)) # steradian
    with pytest.warns(UserWarning):
        area = projection.radecmodel_to_skysurface(radec, favor_skyarea=False, ntrial=2e4, frac=False)
    # (no seed: the model is drawn by ModelDAG internally)
    # the convex hull of N uniform points misses a fraction ~log(N)/N of a rectangle,
    # i.e. ~5e-4 here, far below the 1% tolerance.
    assert area == pytest.approx(expected, rel=1e-2)
    assert area <= expected

def test_radecmodel_to_skysurface_multipolygon_sampling_raises():
    skyarea = geometry.MultiPolygon([geometry.box(0, 0, 10, 10), geometry.box(50, 0, 60, 10)])
    radec = {"func": utils.random_radec, "kwargs": {"skyarea": skyarea}, "as": ["ra", "dec"]}
    with pytest.warns(UserWarning):
        with pytest.raises(NotImplementedError):
            projection.radecmodel_to_skysurface(radec, favor_skyarea=False)


# tests for the func project_to_radec
def test_project_to_radec_vertices_shape():
    verts = np.array([[-1, 1, 0], [0, 0, 1]]) # x, y = verts ; 3 vertices
    pointings = projection.project_to_radec(verts, [100., 200.], [0., 30.])
    assert np.shape(pointings) == (2, 3, 2) # (n_pointings, n_vertices, [ra, dec])

def test_project_to_radec_equator_is_translation():
    """ at dec=0 there is no rotation: vertices are simply shifted by ra """
    verts = np.array([[-1, 1, 0], [0, 0, 1]])
    pointings = projection.project_to_radec(verts, [100., 200.], [0., 0.])
    np.testing.assert_allclose(pointings[0], [[99, 0], [101, 0], [100, 1]], atol=1e-10)
    np.testing.assert_allclose(pointings[1], [[199, 0], [201, 0], [200, 1]], atol=1e-10)

def test_project_to_radec_center_goes_to_pointing():
    verts = np.array([[0, 1], [0, 0]]) # first vertex is the footprint center
    pointings = projection.project_to_radec(verts, [50., 300.], [30., -60.])
    np.testing.assert_allclose(pointings[0][0], [50, 30], atol=1e-10)
    np.testing.assert_allclose(pointings[1][0], [300, -60], atol=1e-10)

def test_project_to_radec_dec_offset_is_kept():
    """ a vertex 1 deg north of the center stays 1 deg north of the pointing """
    verts = np.array([[0, 0], [0, 1]])
    pointings = projection.project_to_radec(verts, [50., 50.], [30., 30.])
    np.testing.assert_allclose(pointings[0][1], [50, 31], atol=1e-10)

def test_project_to_radec_preserves_distance_to_center():
    """ projection is a rotation: the angular distance to the center is preserved """
    verts = np.array([[1, -1, 0, 0], [0, 0, 1, -1]]) # all 1 deg away from (0, 0)
    pointings = projection.project_to_radec(verts, [10., 150.], [45., -70.])
    for (ra0, dec0), pointing in zip([(10., 45.), (150., -70.)], pointings):
        sep = angular_separation(ra0, dec0, pointing[:, 0], pointing[:, 1])
        np.testing.assert_allclose(sep, 1., atol=1e-8)

def test_project_to_radec_single_pointing():
    # one projection per pointing, also for a single pointing (np.squeeze used to drop that axis)
    footprint = geometry.box(-1, -1, 1, 1)
    polygons = projection.project_to_radec(footprint, 100., 0.)
    assert len(polygons) == 1
    np.testing.assert_allclose(polygons[0].bounds, [99, -1, 101, 1], atol=1e-3) # equator: a translation
    verts = np.asarray(footprint.exterior.xy)
    assert projection.project_to_radec(verts, 100., 0.).shape == (1, verts.shape[1], 2)

# tests for the func camera_to_sky / sky_to_camera
def test_camera_to_sky_no_rotation():
    # rotation=0: camera 'up' (+y) is North, +x is West (East on the left)
    np.testing.assert_allclose(projection.camera_to_sky(0, 1, 0), [0, 1], atol=1e-12)
    np.testing.assert_allclose(projection.camera_to_sky(1, 0, 0), [-1, 0], atol=1e-12)

def test_sky_to_camera_north_moves_east():
    # an increasing rotation makes North move toward East (camera left, x<0)
    x, y = projection.sky_to_camera(0, 1, 10)
    assert x < 0 and np.isclose(np.hypot(x, y), 1)

def test_camera_sky_roundtrip():
    x, y = np.random.default_rng(2).uniform(-2, 2, size=(2, 100))
    rot = np.random.default_rng(3).uniform(0, 360, size=100)
    np.testing.assert_allclose(projection.sky_to_camera(*projection.camera_to_sky(x, y, rot), rot), [x, y])


def test_project_to_radec_polygon_returns_polygons():
    footprint = geometry.box(-1, -1, 1, 1)
    polygons = projection.project_to_radec(footprint, [100., 200.], [0., 0.])

    assert len(polygons) == 2
    assert all(isinstance(p, geometry.Polygon) for p in polygons)
    # at dec=0 there is no rotation, but the x-offset of a vertex is converted
    # into an ra offset at its own dec: x / cos(dec). Corners are at dec=+-1.
    dra = 1 / np.cos(np.deg2rad(1))
    np.testing.assert_allclose(polygons[0].bounds, [100-dra, -1, 100+dra, 1], atol=1e-10)
    np.testing.assert_allclose(polygons[1].bounds, [200-dra, -1, 200+dra, 1], atol=1e-10)


# tests for the func parse_fields
def test_parse_fields_dict_of_arrays():
    fields = {10: np.array([[0, 0], [2, 0], [2, 2], [0, 2]]),
              20: np.array([[1, 1], [3, 1], [3, 3], [1, 3]])}
    geodf = projection.parse_fields(fields)

    assert isinstance(geodf, geopandas.GeoDataFrame)
    assert list(geodf["fieldid"]) == [10, 20]
    assert list(geodf.geometry.area) == [4., 4.]

def test_parse_fields_dict_of_regions():
    fields = {450: "box(50,30,3,4,0)", 541: "circle(190,-10,1.5)"}
    geodf = projection.parse_fields(fields)

    assert list(geodf["fieldid"]) == [450, 541]
    np.testing.assert_allclose(geodf.geometry.centroid.x, [50, 190], atol=1e-8)
    np.testing.assert_allclose(geodf.geometry.centroid.y, [30, -10], atol=1e-8)

def test_parse_fields_geoseries():
    geoseries = geopandas.GeoSeries([geometry.box(0, 0, 1, 1), geometry.box(5, 5, 6, 6)], index=[3, 7])
    geodf = projection.parse_fields(geoseries)

    assert isinstance(geodf, geopandas.GeoDataFrame)
    assert list(geodf["fieldid"]) == [3, 7]

def test_parse_fields_geodataframe_unchanged():
    geodf = geopandas.GeoDataFrame({"fieldid": [1]}, geometry=[geometry.box(0, 0, 1, 1)])
    assert projection.parse_fields(geodf) is geodf

def test_parse_fields_bad_format():
    with pytest.raises(ValueError):
        projection.parse_fields([geometry.box(0, 0, 1, 1)])


# tests for the func spatialjoin_radec_to_fields
@pytest.fixture
def two_fields():
    # two overlapping 2x2 deg fields
    return {10: np.array([[0, 0], [2, 0], [2, 2], [0, 2]]),
            20: np.array([[1, 1], [3, 1], [3, 3], [1, 3]])}

def test_spatialjoin_radec_to_fields_array(two_fields):
    radec = np.array([[0.5, 0.5],  # field 10 only
                      [1.5, 1.5],  # both fields
                      [5.0, 5.0]]) # no field
    sjoined = projection.spatialjoin_radec_to_fields(radec, two_fields)

    matches = sorted(zip(sjoined["index_radec"], sjoined["fieldid"]))
    assert matches == [(0, 10), (1, 10), (1, 20)]

def test_spatialjoin_radec_to_fields_dataframe_keeps_index(two_fields):
    radec = pandas.DataFrame({"ra": [0.5, 2.5], "dec": [0.5, 2.5]}, index=[100, 200])
    sjoined = projection.spatialjoin_radec_to_fields(radec, two_fields)

    matches = sorted(zip(sjoined["index_radec"], sjoined["fieldid"]))
    assert matches == [(100, 10), (200, 20)]

def test_spatialjoin_radec_to_fields_bad_shape(two_fields):
    with pytest.raises(ValueError):
        projection.spatialjoin_radec_to_fields(np.array([[0.5, 0.5, 0.5]]), two_fields)

def test_spatialjoin_radec_to_fields_multiindex():
    """ multi-index fields (e.g. fieldid, ccd) are returned as columns named after the index levels """
    index = pandas.MultiIndex.from_tuples([(1, 2), (1, 3)], names=["fieldid", "ccd"])
    fields = geopandas.GeoDataFrame(geometry=[geometry.box(0, 0, 1, 1), geometry.box(1, 0, 2, 1)],
                                    index=index)
    radec = pandas.DataFrame({"ra": [0.5, 1.5], "dec": [0.5, 0.5]}, index=[7, 8])
    sjoined = projection.spatialjoin_radec_to_fields(radec, fields)

    matches = sorted(zip(sjoined["index_radec"], sjoined["fieldid"], sjoined["ccd"]))
    assert matches == [(7, 1, 2), (8, 1, 3)]


# tests for the func regions_to_shapely
def test_regions_to_shapely_box_center():
    geom = projection.regions_to_shapely("box(40.0, 50.0, 5.0, 4.0, 0.0)")
    assert isinstance(geom, geometry.Polygon)
    np.testing.assert_allclose([geom.centroid.x, geom.centroid.y], [40, 50], atol=1e-10)

def test_regions_to_shapely_rotated_box_center():
    geom = projection.regions_to_shapely("box(-30, 0.4, 4, 2, 80)")
    np.testing.assert_allclose([geom.centroid.x, geom.centroid.y], [-30, 0.4], atol=1e-10)

def test_regions_to_shapely_circle():
    geom = projection.regions_to_shapely("circle(10, 20, 2)")
    np.testing.assert_allclose([geom.centroid.x, geom.centroid.y], [10, 20], atol=1e-10)
    np.testing.assert_allclose(geom.bounds, [8, 18, 12, 22], atol=1e-10)
    # buffer is a polygon approximation of the circle
    assert geom.area == pytest.approx(np.pi * 4, rel=1e-2)

def test_regions_to_shapely_ellipse():
    """ ds9 ellipse(x, y, r1, r2, angle): r1 and r2 are the semi-axes """
    geom = projection.regions_to_shapely("ellipse(54, 43.4, 4, 2, 0)")
    np.testing.assert_allclose(geom.bounds, [50, 41.4, 58, 45.4], atol=1e-10)
    assert geom.area == pytest.approx(np.pi * 4 * 2, rel=1e-2)

def test_regions_to_shapely_ellipse_rotated_90():
    geom = projection.regions_to_shapely("ellipse(54, 43.4, 4, 2, 90)")
    # semi-major axis now along y
    np.testing.assert_allclose(geom.bounds, [52, 39.4, 56, 47.4], atol=1e-10)

def test_regions_to_shapely_polygon():
    geom = projection.regions_to_shapely("polygon(10, 20, 30, 20, 30, 40)")
    assert isinstance(geom, geometry.Polygon)
    assert geom.area == pytest.approx(200.) # right triangle 20 x 20 / 2

def test_regions_to_shapely_unknown_shape():
    with pytest.raises(NotImplementedError):
        projection.regions_to_shapely("point(10, 20)")

def test_regions_to_shapely_not_a_str():
    with pytest.raises(ValueError):
        projection.regions_to_shapely(12)


# tests for the func cart2sph
def test_cart2sph_axes():
    np.testing.assert_allclose(projection.cart2sph([1., 0., 0.]), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(projection.cart2sph([0., 2., 0.]), [2, 90, 0], atol=1e-12)
    np.testing.assert_allclose(projection.cart2sph([0., 0., 3.]), [3, 0, 90], atol=1e-12)

def test_cart2sph_longitude_range():
    """ the longitude is returned in [-180, 180) """
    np.testing.assert_allclose(projection.cart2sph([0., -1., 0.]), [1, -90, 0], atol=1e-12)
    np.testing.assert_allclose(projection.cart2sph([-1., 0., 0.]), [1, -180, 0], atol=1e-12)

def test_cart2sph_vectorized():
    vec = np.array([[1., 0.], [0., 0.], [0., -1.]]) # x, y, z for 2 points
    r, lon, lat = projection.cart2sph(vec)
    np.testing.assert_allclose(r, [1, 1])
    np.testing.assert_allclose(lat, [0, -90], atol=1e-12)


# tests for the func sph2cart
def test_sph2cart_radius():
    np.testing.assert_allclose(projection.sph2cart([2., 0., 90.]), [0, 0, 2], atol=1e-12)

def test_sph2cart_roundtrip():
    lon = np.array([-170., -30., 0., 45., 120.])
    lat = np.array([-80., -10., 0., 35., 89.])
    r, lon_, lat_ = projection.cart2sph(projection.sph2cart([1., lon, lat]))
    np.testing.assert_allclose(r, 1.)
    np.testing.assert_allclose(lon_, lon, atol=1e-10)
    np.testing.assert_allclose(lat_, lat, atol=1e-10)


# tests for the func rot_xz
def test_rot_xz_zero_is_identity():
    vec = [np.array([0.3]), np.array([0.4]), np.array([0.5])]
    rotated = projection.rot_xz(vec, 0.)
    np.testing.assert_allclose(np.ravel(rotated[0]), [0.3])
    np.testing.assert_allclose(np.ravel(rotated[1]), [0.4])
    np.testing.assert_allclose(np.ravel(rotated[2]), [0.5])

def test_rot_xz_90():
    """ +90deg around the y-axis sends x to z (and z to -x) """
    vec = [np.array([1., 0.]), np.array([0., 0.]), np.array([0., 1.])]
    x, y, z = projection.rot_xz(vec, 90.)
    np.testing.assert_allclose(np.ravel(x), [0, -1], atol=1e-12)
    np.testing.assert_allclose(np.ravel(y), [0, 0], atol=1e-12)
    np.testing.assert_allclose(np.ravel(z), [1, 0], atol=1e-12)

def test_rot_xz_keeps_y():
    vec = [np.array([0.]), np.array([1.]), np.array([0.])]
    x, y, z = projection.rot_xz(vec, 37.)
    np.testing.assert_allclose([np.ravel(x)[0], np.ravel(y)[0], np.ravel(z)[0]], [0, 1, 0], atol=1e-12)


# tests for the func rot_xz_sph
def test_rot_xz_sph_zero_is_identity():
    lon, lat = projection.rot_xz_sph(np.array([10., -40.]), np.array([20., 60.]), 0.)
    np.testing.assert_allclose(np.ravel(lon), [10, -40], atol=1e-10)
    np.testing.assert_allclose(np.ravel(lat), [20, 60], atol=1e-10)

def test_rot_xz_sph_moves_equator_point_in_latitude():
    lon, lat = projection.rot_xz_sph(np.array([0.]), np.array([0.]), 45.)
    np.testing.assert_allclose(np.ravel([lon, lat]), [0, 45], atol=1e-10)

def test_rot_xz_sph_through_the_pole():
    """ (0, 80) rotated by 20 deg goes 10 deg beyond the pole: lat 80 on the other side """
    lon, lat = projection.rot_xz_sph(np.array([0.]), np.array([80.]), 20.)
    assert np.abs(np.ravel(lon)[0]) == pytest.approx(180.)
    assert np.ravel(lat)[0] == pytest.approx(80.)

def test_rot_xz_sph_inverse():
    lon, lat = np.array([12., -100.]), np.array([33., -5.])
    lon1, lat1 = projection.rot_xz_sph(lon, lat, 25.)
    lon2, lat2 = projection.rot_xz_sph(np.ravel(lon1), np.ravel(lat1), -25.)
    np.testing.assert_allclose(np.ravel(lon2), lon, atol=1e-10)
    np.testing.assert_allclose(np.ravel(lat2), lat, atol=1e-10)
