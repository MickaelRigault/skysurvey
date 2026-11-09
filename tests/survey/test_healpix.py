import warnings

import numpy as np
import pandas
import pytest
from shapely import geometry

import healpy as hp

from skysurvey.survey.healpix import HealpixSurvey, get_ipix_in_range

NSIDE = 4 # 12 x 4^2 = 192 pixels
FULLSKY_DEG2 = 4 * np.pi * (180 / np.pi)**2 # ~41253 deg2


@pytest.fixture
def logs():
    """ tiny observing log: field 10 observed twice, 20 and 30 once """
    return pandas.DataFrame({"mjd": [3., 1., 2., 4.],
                             "band": ["g", "r", "g", "r"],
                             "skynoise": [1., 2., 3., 4.],
                             "gain": [1., 1., 1., 1.],
                             "zp": [25., 25., 25., 25.],
                             "fieldid": [10, 10, 20, 30]})

@pytest.fixture
def survey(logs):
    return HealpixSurvey(nside=NSIDE, data=logs)

def centroids_of(survey, fieldids):
    """ dataframe of the (ra, dec) centroids of the given fields """
    ra, dec = survey.get_field_centroid()
    return pandas.DataFrame({"ra": ra[fieldids], "dec": dec[fieldids]})


# tests for the class HealpixSurvey
def test_init_no_data():
    survey = HealpixSurvey(nside=NSIDE)
    assert survey.nside == NSIDE
    assert survey.data is None

def test_npix():
    # healpix has 12 x nside^2 pixels
    survey = HealpixSurvey(nside=NSIDE)
    assert survey.npix == 192
    assert survey.nfields == 192

def test_fieldids():
    survey = HealpixSurvey(nside=NSIDE)
    assert survey.fieldids.name == "fieldid"
    assert list(survey.fieldids) == list(range(192))


# tests for the func from_data
def test_from_data(logs):
    survey = HealpixSurvey.from_data(nside=NSIDE, data=logs)
    assert isinstance(survey, HealpixSurvey)
    assert survey.nside == NSIDE
    assert len(survey.data) == 4


# tests for the func get_field_area
def test_get_field_area():
    # equal-area pixels: full sky / 12 nside^2
    survey = HealpixSurvey(nside=NSIDE)
    assert np.isclose(survey.get_field_area(), FULLSKY_DEG2 / 192)

def test_get_field_area_fullsky():
    survey = HealpixSurvey(nside=32)
    assert np.isclose(survey.get_field_area() * survey.npix, 41253, rtol=1e-4)


# tests for the func get_observed_area
def test_get_observed_area(survey):
    # 3 distinct fields observed (10, 20, 30)
    assert np.isclose(survey.get_observed_area(), 3 * survey.get_field_area())

def test_get_observed_area_minobs_0(survey):
    # min_obs=0 and min_obs=1 are the same
    assert survey.get_observed_area(min_obs=0) == survey.get_observed_area(min_obs=1)


# tests for the func get_field_centroid
def test_get_field_centroid_shape():
    survey = HealpixSurvey(nside=NSIDE)
    ra, dec = survey.get_field_centroid()
    assert ra.shape == (192,)
    assert dec.shape == (192,)

def test_get_field_centroid_range():
    survey = HealpixSurvey(nside=NSIDE)
    ra, dec = survey.get_field_centroid()
    assert np.all((ra >= 0) & (ra < 360))
    assert np.all((dec > -90) & (dec < 90))

def test_get_field_centroid_equal_area():
    # equal-area pixels: half of the pixel centroids are in the north hemisphere
    survey = HealpixSurvey(nside=NSIDE)
    _, dec = survey.get_field_centroid()
    assert np.sum(dec > 0) == np.sum(dec < 0) == 96 - 2 * NSIDE
    # the remaining 4*nside pixels are centered on the equator
    assert np.sum(np.isclose(dec, 0)) == 4 * NSIDE


# tests for the func radec_to_fieldid
def test_radec_to_fieldid_centroids(survey):
    # the centroid of a field belongs to that field
    fieldids = [0, 10, 20, 100, 191]
    radec = centroids_of(survey, fieldids)
    assert list(survey.radec_to_fieldid(radec)["fieldid"]) == fieldids

def test_radec_to_fieldid_index(survey):
    radec = centroids_of(survey, [10, 20])
    radec.index = [5, 7]
    fields = survey.radec_to_fieldid(radec)
    assert list(fields.index) == [5, 7]
    assert fields.index.name == "index_radec", "unnamed input index should be named index_radec"
    assert list(fields["fieldid"]) == [10, 20]

def test_radec_to_fieldid_named_index(survey):
    radec = centroids_of(survey, [10, 20])
    radec.index.name = "target"
    assert survey.radec_to_fieldid(radec).index.name == "target"

def test_radec_to_fieldid_array(survey):
    # a single (ra, dec) pair
    radec = centroids_of(survey, [20]).iloc[0]
    fields = survey.radec_to_fieldid([radec["ra"], radec["dec"]])
    assert list(fields["fieldid"]) == [20]
    assert list(fields.index) == [0]

def test_radec_to_fieldid_poles(survey):
    # the first (last) healpix pixels are at the north (south) pole
    radec = pandas.DataFrame({"ra": [10., 10.], "dec": [89.99, -89.99]})
    fields = survey.radec_to_fieldid(radec)["fieldid"]
    assert fields.iloc[0] in [0, 1, 2, 3]
    assert fields.iloc[1] in [188, 189, 190, 191]

def test_radec_to_fieldid_observed_fields(survey):
    # 5 is not observed
    radec = centroids_of(survey, [10, 5, 20])
    fields = survey.radec_to_fieldid(radec, observed_fields=True)
    assert list(fields["fieldid"]) == [10, 20]
    assert list(fields.index) == [0, 2], "index should follow the input radec"


# tests for the func get_observations_from_coords
def test_get_observations_from_coords(survey):
    radec = centroids_of(survey, [10])
    obs = survey.get_observations_from_coords(radec)
    assert list(obs["fieldid"]) == [10, 10]
    assert list(obs["mjd"]) == [1., 3.]

def test_get_observations_from_coords_unobserved(survey):
    radec = centroids_of(survey, [5])
    assert len(survey.get_observations_from_coords(radec)) == 0


# tests for the func get_fieldstat
def test_get_fieldstat_size(survey):
    nobs = survey.get_fieldstat("size")
    assert nobs.to_dict() == {10: 2, 20: 1, 30: 1}

def test_get_fieldstat_mean(survey):
    # field 10 has skynoise 1 and 2
    skynoise = survey.get_fieldstat("mean", columns="skynoise")
    assert skynoise.to_dict() == {10: 1.5, 20: 3., 30: 4.}

def test_get_fieldstat_incl_zeros(survey):
    nobs = survey.get_fieldstat("size", incl_zeros=True)
    assert len(nobs) == 192
    assert nobs.loc[10] == 2
    assert np.isnan(nobs.loc[0]), "unobserved fields should be N/A"


# tests for the func get_fieldcoverage
def test_get_fieldcoverage(survey):
    assert survey.get_fieldcoverage().to_dict() == {10: 2, 20: 1, 30: 1}


# tests for the func get_polygons
def test_get_polygons_vertices():
    # 4 corners per pixel, (ra, dec) each
    survey = HealpixSurvey(nside=NSIDE)
    assert survey.get_polygons(as_vertices=True).shape == (192, 2, 4)

def test_get_polygons_observed(survey):
    polygons = survey.get_polygons(observed_fields=True)
    assert len(polygons) == 3
    assert all(isinstance(p_, geometry.Polygon) for p_ in polygons)

def test_get_polygons_contain_centroid(survey):
    # each pixel polygon contains its own centroid (pixels away from the ra=0/360 edge)
    fieldids = [10, 20, 30, 100]
    polygons = survey.get_polygons()
    radec = centroids_of(survey, fieldids)
    for fieldid_, (ra_, dec_) in zip(fieldids, radec.values):
        assert polygons[fieldid_].contains(geometry.Point(ra_, dec_)), f"field {fieldid_} does not contain its centroid"


# tests for the func get_skyarea
def test_get_skyarea_multipolygon(survey):
    skyarea = survey.get_skyarea(buffer=None)
    assert isinstance(skyarea, geometry.MultiPolygon)
    assert len(skyarea.geoms) == 3

def test_get_skyarea_contains_observed(survey):
    skyarea = survey.get_skyarea()
    for ra_, dec_ in centroids_of(survey, [10, 20, 30]).values:
        assert skyarea.contains(geometry.Point(ra_, dec_))
    # 5 is not observed
    ra_, dec_ = centroids_of(survey, [5]).values[0]
    assert not skyarea.contains(geometry.Point(ra_, dec_))


# tests for the func from_random
def test_from_random():
    survey = HealpixSurvey.from_random(nside=NSIDE, size=50, bands=["g", "r"],
                                       mjd_range=[59000, 59100], skynoise_range=[10, 20],
                                       rng=1234)
    assert len(survey.data) == 50
    assert survey.nside == NSIDE
    assert set(survey.data["band"]) <= {"g", "r"}
    assert survey.data["mjd"].between(59000, 59100).all()
    assert survey.data["skynoise"].between(10, 20).all()
    assert survey.data["mjd"].is_monotonic_increasing
    # default gain and zp of draw_random
    assert (survey.data["gain"] == 1).all()
    assert (survey.data["zp"] == 25).all()

def test_from_random_seed():
    kwargs = dict(nside=NSIDE, size=10, bands=["g", "r"],
                  mjd_range=[59000, 59100], skynoise_range=[10, 20])
    survey_1 = HealpixSurvey.from_random(rng=1234, **kwargs)
    survey_2 = HealpixSurvey.from_random(rng=1234, **kwargs)
    pandas.testing.assert_frame_equal(survey_1.data, survey_2.data)

def test_from_random_fieldid_are_pixels():
    survey = HealpixSurvey.from_random(nside=NSIDE, size=50, bands=["g"],
                                       mjd_range=[59000, 59100], skynoise_range=[10, 20],
                                       rng=1)
    fieldids = survey.data["fieldid"].to_numpy()
    assert np.all(fieldids == np.round(fieldids)), "fieldid must be integer healpix pixel ids"
    assert np.all((fieldids >= 0) & (fieldids < 12 * NSIDE**2))

def test_from_random_dec_range():
    survey = HealpixSurvey.from_random(nside=NSIDE, size=50, bands=["g"],
                                       mjd_range=[59000, 59100], skynoise_range=[10, 20],
                                       dec_range=[30, 90], rng=1)
    _, dec = hp.pix2ang(NSIDE, survey.data["fieldid"].to_numpy(), lonlat=True)
    assert np.all(dec >= 30), "drawn fields outside of dec_range"


# tests for the func draw_random
def test_draw_random_not_inplace():
    survey = HealpixSurvey(nside=NSIDE)
    newsurvey = survey.draw_random(10, ["g"], [0, 1], [1, 2], rng=1)
    assert survey.data is None, "inplace=False should not change the survey"
    assert isinstance(newsurvey, HealpixSurvey)
    assert len(newsurvey.data) == 10

def test_draw_random_inplace():
    survey = HealpixSurvey(nside=NSIDE)
    out = survey.draw_random(10, ["g"], [0, 1], [1, 2], inplace=True, rng=1)
    assert out is None
    assert len(survey.data) == 10

def test_draw_random_fixed_gain_zp():
    survey = HealpixSurvey(nside=NSIDE)
    newsurvey = survey.draw_random(10, ["g"], [0, 1], [1, 2],
                                   gain_range=2, zp_range=30, rng=1)
    assert (newsurvey.data["gain"] == 2).all()
    assert (newsurvey.data["zp"] == 30).all()

def test_draw_random_new_nside_inplace():
    # cannot change nside inplace: warns and returns a copy
    survey = HealpixSurvey(nside=NSIDE)
    with pytest.warns(UserWarning, match="nside"):
        newsurvey = survey.draw_random(10, ["g"], [0, 1], [1, 2],
                                       nside=8, inplace=True, rng=1)
    assert survey.data is None
    assert survey.nside == NSIDE
    assert newsurvey.nside == 8
    assert len(newsurvey.data) == 10

# tests for the func get_ipix_in_range
def test_get_ipix_in_range_no_range():
    assert np.array_equal(get_ipix_in_range(NSIDE), np.arange(12 * NSIDE**2))

def test_get_ipix_in_range_dec():
    ipix = get_ipix_in_range(NSIDE, dec_range=[30, 90])
    ra, dec = hp.pix2ang(NSIDE, np.arange(12 * NSIDE**2), lonlat=True)
    assert np.array_equal(ipix, np.flatnonzero(dec >= 30))
    # the cap above dec=30 is (1 - sin(30))/2 = 1/4 of the sky: a quarter of the
    # pixels up to the pixel rings crossing the boundary.
    assert np.isclose(len(ipix) / (12 * NSIDE**2), 0.25, atol=0.05)

def test_get_ipix_in_range_ra():
    ipix = get_ipix_in_range(NSIDE, ra_range=[0, 90])
    ra, _ = hp.pix2ang(NSIDE, ipix, lonlat=True)
    assert np.all((ra >= 0) & (ra <= 90))
    # pixels are evenly spread in ra: ~ a quarter of them.
    assert np.isclose(len(ipix) / (12 * NSIDE**2), 0.25, atol=0.05)

def test_get_ipix_in_range_both():
    ipix = get_ipix_in_range(NSIDE, ra_range=[0, 90], dec_range=[30, 90])
    expected = np.intersect1d(get_ipix_in_range(NSIDE, ra_range=[0, 90]),
                              get_ipix_in_range(NSIDE, dec_range=[30, 90]))
    assert np.array_equal(ipix, expected)

def test_get_ipix_in_range_in_rad():
    ipix_deg = get_ipix_in_range(NSIDE, ra_range=[0, 90], dec_range=[30, 90])
    ipix_rad = get_ipix_in_range(NSIDE, ra_range=np.radians([0, 90]),
                                 dec_range=np.radians([30, 90]), in_rad=True)
    assert np.array_equal(ipix_deg, ipix_rad)
