import numpy as np
import pandas
import geopandas
import pytest
from shapely import geometry

from skysurvey.survey.polygon import PolygonSurvey
from skysurvey.survey.healpix import HealpixSurvey


@pytest.fixture
def fields():
    """ three 2x2 deg boxes, far from each other """
    # field 1 is offset from ra=0 and dec=0 where healpix pixel centroids lie on its edges
    return geopandas.GeoDataFrame(geometry=[geometry.box(0.3, 0.3, 2.3, 2.3),
                                            geometry.box(10, 0, 12, 2),
                                            geometry.box(20, 20, 22, 22)],
                                  index=pandas.Index([1, 2, 3], name="fieldid"))

@pytest.fixture
def logs():
    """ field 1 observed twice, field 2 once, field 3 never """
    return pandas.DataFrame({"mjd": [3., 1., 2.],
                             "band": ["g", "r", "g"],
                             "skynoise": [1., 2., 3.],
                             "gain": [1., 1., 1.],
                             "zp": [25., 25., 25.],
                             "fieldid": [1, 1, 2]})

@pytest.fixture
def survey(logs, fields):
    return PolygonSurvey(data=logs, fields=fields)

@pytest.fixture
def pointings():
    """ two pointings on the equator, one at dec=30 """
    return pandas.DataFrame({"mjd": [1., 2., 3.],
                             "band": ["g", "r", "g"],
                             "skynoise": [1., 2., 3.],
                             "gain": [1., 1., 1.],
                             "zp": [25., 25., 25.],
                             "ra": [10., 10., 100.],
                             "dec": [0., 0., 30.]})

SQUARE = geometry.box(-1, -1, 1, 1) # 2x2 deg camera centered on (0, 0)


# tests for the class PolygonSurvey
def test_init_no_fields():
    # no default fields for the base class
    with pytest.raises(NotImplementedError):
        PolygonSurvey()

def test_init(survey):
    assert survey.nfields == 3
    assert list(survey.fieldids) == [1, 2, 3]
    assert survey.fieldids.name == "fieldid"
    assert len(survey.data) == 3

def test_init_no_data(fields):
    survey = PolygonSurvey(fields=fields)
    assert survey.data is None
    assert survey.nfields == 3


# tests for the func get_fields
def test_get_fields_observed(survey):
    assert list(survey.get_fields().index) == [1, 2]

def test_get_fields_all(survey):
    assert list(survey.get_fields(observed=False).index) == [1, 2, 3]

def test_get_fields_is_copy(survey):
    fields = survey.get_fields(observed=False)
    fields.loc[1, "geometry"] = geometry.box(50, 50, 51, 51)
    assert survey.fields.loc[1, "geometry"].equals(geometry.box(0.3, 0.3, 2.3, 2.3))


# tests for the func radec_to_fieldid
def test_radec_to_fieldid_inside(survey):
    # each point is at the center of one box.
    fields = survey.radec_to_fieldid([[1, 1], [11, 1], [21, 21]])
    assert list(fields["fieldid"]) == [1, 2, 3]
    assert list(fields.index) == [0, 1, 2]
    assert fields.index.name == "index_radec"

def test_radec_to_fieldid_outside(survey):
    # (50, 50) is in no field: dropped from the output
    fields = survey.radec_to_fieldid([[1, 1], [50, 50]])
    assert list(fields["fieldid"]) == [1]
    assert list(fields.index) == [0]

def test_radec_to_fieldid_dataframe(survey):
    radec = pandas.DataFrame({"ra": [11., 1.], "dec": [1., 1.]}, index=[5, 7])
    fields = survey.radec_to_fieldid(radec)
    assert fields.loc[5, "fieldid"] == 2
    assert fields.loc[7, "fieldid"] == 1

def test_radec_to_fieldid_observed_fields(survey):
    # field 3 is not observed
    fields = survey.radec_to_fieldid([[1, 1], [11, 1], [21, 21]], observed_fields=True)
    assert list(fields["fieldid"]) == [1, 2]

def test_radec_to_fieldid_overlapping(fields):
    # a point in two overlapping fields is matched to both
    fields = pandas.concat([fields,
                            geopandas.GeoDataFrame(geometry=[geometry.box(1, 1, 3, 3)],
                                                   index=pandas.Index([4], name="fieldid"))])
    survey = PolygonSurvey(fields=fields)
    matched = survey.radec_to_fieldid([[1.5, 1.5]], observed_fields=False)
    assert sorted(matched["fieldid"]) == [1, 4]

def test_radec_to_fieldid_badshape(survey):
    with pytest.raises(ValueError):
        survey.radec_to_fieldid([[1, 1, 1]])


# tests for the func get_observations_from_coords
def test_get_observations_from_coords(survey):
    obs = survey.get_observations_from_coords([[1, 1]])
    assert list(obs["fieldid"]) == [1, 1]
    assert list(obs["mjd"]) == [1., 3.]

def test_get_observations_from_coords_unobserved(survey):
    # (21, 21) is in field 3, never observed
    assert len(survey.get_observations_from_coords([[21, 21]])) == 0


# tests for the func get_fieldstat
def test_get_fieldstat_size(survey):
    assert survey.get_fieldstat("size").to_dict() == {1: 2, 2: 1}

def test_get_fieldstat_incl_zeros(survey):
    nobs = survey.get_fieldcoverage(incl_zeros=True)
    assert list(nobs.index) == [1, 2, 3]
    assert np.isnan(nobs.loc[3])


# tests for the func from_pointings
def test_from_pointings_one_field_per_pointing(pointings):
    survey = PolygonSurvey.from_pointings(pointings, footprint=SQUARE)
    assert survey.nfields == 3
    assert list(survey.data["fieldid"]) == [0, 1, 2]

def test_from_pointings_dict(pointings):
    survey = PolygonSurvey.from_pointings(pointings.to_dict(orient="list"), footprint=SQUARE)
    assert survey.nfields == 3

def test_from_pointings_field_contains_pointing(pointings):
    # the footprint is centered on (0, 0), so each pointing is at the center of its field.
    survey = PolygonSurvey.from_pointings(pointings, footprint=SQUARE)
    for fieldid_, (ra_, dec_) in zip(survey.data["fieldid"], pointings[["ra", "dec"]].values):
        assert survey.fields.loc[fieldid_, "geometry"].contains(geometry.Point(ra_, dec_))

def test_from_pointings_dec_size(pointings):
    # the footprint extends +/-1 deg in dec around the pointing
    survey = PolygonSurvey.from_pointings(pointings, footprint=SQUARE)
    field = survey.fields.loc[2, "geometry"] # pointing at (100, 30)
    assert field.contains(geometry.Point(100, 30.9))
    assert not field.contains(geometry.Point(100, 31.1))

def test_from_pointings_radec_to_fieldid(pointings):
    # (10, 0) observed by the first 2 pointings
    survey = PolygonSurvey.from_pointings(pointings, footprint=SQUARE)
    fields = survey.radec_to_fieldid([[10, 0.5], [100, 30], [50, 0]])
    assert fields.loc[0, "fieldid"].tolist() == [0, 1]
    assert fields.loc[1, "fieldid"] == 2
    assert 2 not in fields.index


# tests for the func from_random
def test_from_random(fields):
    survey = PolygonSurvey.from_random(20, ["g", "r"], [59000, 59010], [1, 2], fields=fields)
    assert len(survey.data) == 20
    assert set(survey.data["fieldid"]) <= {1, 2, 3}
    assert set(survey.data["band"]) <= {"g", "r"}
    assert survey.data["mjd"].between(59000, 59010).all()
    assert survey.data["skynoise"].between(1, 2).all()
    assert survey.data["mjd"].is_monotonic_increasing

def test_from_random_fieldids(fields):
    # only draw in field 2
    survey = PolygonSurvey.from_random(20, ["g"], [59000, 59010], [1, 2],
                                       fields=fields, fieldids=[2])
    assert (survey.data["fieldid"] == 2).all()


# tests for the func to_healpix
def test_to_healpix(survey):
    nside = 64 # ~0.84 deg2 pixels: a few pixels per 2x2 deg box
    hpsurvey = survey.to_healpix(nside, backend="pandas")
    assert isinstance(hpsurvey, HealpixSurvey)
    assert hpsurvey.nside == nside
    # input fieldid kept as fieldid_survey ; only observed fields 1 and 2
    assert set(hpsurvey.data["fieldid_survey"]) == {1, 2}

def test_to_healpix_pixels_in_field(survey):
    # healpix pixels associated to field 1 are those whose centroid is in its box
    nside = 64
    hpsurvey = survey.to_healpix(nside, backend="pandas")
    ra, dec = hpsurvey.get_field_centroid()
    expected = np.flatnonzero((ra > 0.3) & (ra < 2.3) & (dec > 0.3) & (dec < 2.3))
    assert len(expected) > 0
    field1 = hpsurvey.data[hpsurvey.data["fieldid_survey"] == 1]
    assert set(field1["fieldid"]) == set(expected)
    # field 1 observed twice: each pixel twice
    assert len(field1) == 2 * len(expected)

def test_to_healpix_unknown_backend(survey):
    with pytest.raises(NotImplementedError):
        survey.to_healpix(64, backend="unknown")
