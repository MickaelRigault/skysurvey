import numpy as np
import pandas
import geopandas
import pytest
from shapely import geometry

from skysurvey.survey import get_footprint
from skysurvey.survey.basesurvey import Survey, GridSurvey
from skysurvey.survey.healpix import HealpixSurvey

SQUARE = geometry.box(-1, -1, 1, 1) # 2x2 deg camera centered on (0, 0)


@pytest.fixture
def pointings():
    """ two pointings at the same place, one at dec=30 """
    return pandas.DataFrame({"mjd": [1., 2., 3.],
                             "band": ["g", "r", "g"],
                             "skynoise": [1., 2., 3.],
                             "gain": [1., 1., 1.],
                             "zp": [25., 25., 25.],
                             "ra": [10.3, 10.3, 100.3],
                             "dec": [0.3, 0.3, 30.3]})

@pytest.fixture
def logs():
    """ observing logs for a grid survey of 2 fields """
    return pandas.DataFrame({"mjd": [1., 2., 3.],
                             "band": ["g", "r", "g"],
                             "skynoise": [1., 2., 3.],
                             "gain": [1., 1., 1.],
                             "zp": [25., 25., 25.],
                             "fieldid": [0, 0, 1]})

@pytest.fixture
def field_coords():
    """ field centers (in deg) as a dict of dict """
    return {0: {"ra": 10., "dec": 0.}, 1: {"ra": 100., "dec": 30.}}


# tests for the class Survey
def test_survey_init():
    survey = Survey(footprint=SQUARE, nside=16)
    assert isinstance(survey, HealpixSurvey)
    assert survey.nside == 16
    assert survey.footprint.equals(SQUARE)
    assert survey.data is None

def test_survey_no_footprint():
    assert Survey(nside=16).footprint is None

def test_survey_from_random_not_implemented():
    with pytest.raises(NotImplementedError):
        Survey.from_random(nside=16, size=10)


# tests for the func from_data
def test_survey_from_data(logs):
    survey = Survey.from_data(logs, footprint=SQUARE, nside=16)
    assert survey.nside == 16
    assert survey.footprint.equals(SQUARE)
    assert len(survey.data) == 3


# tests for the func from_healpix
def test_survey_from_healpix(logs):
    hpsurvey = HealpixSurvey(nside=16, data=logs)
    survey = Survey.from_healpix(hpsurvey, footprint=SQUARE)
    assert isinstance(survey, Survey)
    assert survey.nside == 16
    pandas.testing.assert_frame_equal(survey.data, hpsurvey.data)


# tests for the func from_pointings
def test_survey_from_pointings(pointings):
    nside = 64 # ~0.84 deg2 pixels: a few pixels per 4 deg2 pointing
    survey = Survey.from_pointings(pointings, footprint=SQUARE, nside=nside, backend="pandas")
    assert isinstance(survey, Survey)
    assert survey.nside == nside
    assert survey.footprint.equals(SQUARE)
    # each pointing is a polygon field, stored as fieldid_survey
    assert set(survey.data["fieldid_survey"]) == {0, 1, 2}

def test_survey_from_pointings_pixels(pointings):
    # the healpix pixels of a pointing have their centroid within the projected footprint
    nside = 64
    survey = Survey.from_pointings(pointings, footprint=SQUARE, nside=nside, backend="pandas")
    ra, dec = survey.get_field_centroid()
    expected = np.flatnonzero((np.abs(ra - 10.3) < 1) & (np.abs(dec - 0.3) < 1))
    pixels = survey.data[survey.data["fieldid_survey"] == 0]["fieldid"]
    assert set(pixels) == set(expected)

def test_survey_from_pointings_observations(pointings):
    # the position of the first two pointings has 2 observations
    survey = Survey.from_pointings(pointings, footprint=SQUARE, nside=64, backend="pandas")
    obs = survey.get_observations_from_coords(pandas.DataFrame({"ra": [10.3], "dec": [0.3]}))
    assert sorted(obs["mjd"]) == [1., 2.]


# tests for the class GridSurvey
def test_gridsurvey_no_fields():
    with pytest.raises(NotImplementedError):
        GridSurvey()

def test_gridsurvey_from_logs_not_implemented():
    with pytest.raises(NotImplementedError):
        GridSurvey.from_logs()


# tests for the func from_pointings (GridSurvey)
def test_gridsurvey_from_pointings_coords(logs, field_coords):
    survey = GridSurvey.from_pointings(logs, fields_or_coords=field_coords, footprint=SQUARE)
    assert survey.nfields == 2
    assert list(survey.fieldids) == [0, 1]
    assert survey.footprint.equals(SQUARE)
    # input coordinates are stored with the fields
    assert list(survey.fields["ra"]) == [10., 100.]

def test_gridsurvey_from_pointings_dataframe(logs, field_coords):
    coords = pandas.DataFrame(field_coords).T
    survey = GridSurvey.from_pointings(logs, fields_or_coords=coords, footprint=SQUARE)
    assert survey.fieldids.name == "fieldid", "unnamed index should be named fieldid"
    assert survey.nfields == 2

def test_gridsurvey_from_pointings_dict_data(logs, field_coords):
    survey = GridSurvey.from_pointings(logs.to_dict(orient="list"),
                                       fields_or_coords=field_coords, footprint=SQUARE)
    assert len(survey.data) == 3

def test_gridsurvey_from_pointings_geodataframe(logs):
    fields = geopandas.GeoDataFrame(geometry=[geometry.box(9, -1, 11, 1), geometry.box(99, 29, 101, 31)],
                                    index=pandas.Index([0, 1], name="fieldid"))
    survey = GridSurvey.from_pointings(logs, fields_or_coords=fields)
    assert survey.nfields == 2
    assert survey.footprint is None

def test_gridsurvey_coords_no_footprint(logs, field_coords):
    with pytest.raises(ValueError):
        GridSurvey.from_pointings(logs, fields_or_coords=field_coords)

def test_gridsurvey_radec_to_fieldid(logs, field_coords):
    survey = GridSurvey.from_pointings(logs, fields_or_coords=field_coords, footprint=SQUARE)
    # field centers and a far away point
    fields = survey.radec_to_fieldid([[10, 0], [100, 30], [50, 50]])
    assert list(fields["fieldid"]) == [0, 1]
    assert list(fields.index) == [0, 1]

def test_gridsurvey_field_size(logs, field_coords):
    # the footprint extends +/-1 deg in dec around the field center
    survey = GridSurvey.from_pointings(logs, fields_or_coords=field_coords, footprint=SQUARE)
    field = survey.fields.loc[1, "geometry"]
    assert field.contains(geometry.Point(100, 30.9))
    assert not field.contains(geometry.Point(100, 31.1))


# tests for the func get_skyarea (GridSurvey)
def test_gridsurvey_get_skyarea(logs, field_coords):
    survey = GridSurvey.from_pointings(logs, fields_or_coords=field_coords, footprint=SQUARE)
    skyarea = survey.get_skyarea()
    assert isinstance(skyarea, geometry.MultiPolygon)
    assert len(skyarea.geoms) == 2
    assert skyarea.contains(geometry.Point(10, 0))
    assert not skyarea.contains(geometry.Point(50, 50))

def test_gridsurvey_get_skyarea_list(logs, field_coords):
    survey = GridSurvey.from_pointings(logs, fields_or_coords=field_coords, footprint=SQUARE)
    assert len(survey.get_skyarea(as_multipolygon=False)) == 2


# tests for the func get_footprint
def test_get_footprint_lsst():
    from skysurvey.survey.lsst import get_lsst_footprint
    assert get_footprint("lsst").equals(get_lsst_footprint())

def test_get_footprint_case():
    from skysurvey.survey.des import get_des_footprint
    assert get_footprint("DES").equals(get_des_footprint())

def test_get_footprint_kwargs():
    from skysurvey.survey.des import get_des_footprint
    assert get_footprint("des", incl_focus=True).equals(get_des_footprint(incl_focus=True))
