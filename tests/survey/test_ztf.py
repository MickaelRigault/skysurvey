import numpy as np
import pandas
import pytest
from shapely import geometry

from skysurvey.survey.ztf import ZTF


@pytest.fixture
def logs():
    """ two observations of two different ccds """
    return pandas.DataFrame({"mjd": [2., 1.],
                             "band": ["ztfg", "ztfr"],
                             "skynoise": [1., 2.],
                             "gain": [1., 1.],
                             "zp": [30., 30.],
                             "fieldid": [396, 500],
                             "ccdid": [15, 3]})

@pytest.fixture
def survey(logs):
    # ccd level is ~5 times faster to load than quadrant
    return ZTF(data=logs, level="ccd")


# tests for the class ZTF
def test_ztf_ccd_fieldids(survey):
    assert survey.fieldids.names == ["fieldid", "ccdid"]

def test_ztf_quadrant_fieldids():
    survey = ZTF(level="quadrant")
    assert survey.fieldids.names == ["fieldid", "rcid"]
    # 4 quadrants per ccd
    assert survey.nfields == 4 * ZTF(level="ccd").nfields

def test_ztf_ccd_per_field(survey):
    # 16 ccds per field
    assert survey.fields.loc[396].shape[0] == 16

def test_ztf_footprint(survey):
    # ZTF camera: ~47 deg2 field of view
    assert np.isclose(survey.footprint.area, 47, rtol=0.03)


# tests for the func from_pointings
def test_from_pointings(logs):
    survey = ZTF.from_pointings(logs, level="ccd")
    assert isinstance(survey, ZTF)
    assert len(survey.data) == 2

def test_from_pointings_dict(logs):
    survey = ZTF.from_pointings(logs.to_dict(orient="list"), level="ccd")
    assert len(survey.data) == 2


# tests for the func radec_to_fieldid
def test_radec_to_fieldid_in_polygon(survey):
    # the matched ccd(s) contain the input coordinate
    fields = survey.radec_to_fieldid([[0., 0.]])
    assert len(fields) >= 1
    for fieldid_, ccdid_ in fields.values:
        assert survey.fields.loc[(fieldid_, ccdid_), "geometry"].intersects(geometry.Point(0, 0))

def test_radec_to_fieldid_centroid(survey):
    # the centroid of a ccd polygon is matched to that ccd
    centroid = survey.fields.loc[(396, 15), "geometry"].centroid
    fields = survey.radec_to_fieldid([[centroid.x, centroid.y]])
    assert (396, 15) in [tuple(f_) for f_ in fields.values]


# tests for the func get_fieldcoverage
def test_get_fieldcoverage(survey):
    nobs = survey.get_fieldcoverage()
    assert nobs.loc[(396, 15)] == 1
    assert nobs.loc[(500, 3)] == 1
