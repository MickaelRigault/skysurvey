import numpy as np
import pandas
import pytest
from shapely import geometry

from skysurvey.survey.des import (get_des_footprint, get_des_field_coordinates,
                                  get_des_fields, DES, DESWide)


# tests for the func get_des_footprint
def test_get_des_footprint():
    footprint = get_des_footprint()
    assert footprint.is_valid
    # DECam: ~3 deg2 field of view
    assert 2.5 < footprint.area < 3.5

def test_get_des_footprint_incl_focus():
    # adding the focus CCDs increases the area
    assert get_des_footprint(incl_focus=True).area > get_des_footprint().area

def test_get_des_footprint_coef():
    # pixel-to-degree coefficients scale the area as 1/(coef_x * coef_y)
    area = get_des_footprint(coef=(1, 1)).area
    assert np.isclose(get_des_footprint(coef=(2, 4)).area, area / 8)


# tests for the func get_des_field_coordinates
def test_get_des_field_coordinates():
    coords = get_des_field_coordinates()
    # 8 shallow and 2 deep fields
    assert len(coords) == 10
    assert coords.index.name == "fieldid"
    assert set(coords.columns) == {"ra", "dec"}
    assert np.isclose(coords.loc["S1", "ra"], 42.82)
    assert np.isclose(coords.loc["S1", "dec"], 0.)

def test_get_des_field_coordinates_name():
    assert get_des_field_coordinates(fieldid_name="FIELD").index.name == "FIELD"
    assert get_des_field_coordinates(fieldid_name=None).index.name == "fieldid"


# tests for the func get_des_fields
def test_get_des_fields():
    fields = get_des_fields()
    assert len(fields) == 10
    assert list(fields.index) == list(get_des_field_coordinates().index)

def test_get_des_fields_area():
    # projection near the equator barely changes the footprint area (S1 is at dec=0)
    fields = get_des_fields()
    assert np.isclose(fields.loc["S1", "geometry"].area, get_des_footprint().area, rtol=1e-3)


# tests for the class DES
def test_des_default():
    survey = DES()
    assert survey.nfields == 10
    assert survey.fieldids.name == "FIELD"
    assert survey.footprint.equals(get_des_footprint())


# tests for the class DESWide
def test_deswide_default_footprint():
    survey = DESWide(nside=16)
    assert survey.footprint.equals(get_des_footprint())
