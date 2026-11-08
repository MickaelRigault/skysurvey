import numpy as np
import pandas
import pytest
from shapely import geometry

from skysurvey.survey.snls import get_snls_footprint, get_snls_field_coordinates, SNLS


# tests for the func get_snls_footprint
def test_get_snls_footprint():
    # MegaCam: 1 deg side square centered on 0
    footprint = get_snls_footprint()
    assert np.isclose(footprint.area, 1.)
    assert footprint.bounds == (-0.5, -0.5, 0.5, 0.5)


# tests for the func get_snls_field_coordinates
def test_get_snls_field_coordinates():
    coords = get_snls_field_coordinates()
    assert list(coords.index) == ["D1", "D2", "D3", "D4"]
    assert coords.index.name == "fieldid"
    assert np.isclose(coords.loc["D3", "dec"], 52.666)

def test_get_snls_field_coordinates_name():
    assert get_snls_field_coordinates(fieldid_name="FIELD").index.name == "FIELD"


# tests for the class SNLS
# SNLS() registers the megacampsf bandpasses, downloaded by sncosmo if not cached.
@pytest.fixture
def snls():
    return SNLS()

@pytest.mark.remote_data
def test_snls_fields(snls):
    assert snls.nfields == 4
    assert snls.footprint.equals(get_snls_footprint())

@pytest.mark.remote_data
def test_snls_radec_to_fieldid(snls):
    # each field center is in its own field
    coords = get_snls_field_coordinates()
    fields = snls.radec_to_fieldid(coords[["ra", "dec"]].astype(float).values)
    assert list(fields["fieldid"]) == ["D1", "D2", "D3", "D4"]

@pytest.mark.remote_data
def test_snls_from_pointings():
    data = {"mjd": [1., 2.], "band": ["megacampsf::g", "megacampsf::r"],
            "skynoise": [1., 2.], "gain": [1., 1.], "zp": [30., 30.],
            "fieldid": ["D1", "D3"]}
    survey = SNLS.from_pointings(data)
    assert isinstance(survey, SNLS)
    assert len(survey.data) == 2
    obs = survey.get_observations_from_coords([[214.90738, 52.6660]])
    assert list(obs["fieldid"]) == ["D3"]
