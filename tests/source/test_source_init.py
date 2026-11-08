import numpy as np
import pytest

from skysurvey.source import get_sncosmo_sourcenames, SNCOSMO_SOURCES_DF


# tests for the func get_sncosmo_sourcenames
def test_get_sncosmo_sourcenames_all():
    names = get_sncosmo_sourcenames()
    assert isinstance(names, list)
    assert names == list(SNCOSMO_SOURCES_DF["name"])

def test_get_sncosmo_sourcenames_of_type():
    names = get_sncosmo_sourcenames(of_type="SN Ia")
    assert "salt2" in names
    assert len(names) > 0
    types = SNCOSMO_SOURCES_DF[SNCOSMO_SOURCES_DF["name"].isin(names)]["type"]
    assert set(types) == {"SN Ia"}

def test_get_sncosmo_sourcenames_of_type_list():
    names_ia = get_sncosmo_sourcenames(of_type="SN Ia")
    names_iip = get_sncosmo_sourcenames(of_type="SN IIP")
    names_both = get_sncosmo_sourcenames(of_type=["SN Ia", "SN IIP"])
    assert set(names_both) == set(names_ia) | set(names_iip)

def test_get_sncosmo_sourcenames_unknown_type():
    assert get_sncosmo_sourcenames(of_type="not_a_type") == []

def test_get_sncosmo_sourcenames_startswith_and_endswith():
    names = get_sncosmo_sourcenames(of_type="SN II", startswith="v19", endswith="corr")
    assert len(names) > 0
    assert all(name.startswith("v19") and name.endswith("corr") for name in names)

def test_get_sncosmo_sourcenames_startswith_selects():
    """ v19-1987a-corr also ends with 'corr' but must be excluded by startswith """
    names = get_sncosmo_sourcenames(of_type="SN II", startswith="v19-asassn", endswith="corr")
    assert sorted(names) == ["v19-asassn14jb-corr", "v19-asassn15oz-corr"]
