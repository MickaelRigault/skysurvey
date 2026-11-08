import sqlite3

import numpy as np
import pandas
import pytest
from shapely import geometry

from skysurvey.survey.lsst import get_lsst_footprint, read_opsim, LSST


@pytest.fixture
def opsim_path(tmp_path):
    """ tiny OpSim-like sqlite database: 3 visits, two at the same place """
    observations = pandas.DataFrame({"fieldRA": [10.3, 10.3, 100.3],
                                     "fieldDec": [-20.3, -20.3, -30.3],
                                     "observationStartMJD": [60001., 60000., 60002.],
                                     "visitExposureTime": [30., 30., 30.],
                                     "filter": ["g", "r", "i"],
                                     "skyBrightness": [21., 20., 19.],
                                     "fiveSigmaDepth": [25., 24., 23.],
                                     "night": [1, 0, 2],
                                     "numExposures": [1, 1, 1],
                                     "observationId": [11, 10, 12],
                                     "scheduler_note": ["a", "b", "c"]})
    filepath = tmp_path / "opsim.db"
    connect = sqlite3.connect(filepath)
    observations.to_sql("OBSERVATIONS", connect, index=False)
    connect.close()
    return str(filepath)


# tests for the func get_lsst_footprint
def test_get_lsst_footprint():
    footprint = get_lsst_footprint()
    assert isinstance(footprint, geometry.Polygon)
    # 9.6 deg2 field of view (see docstring)
    assert np.isclose(footprint.area, 9.6, rtol=0.01)

def test_get_lsst_footprint_centered():
    # symmetric (3 5 5 5 3) CCD structure centered on 0
    centroid = get_lsst_footprint().centroid
    assert np.isclose(centroid.x, 0) and np.isclose(centroid.y, 0)


# tests for the class LSST
def test_lsst_default_footprint():
    survey = LSST(nside=16)
    assert survey.footprint.equals(get_lsst_footprint())

def test_lsst_from_pointings():
    data = pandas.DataFrame({"mjd": [1., 2.], "band": ["lsstg", "lsstr"],
                             "skynoise": [1., 2.], "gain": [1., 1.], "zp": [30., 30.],
                             "ra": [10.3, 100.3], "dec": [-20.3, -30.3]})
    survey = LSST.from_pointings(data, nside=32, backend="pandas")
    assert isinstance(survey, LSST)
    assert survey.footprint.equals(get_lsst_footprint())
    # pointing centers are observed
    obs = survey.get_observations_from_coords(pandas.DataFrame({"ra": [10.3], "dec": [-20.3]}))
    assert list(obs["mjd"]) == [1.]


# tests for the func read_opsim
def test_read_opsim(opsim_path):
    df = read_opsim(opsim_path)
    assert len(df) == 3
    # default columns (scheduler_note is not loaded)
    assert "scheduler_note" not in df
    assert list(df["observationId"]) == [11, 10, 12]

def test_read_opsim_sql_where(opsim_path):
    df = read_opsim(opsim_path, sql_where="night<2")
    assert sorted(df["observationId"]) == [10, 11]

def test_read_opsim_all_columns(opsim_path):
    df = read_opsim(opsim_path, columns=None)
    assert "scheduler_note" in df
    assert df.shape == (3, 11)

def test_read_opsim_note_renamed(opsim_path):
    # 'note' does not exist in this db, 'scheduler_note' is used instead
    df = read_opsim(opsim_path, columns=["observationId", "note"])
    assert list(df.columns) == ["observationId", "scheduler_note"]


# tests for the func from_opsim
def test_from_opsim(opsim_path):
    survey = LSST.from_opsim(opsim_path, nside=32)
    assert isinstance(survey, LSST)
    # sorted by mjd
    assert list(survey.data.groupby("observationId")["mjd"].first()) == [60000., 60001., 60002.]
    assert set(survey.data["band"]) == {"lsstg", "lsstr", "lssti"}
    assert (survey.data["zp"] == 30).all()
    assert (survey.data["gain"] == 1).all()

def test_from_opsim_skynoise(opsim_path):
    # 5 sigma depth of 25 at zp=30: flux_5sigma = 10**(0.4*5) = 100 => skynoise = 20
    survey = LSST.from_opsim(opsim_path, nside=32, zp=30)
    skynoise = survey.data.groupby("observationId")["skynoise"].first()
    assert np.isclose(skynoise.loc[11], 20)
    # depth of 24 is 1 mag shallower: 10**0.4 times noisier
    assert np.isclose(skynoise.loc[10], 20 * 10**0.4)
