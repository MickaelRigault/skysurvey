import numpy as np
import pytest
from shapely import geometry

from skysurvey.survey.roman import get_roman_footprint, parse_simlib, parse_simlib_block

SIMLIB = """BEGIN LIBGEN
LIBID: 1
RA: 10.0  DEC: -5.0
# Below data READ by snana
#  MJD  BAND  SKYSIG
S: 60000.0 R 12.5 # first
S: 60001.0 Z 13.5 # second
END_LIBID: 1
LIBID: 2
RA: 20.0  DEC: -6.0
# Below data READ by snana
#  MJD  BAND  SKYSIG
S: 60002.0 R 14.5 # third
END_LIBID: 2
"""

@pytest.fixture
def simlib_path(tmp_path):
    """ tiny SIMLIB file with 2 blocks (2 and 1 observations) """
    filepath = tmp_path / "roman.simlib"
    filepath.write_text(SIMLIB)
    return str(filepath)


# tests for the func get_roman_footprint
def test_get_roman_footprint():
    footprint = get_roman_footprint()
    assert isinstance(footprint, geometry.Polygon)
    # 0.281 deg2 field of view (see docstring)
    assert np.isclose(footprint.area, 0.281, rtol=0.01)


# tests for the func parse_simlib
def test_parse_simlib_data(simlib_path):
    data, metadata = parse_simlib(simlib_path)
    # one entry per observation, level 0 is the block index
    assert len(data) == 3
    assert list(data.index.get_level_values(0)) == [0, 0, 1]
    assert list(data.columns) == ["case", "mjd", "band", "skysig", "comments"]
    assert list(data["mjd"].astype(float)) == [60000., 60001., 60002.]
    assert list(data["band"]) == ["R", "Z", "R"]
    assert list(data["comments"]) == ["first", "second", "third"]

def test_parse_simlib_metadata(simlib_path):
    data, metadata = parse_simlib(simlib_path)
    assert len(metadata) == 2
    assert list(metadata["libid"]) == ["1", "2"]
    assert list(metadata["ra"].astype(float)) == [10., 20.]
    assert list(metadata["dec"].astype(float)) == [-5., -6.]

def test_parse_simlib_no_libgen(tmp_path):
    filepath = tmp_path / "bad.simlib"
    filepath.write_text(SIMLIB.replace("BEGIN LIBGEN", ""))
    with pytest.raises(ValueError):
        parse_simlib(str(filepath))


# tests for the func parse_simlib_block
def test_parse_simlib_block_no_read():
    with pytest.raises(ValueError):
        parse_simlib_block(["LIBID: 1", "S: 60000.0 R 12.5 # first"])

def test_parse_simlib_block_multiple_read():
    block = ["# data READ by snana", "# MJD BAND", "# data READ by snana", "# MJD BAND"]
    with pytest.raises(ValueError):
        parse_simlib_block(block)
