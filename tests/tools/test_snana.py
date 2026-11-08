import numpy as np
import pandas
import pytest

from skysurvey.tools.snana import (parse_simlib, parse_simlib_block,
                                   parse_simlib_des, parse_simlib_block_des)


# a minimal 2-LIBID simlib, in the format expected by the parser:
# metadata "KEY: value" lines, a line containing " READ ", then the column names
# and data lines of the form "S: values # comment".
SIMLIB = """DOCUMENTATION:
  PURPOSE: test
DOCUMENTATION_END:

BEGIN LIBGEN

LIBID: 1
RA: 10.5   DECL: -5.0   NOBS: 2
#                           CCD  READ  NOISE
#     MJD      IDEXPT  FLT GAIN NOISE SKYSIG ZPTAVG
S: 53000.1   11  g 1.0 0.5 10.0 30.0 # first
S: 53001.2   12  r 1.5 0.6 20.0 31.0 # second
END_LIBID: 1

LIBID: 2
RA: 20.5   DECL: 5.0   NOBS: 1
#                           CCD  READ  NOISE
#     MJD      IDEXPT  FLT GAIN NOISE SKYSIG ZPTAVG
S: 53010.3   13  i 2.0 0.7 30.0 32.0 # third
END_LIBID: 2
"""

COLUMNS = ["case", "mjd", "idexpt", "flt", "gain", "noise", "skysig", "zptavg", "comments"]


@pytest.fixture
def simlib_file(tmp_path):
    filename = tmp_path / "test.simlib"
    filename.write_text(SIMLIB)
    return str(filename)

@pytest.fixture
def block():
    return SIMLIB.splitlines()[5:12] # from "BEGIN LIBGEN" to before "END_LIBID: 1"


# tests for the func parse_simlib_block
def test_parse_simlib_block_data(block):
    data, meta = parse_simlib_block(block)

    assert list(data.columns) == COLUMNS
    assert len(data) == 2
    assert list(data["case"]) == ["S", "S"]
    assert list(data["mjd"]) == ["53000.1", "53001.2"] # parsed as strings
    assert list(data["flt"]) == ["g", "r"]
    assert list(data["comments"]) == ["first", "second"]

def test_parse_simlib_block_meta(block):
    data, meta = parse_simlib_block(block)
    # keys are lower-cased, LIBGEN line ignored.
    assert meta.to_dict() == {"libid": "1", "ra": "10.5", "decl": "-5.0", "nobs": "2"}

def test_parse_simlib_block_noread(block):
    block = [line for line in block if " READ" not in line]
    with pytest.raises(ValueError):
        parse_simlib_block(block)

def test_parse_simlib_block_multiple_read(block):
    block = block[:3] + ["#   READ  "] + block[3:]
    with pytest.raises(ValueError):
        parse_simlib_block(block)


# tests for the func parse_simlib
def test_parse_simlib(simlib_file):
    data, metadata = parse_simlib(simlib_file)

    # one key per LIBID block
    assert list(data.index.get_level_values(0).unique()) == [0, 1]
    assert list(data.columns) == COLUMNS
    assert list(data.xs(0)["flt"]) == ["g", "r"]
    assert list(data.xs(1)["flt"]) == ["i"]
    assert list(data.xs(1)["mjd"]) == ["53010.3"]

    assert list(metadata.index) == [0, 1]
    assert list(metadata["libid"]) == ["1", "2"]
    assert list(metadata["ra"]) == ["10.5", "20.5"]
    assert list(metadata["nobs"]) == ["2", "1"]

def test_parse_simlib_no_libgen(tmp_path):
    filename = tmp_path / "bad.simlib"
    filename.write_text(SIMLIB.replace("BEGIN LIBGEN", ""))
    with pytest.raises(ValueError):
        parse_simlib(str(filename))

def test_parse_simlib_des(simlib_file):
    # same format => same output
    data, metadata = parse_simlib(simlib_file)
    data_des, metadata_des = parse_simlib_des(simlib_file)
    pandas.testing.assert_frame_equal(data, data_des)
    pandas.testing.assert_frame_equal(metadata, metadata_des)

# tests for the func parse_simlib_block_des
def test_parse_simlib_block_des(block):
    # same format => same output as parse_simlib_block
    data, meta = parse_simlib_block(block)
    data_des, meta_des = parse_simlib_block_des(block)
    pandas.testing.assert_frame_equal(data, data_des)
    pandas.testing.assert_series_equal(meta, meta_des)

def test_parse_simlib_block_des_noread(block):
    block = [line for line in block if " READ" not in line]
    with pytest.raises(ValueError):
        parse_simlib_block_des(block)
