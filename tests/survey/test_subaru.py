import warnings

import numpy as np
import pandas
import pytest
import shapely
from shapely import geometry

from skysurvey.survey.subaru import (HSC, get_hsc_ccd_footprints, get_hsc_footprint,
                                     get_hsc_fields, parse_hsc_logs)
from skysurvey.survey import subaru


# HSC camera, from the instrument pages: 2048 x 4096 pixels of 0.17 arcsec,
# gaps of 12 arcsec (within a row) and 53 arcsec (between rows).
CCD_WIDTH = 2048 * 0.17 / 3600 # [deg]
CCD_HEIGHT = 4096 * 0.17 / 3600 # [deg]
PITCH_X = CCD_WIDTH + 12 / 3600 # [deg] distance between the centers of neighboring CCDs in a row
PITCH_Y = CCD_HEIGHT + 53 / 3600 # [deg] distance between the centers of neighboring rows


@pytest.fixture
def logs():
    """ 4 exposures: 3 at the same pointing (2 in g, 1 in r), 1 at a pointing 0.5 deg away """
    return pandas.DataFrame({"ra": [150., 150., 150.00001, 150.5],
                             "dec": [0., 0., 0., 0.],
                             "mjd": [59000., 59001., 59002., 59003.],
                             "band": ["hsc::g", "hsc::g", "hsc::r2", "hsc::g"],
                             "zp": 30., "gain": 3., "skynoise": 100.})

@pytest.fixture
def logs_per_ccd(logs):
    """ the same exposures, one row per exposure and CCD """
    return logs.merge(pandas.DataFrame({"ccdid": np.arange(104)}), how="cross")


# ============== #
#  Camera layout #
# ============== #
# tests for the func get_hsc_ccd_footprints
def test_get_hsc_ccd_footprints_ids():
    ccds = get_hsc_ccd_footprints()
    assert len(ccds) == 104, "HSC has 104 science CCDs"
    assert list(ccds.index) == list(range(104)), "DET-ID go from 0 to 103"
    assert ccds.index.name == "ccdid"

def test_get_hsc_ccd_footprints_ccd_size():
    ccds = get_hsc_ccd_footprints()
    np.testing.assert_allclose(ccds.area, CCD_WIDTH * CCD_HEIGHT)

def test_get_hsc_ccd_footprints_no_overlap():
    ccds = get_hsc_ccd_footprints()
    assert np.isclose(ccds.union_all().area, ccds.area.sum()), "CCDs overlap"

def test_get_hsc_ccd_footprints_extent():
    # rows of 15 CCDs: 15 widths and 14 gaps; 8 rows: 8 heights and 7 gaps
    xmin, ymin, xmax, ymax = get_hsc_ccd_footprints().total_bounds
    assert np.isclose(xmax - xmin, 15 * CCD_WIDTH + 14 * 12 / 3600)
    assert np.isclose(ymax - ymin, 8 * CCD_HEIGHT + 7 * 53 / 3600)
    assert np.isclose(xmax, -xmin) and np.isclose(ymax, -ymin), "camera not centered on 0"

def test_get_hsc_ccd_footprints_rows():
    # upright CCDs: rows of 7, 13, 15, 15, 15, 15, 13 and 7 from top to bottom
    ccds = get_hsc_ccd_footprints().drop([100, 101, 102, 103])
    rows = np.round(ccds.centroid.y / PITCH_Y + 3.5).astype(int) # 0 (bottom) .. 7 (top)
    assert list(rows.value_counts().sort_index()) == [7, 13, 15, 15, 15, 15, 13, 7]

def test_get_hsc_ccd_footprints_rotated_ccds():
    # CCDs 100 to 103 lie on their side: long axis along x
    ccds = get_hsc_ccd_footprints()
    for ccdid_ in [100, 101, 102, 103]:
        xmin, ymin, xmax, ymax = ccds[ccdid_].bounds
        assert np.isclose(xmax - xmin, CCD_HEIGHT) and np.isclose(ymax - ymin, CCD_WIDTH)

def test_get_hsc_ccd_footprints_known_positions():
    # from the CCD arrangement diagram: rows of 15 have a central CCD (column 0).
    # 49 and 50 are the central CCDs of the two rows around the center (below and above),
    # 1 is at the right end of the row of 49, 0 at the right end of the row below.
    ccds = get_hsc_ccd_footprints()
    assert np.isclose(ccds[49].centroid.x, 0) and np.isclose(ccds[49].centroid.y, -0.5 * PITCH_Y)
    assert np.isclose(ccds[50].centroid.x, 0) and np.isclose(ccds[50].centroid.y, 0.5 * PITCH_Y)
    assert np.isclose(ccds[1].centroid.x, 7 * PITCH_X) and np.isclose(ccds[1].centroid.y, -0.5 * PITCH_Y)
    assert np.isclose(ccds[0].centroid.x, 7 * PITCH_X) and np.isclose(ccds[0].centroid.y, -1.5 * PITCH_Y)

def test_get_hsc_ccd_footprints_exclude():
    ccds = get_hsc_ccd_footprints(exclude_ccds=subaru.HSC_DISABLED_CCDS)
    assert len(ccds) == 101
    assert not set(subaru.HSC_DISABLED_CCDS) & set(ccds.index)

def test_get_hsc_ccd_footprints_rotation():
    ccds = get_hsc_ccd_footprints()
    rotated = get_hsc_ccd_footprints(rotation=90)
    np.testing.assert_allclose(rotated.area, ccds.area)
    # 90 deg: x and y extents are swapped
    xmin, ymin, xmax, ymax = ccds.total_bounds
    rxmin, rymin, rxmax, rymax = rotated.total_bounds
    assert np.isclose(rxmax - rxmin, ymax - ymin) and np.isclose(rymax - rymin, xmax - xmin)

def test_get_hsc_ccd_footprints_pixel_scale():
    # the CCD area scales as the pixel scale squared
    ccds = get_hsc_ccd_footprints(pixel_scale=0.34)
    np.testing.assert_allclose(ccds.area, 4 * CCD_WIDTH * CCD_HEIGHT)


# tests for the func get_hsc_footprint
def test_get_hsc_footprint_per_ccd():
    footprint = get_hsc_footprint()
    assert footprint.geom_type == "MultiPolygon"
    assert len(footprint.geoms) == 104
    assert np.isclose(footprint.area, 104 * CCD_WIDTH * CCD_HEIGHT)

def test_get_hsc_footprint_outline():
    outline = get_hsc_footprint(per_ccd=False)
    ccds = get_hsc_footprint()
    assert outline.geom_type == "Polygon"
    assert len(outline.interiors) == 0, "gaps between CCDs should be filled"
    assert outline.buffer(1e-9).contains(ccds), "the outline should contain all the CCDs"
    np.testing.assert_allclose(outline.bounds, ccds.bounds)

def test_get_hsc_footprint_outline_fill_factor():
    # the CCDs fill the outline but for the gaps: ~(1 - 12/(348+12)) x (1 - 53/(696+53)) = 0.898
    # within a row and between rows (edge effects of the outer CCDs move it slightly).
    fill = get_hsc_footprint().area / get_hsc_footprint(per_ccd=False).area
    assert 0.88 < fill < 0.93

def test_get_hsc_footprint_outline_excluded_ccd_hole():
    # an excluded inner CCD leaves a hole in the outline: the CCD and the gaps on both of its
    # sides, i.e. (width + 2 x 12 arcsec) x (height + 2 x 53 arcsec).
    outline = get_hsc_footprint(per_ccd=False)
    outline_excl = get_hsc_footprint(per_ccd=False, exclude_ccds=[49])
    assert len(outline_excl.interiors) == 1
    expected = (CCD_WIDTH + 2 * 12 / 3600) * (CCD_HEIGHT + 2 * 53 / 3600)
    assert np.isclose(outline.area - outline_excl.area, expected)


# =============== #
#  Logs & fields  #
# =============== #
# tests for the func parse_hsc_logs
def test_parse_hsc_logs_per_exposure(logs):
    parsed, per_ccd = parse_hsc_logs(logs)
    assert not per_ccd
    assert len(parsed) == len(logs)
    # the 3 first exposures share the same pointing (1e-5 deg apart, rounded at 1e-4)
    assert parsed["fieldid"].iloc[0] == parsed["fieldid"].iloc[1] == parsed["fieldid"].iloc[2]
    assert parsed["fieldid"].nunique() == 2

def test_parse_hsc_logs_per_ccd(logs_per_ccd):
    parsed, per_ccd = parse_hsc_logs(logs_per_ccd)
    assert per_ccd
    assert parsed["fieldid"].nunique() == 2

def test_parse_hsc_logs_ccdkey(logs_per_ccd):
    parsed, per_ccd = parse_hsc_logs(logs_per_ccd.rename(columns={"ccdid": "detid"}), ccdkey="detid")
    assert per_ccd
    assert "ccdid" in parsed and "detid" not in parsed

def test_parse_hsc_logs_input_not_modified(logs):
    logs_in = logs.copy()
    _ = parse_hsc_logs(logs)
    pandas.testing.assert_frame_equal(logs, logs_in)

def test_parse_hsc_logs_filepath(logs, tmp_path):
    filepath = tmp_path / "hsc_logs.csv"
    logs.to_csv(filepath)
    parsed, per_ccd = parse_hsc_logs(str(filepath))
    assert not per_ccd
    np.testing.assert_allclose(parsed["mjd"], logs["mjd"])


# tests for the func get_hsc_fields
def test_get_hsc_fields_outline():
    fields = get_hsc_fields([150., 151.], [0., 2.], footprint=get_hsc_footprint(per_ccd=False))
    assert list(fields.index) == [0, 1]
    assert fields.index.name == "fieldid"
    # each field is centered on its pointing
    np.testing.assert_allclose(fields.centroid.x, [150., 151.], atol=1e-3)
    np.testing.assert_allclose(fields.centroid.y, [0., 2.], atol=1e-3)

def test_get_hsc_fields_per_ccd():
    fields = get_hsc_fields([150., 151.], [0., 2.], fieldid=[10, 20])
    assert fields.index.names == ["fieldid", "ccdid"]
    assert len(fields) == 2 * 104
    assert set(fields.index.get_level_values("fieldid")) == {10, 20}

def test_get_hsc_fields_area():
    # at the equator, the projected CCDs keep their area
    fields = get_hsc_fields(150., 0.) # a single pointing, given as floats
    assert len(fields) == 104
    np.testing.assert_allclose(fields.area, CCD_WIDTH * CCD_HEIGHT, rtol=1e-4)


# ============== #
#  HSC survey    #
# ============== #
# tests for the func HSC.from_logs
def test_from_logs_per_exposure(logs):
    survey = HSC.from_logs(logs)
    assert not survey.per_ccd
    assert survey.fields.index.names == ["fieldid"]
    assert len(survey.fields) == 2
    assert len(survey.data) == len(logs)
    assert survey.footprint.geom_type == "Polygon"

def test_from_logs_per_ccd(logs_per_ccd):
    survey = HSC.from_logs(logs_per_ccd)
    assert survey.per_ccd
    assert survey.fields.index.names == ["fieldid", "ccdid"]
    assert len(survey.fields) == 2 * 104
    assert len(survey.data) == len(logs_per_ccd)

def test_from_logs_single_pointing(logs, logs_per_ccd):
    # a single pointing (regression: project_to_radec used to fail with one pointing)
    assert len(HSC.from_logs(logs.iloc[:1]).fields) == 1
    assert len(HSC.from_logs(logs_per_ccd[logs_per_ccd["mjd"] == 59000.]).fields) == 104

def test_from_logs_filepath(logs, tmp_path):
    filepath = tmp_path / "hsc_logs.csv"
    logs.to_csv(filepath)
    survey = HSC.from_logs(str(filepath))
    assert not survey.per_ccd
    assert len(survey.data) == len(logs)

def test_from_logs_exclude_ccds(logs_per_ccd):
    survey = HSC.from_logs(logs_per_ccd, exclude_ccds=subaru.HSC_DISABLED_CCDS)
    assert len(survey.data) == len(logs_per_ccd) - 4 * 3, "rows of the excluded CCDs should be dropped"
    assert not set(subaru.HSC_DISABLED_CCDS) & set(survey.fields.index.get_level_values("ccdid"))

def test_from_logs_unknown_ccdid(logs_per_ccd):
    logs_ = pandas.concat([logs_per_ccd, logs_per_ccd.iloc[:1].assign(ccdid=999)])
    with pytest.warns(UserWarning):
        survey = HSC.from_logs(logs_)

    assert 999 not in set(survey.data["ccdid"])

def test_from_logs_rotation(logs):
    survey = HSC.from_logs(logs, rotation=90)
    xmin, ymin, xmax, ymax = survey.footprint.bounds
    assert np.isclose(xmax - xmin, 8 * CCD_HEIGHT + 7 * 53 / 3600), "rows should be along y after 90 deg"


# tests for the func radec_to_fieldid
def test_radec_to_fieldid_pointing_center(logs):
    # ra=150 is the center of pointing 0, and within the outline of pointing 1 (0.5 deg away,
    # the camera is 1.5 deg wide); ra=149.5 is only in pointing 0.
    survey = HSC.from_logs(logs)
    assert set(survey.radec_to_fieldid(pandas.DataFrame({"ra": [150.], "dec": [0.]}))["fieldid"]) == {0, 1}
    assert set(survey.radec_to_fieldid(pandas.DataFrame({"ra": [149.5], "dec": [0.]}))["fieldid"]) == {0}

def test_radec_to_fieldid_ccd_gap(logs, logs_per_ccd):
    # a single pointing at (150, 0). Halfway between CCD 49 (column 0) and 41 (column 1) is the
    # 12 arcsec gap; the pointing center itself is in the 53 arcsec gap between rows.
    # Positions in gaps are observed with the camera outline, not per CCD.
    survey_outline = HSC.from_logs(logs.iloc[:1])
    survey_ccd = HSC.from_logs(logs_per_ccd[logs_per_ccd["mjd"] == 59000.])
    on_ccd49 = pandas.DataFrame({"ra": [150.], "dec": [-0.5 * PITCH_Y]})
    in_gaps = pandas.DataFrame({"ra": [150. + 0.5 * PITCH_X, 150.], "dec": [-0.5 * PITCH_Y, 0.]})

    assert list(survey_ccd.radec_to_fieldid(on_ccd49)["ccdid"]) == [49]
    assert len(survey_ccd.radec_to_fieldid(in_gaps)) == 0, "positions in gaps should not be observed"
    assert len(survey_outline.radec_to_fieldid(in_gaps)) == 2, "the outline has no gaps"


# tests for the func get_regions
def test_get_regions(logs):
    far = logs.assign(ra=logs["ra"] + 10) # a second, separate region
    survey = HSC.from_logs(pandas.concat([logs, far], ignore_index=True))
    regions = survey.get_regions()
    assert len(regions) == 2
    assert regions[0].bounds[0] < regions[1].bounds[0], "regions should be sorted by ra"

def test_get_regions_single(logs):
    regions = HSC.from_logs(logs).get_regions()
    assert len(regions) == 1


# tests for the func get_nvisits_per_regions
def test_get_nvisits_per_regions_counts(logs):
    survey = HSC.from_logs(logs)
    region = survey.get_regions()[0]
    rabins, decbins, nvisits = survey.get_nvisits_per_regions(region, step=0.05)
    assert nvisits.shape == (len(decbins) - 1, len(rabins) - 1)
    # at the first pointing only (west of the second pointing's field): its 3 exposures
    ra, dec = 149.5, 0.
    i, j = np.searchsorted(decbins, dec) - 1, np.searchsorted(rabins, ra) - 1
    assert nvisits[i, j] == 3
    # where both pointings overlap: 3 + 1
    ra = 150.25
    j = np.searchsorted(rabins, ra) - 1
    assert nvisits[i, j] == 4
    assert nvisits.max() == 4

def test_get_nvisits_per_regions_bands(logs):
    survey = HSC.from_logs(logs)
    region = survey.get_regions()[0]
    _, _, nvisits_all = survey.get_nvisits_per_regions(region, step=0.05)
    _, _, nvisits_g = survey.get_nvisits_per_regions(region, step=0.05, bands="hsc::g")
    _, _, nvisits_r = survey.get_nvisits_per_regions(region, step=0.05, bands=["hsc::r2"])
    np.testing.assert_array_equal(nvisits_g + nvisits_r, nvisits_all)

def test_get_nvisits_per_regions_total(logs):
    # sum over the sky of the visits = number of exposures x camera outline area
    survey = HSC.from_logs(logs)
    step = 0.01
    _, _, nvisits = survey.get_nvisits_per_regions(survey.get_regions()[0], step=step)
    expected = len(logs) * survey.fields.area.mean()
    assert np.isclose(nvisits.sum() * step**2, expected, rtol=0.02) # 2%: grid discretisation of the edges


# ============== #
#  Plotting      #
# ============== #
@pytest.mark.parametrize("incl_both", [True, False])
@pytest.mark.parametrize("per_ccd", [True, False])
def test_show_footprint(logs, logs_per_ccd, incl_both, per_ccd):
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    survey = HSC.from_logs(logs_per_ccd if per_ccd else logs)
    fig = survey.show_footprint(incl_both=incl_both)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)

def test_show_deepfields_and_nvisits(logs):
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    survey = HSC.from_logs(logs) # a single region
    for fig in [survey.show_deepfields(), survey.show_nvisits()]:
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)
