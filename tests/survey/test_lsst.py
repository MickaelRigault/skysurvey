import sqlite3
import warnings

import numpy as np
import pandas
import pytest
import shapely
import geopandas
from shapely import geometry

from skysurvey.survey.lsst import (LSST, get_lsst_ccd_footprints, get_lsst_footprint, get_lsst_ccdid,
                                    get_lsst_fields, get_lsst_gain, parse_opsim, read_opsim,
                                    project_lsst_footprint, radec_to_camera)


# LSST camera, from lsst/obs_lsst: plate scale 20.006 arcsec/mm, rafts on a 127 mm pitch,
# CCDs on a 42.25 mm pitch within a raft, 10 micron pixels.
SCALE = 20.005867576692737 / 3600 # [deg/mm]
E2V_SIZE = np.array([4096, 4004]) * 0.010 * SCALE # [deg] (x, y)
ITL_SIZE = np.array([4072, 4000]) * 0.010 * SCALE # [deg]


@pytest.fixture
def opsim_path(tmp_path):
    """ tiny OpSim-like database: 2 visits at the same boresight (rotations 0 and 30 deg), 1 elsewhere """
    observations = pandas.DataFrame({"observationId": [10, 11, 12],
                                     "fieldRA": [10., 10., 100.],
                                     "fieldDec": [0., 0., -30.],
                                     "rotSkyPos": [0., 30., 90.],
                                     "observationStartMJD": [60000., 60001., 60002.],
                                     "filter": ["r", "r", "g"],
                                     "fiveSigmaDepth": [24.5, 24., 25.],
                                     "visitExposureTime": [30., 15., 30.],
                                     "night": [0, 1, 2],
                                     "skyBrightness": [21., 20., 19.],
                                     "numExposures": [1, 1, 1],
                                     "scheduler_note": ["a", "b", "c"]})
    filepath = tmp_path / "opsim.db"
    connect = sqlite3.connect(filepath)
    observations.to_sql("OBSERVATIONS", connect, index=False)
    connect.close()
    return str(filepath)

@pytest.fixture
def survey(opsim_path):
    return LSST.from_opsim(opsim_path)


# ============== #
#  Camera layout #
# ============== #
# tests for the func get_lsst_ccd_footprints
def test_get_lsst_ccd_footprints_ids():
    ccds = get_lsst_ccd_footprints(as_dataframe=True)
    assert len(ccds) == 189, "LSST has 189 science CCDs"
    assert list(ccds.index) == list(range(189))
    assert ccds.index.name == "ccdid"
    assert ccds["vendor"].value_counts().to_dict() == {"E2V": 117, "ITL": 72} # 13 E2V and 8 ITL rafts

def test_get_lsst_ccd_footprints_names():
    # obs_lsst: detector id = raft id0 + 3 x row + column
    names = get_lsst_ccd_footprints(as_dataframe=True)["name"]
    assert names[0] == "R01_S00"
    assert names[94] == "R22_S11", "94 is the central CCD"
    assert names[188] == "R43_S22"

def test_get_lsst_ccd_footprints_sizes():
    ccds = get_lsst_ccd_footprints(as_dataframe=True)
    widths = ccds.bounds["maxx"] - ccds.bounds["minx"]
    heights = ccds.bounds["maxy"] - ccds.bounds["miny"]
    e2v = ccds["vendor"] == "E2V"
    np.testing.assert_allclose(widths[e2v], E2V_SIZE[0])
    np.testing.assert_allclose(heights[e2v], E2V_SIZE[1])
    np.testing.assert_allclose(widths[~e2v], ITL_SIZE[0])
    np.testing.assert_allclose(heights[~e2v], ITL_SIZE[1])

def test_get_lsst_ccd_footprints_positions():
    ccds = get_lsst_ccd_footprints()
    # R22_S11 is at the center; R01_S00 at x=(1-2) x 127 - 42.25 mm, y=(0-2) x 127 - 42.25 mm
    np.testing.assert_allclose([ccds[94].centroid.x, ccds[94].centroid.y], [0, 0], atol=1e-12)
    np.testing.assert_allclose([ccds[0].centroid.x, ccds[0].centroid.y],
                               [(-127 - 42.25) * SCALE, (-254 - 42.25) * SCALE])

def test_get_lsst_ccd_footprints_no_overlap_and_area():
    ccds = get_lsst_ccd_footprints()
    assert np.isclose(ccds.union_all().area, ccds.area.sum()), "CCDs overlap"
    expected = 117 * np.prod(E2V_SIZE) + 72 * np.prod(ITL_SIZE)
    assert np.isclose(ccds.area.sum(), expected)
    assert np.isclose(expected, 9.55, atol=0.01), "official field of view: 9.6 deg2"

def test_get_lsst_ccd_footprints_extent():
    # outer edge: 2 rafts and 1 CCD pitch from the center, plus half an E2V CCD (R30, R14, R24, R34)
    xmin, ymin, xmax, ymax = get_lsst_ccd_footprints().total_bounds
    assert np.isclose(xmax, (2 * 127 + 42.25) * SCALE + E2V_SIZE[0] / 2)
    assert np.isclose(xmin, -xmax)

def test_get_lsst_ccd_footprints_exclude():
    ccds = get_lsst_ccd_footprints(exclude_ccds=[0, 94])
    assert len(ccds) == 187
    assert 94 not in ccds.index

def test_get_lsst_ccd_footprints_rotation():
    ccds = get_lsst_ccd_footprints()
    rotated = get_lsst_ccd_footprints(rotation=90)
    np.testing.assert_allclose(rotated.area, ccds.area)
    np.testing.assert_allclose([rotated[0].centroid.x, rotated[0].centroid.y],
                               [-ccds[0].centroid.y, ccds[0].centroid.x]) # (x, y) -> (-y, x)


# tests for the func get_lsst_footprint
def test_get_lsst_footprint_per_ccd():
    footprint = get_lsst_footprint()
    assert footprint.geom_type == "MultiPolygon"
    assert len(footprint.geoms) == 189

def test_get_lsst_footprint_outline():
    outline = get_lsst_footprint(per_ccd=False)
    ccds = get_lsst_footprint()
    assert outline.geom_type == "Polygon"
    assert len(outline.interiors) == 0, "gaps between CCDs should be filled"
    assert outline.buffer(1e-9).contains(ccds)
    np.testing.assert_allclose(outline.bounds, ccds.bounds)
    # Rubin quotes a filling factor > 90%
    assert 0.90 < ccds.area / outline.area < 1


# tests for the func get_lsst_ccdid
def test_get_lsst_ccdid_known_positions():
    gap_x = (42.25 * SCALE - E2V_SIZE[0]) # gap between R22_S11 and R22_S12
    x = [0, 0.5 * (E2V_SIZE[0] + gap_x), 5., (-127 - 42.25) * SCALE]
    y = [0, 0, 0, (-254 - 42.25) * SCALE]
    np.testing.assert_array_equal(get_lsst_ccdid(x, y), [94, -1, -1, 0]) # center, gap, outside, CCD 0

def test_get_lsst_ccdid_same_as_shapely():
    ccds = get_lsst_ccd_footprints()
    x, y = np.random.default_rng(1).uniform(-1.8, 1.8, size=(2, 100_000))
    points = geopandas.GeoDataFrame(geometry=shapely.points(x, y))
    joined = geopandas.sjoin(points, geopandas.GeoDataFrame(geometry=ccds), predicate="within")
    expected = np.full(len(x), -1)
    expected[joined.index.to_numpy()] = joined["ccdid"].to_numpy()
    np.testing.assert_array_equal(get_lsst_ccdid(x, y), expected)

def test_get_lsst_ccdid_excluded():
    ccds = get_lsst_ccd_footprints(exclude_ccds=[94])
    assert get_lsst_ccdid(0, 0, ccd_footprints=ccds)[0] == -1


# ======================= #
#   Camera on the sky     #
# ======================= #
# tests for the func project_lsst_footprint / radec_to_camera
def test_radec_to_camera_boresight():
    x, y = radec_to_camera(10., 0., 10., 0., 30.)
    np.testing.assert_allclose([x, y], [[0], [0]], atol=1e-10)
    # 1 deg north of the boresight, no rotation: camera 'up'
    x, y = radec_to_camera(10., 1., 10., 0., 0.)
    np.testing.assert_allclose([x[0], y[0]], [0, 1], atol=1e-10)

@pytest.mark.parametrize("ra, dec, rot", [(10., 0., 0.), (35., -60., 37.), (359.5, 10., 120.), (300., 85., 250.)])
def test_radec_to_camera_inverts_projection(ra, dec, rot):
    square = geometry.box(-1, -0.5, 1.5, 1)
    polygon = project_lsst_footprint(square, ra, dec, rot)[0]
    sky_ra, sky_dec = np.asarray(polygon.exterior.xy)
    x, y = radec_to_camera(sky_ra % 360, sky_dec, ra, dec, rot)
    np.testing.assert_allclose([x, y], np.asarray(square.exterior.xy), atol=1e-9)

def test_project_lsst_footprint_one_per_visit():
    outline = get_lsst_footprint(per_ccd=False)
    polygons = project_lsst_footprint(outline, [10., 20.], [0., 0.], [0., 90.])
    assert len(polygons) == 2
    # at the equator, the area is kept, and 90 deg swaps the x and y extents
    np.testing.assert_allclose([p_.area for p_ in polygons], outline.area, rtol=1e-3)


# ================== #
#   Visits (OpSim)   #
# ================== #
# tests for the func read_opsim
def test_read_opsim(opsim_path):
    df = read_opsim(opsim_path)
    assert list(df["observationId"]) == [10, 11, 12]
    assert "scheduler_note" not in df, "not a default column"

def test_read_opsim_sql_where(opsim_path):
    assert list(read_opsim(opsim_path, sql_where="night < 2")["observationId"]) == [10, 11]

def test_read_opsim_all_columns(opsim_path):
    df = read_opsim(opsim_path, columns=None)
    assert "scheduler_note" in df
    assert df.shape == (3, 12)

def test_read_opsim_note_renamed(opsim_path):
    # 'note' does not exist in this db, 'scheduler_note' is used instead
    df = read_opsim(opsim_path, columns=["observationId", "note"])
    assert list(df.columns) == ["observationId", "scheduler_note"]


# tests for the func get_lsst_gain
def test_get_lsst_gain():
    # SMTN-002: r gives 1 e-/s at 28.36 mag; in 30 s at zp=30
    expected = 10**(0.4 * (28.36 + 2.5 * np.log10(30) - 30))
    np.testing.assert_allclose(get_lsst_gain("r", 30.), expected)
    # half the exposure time, half the electrons
    np.testing.assert_allclose(get_lsst_gain(["r", "r"], [30., 15.]), [expected, expected / 2])

def test_get_lsst_gain_zp():
    # flux units 2.5 mag fainter (zp=32.5): 10 times fewer electrons per unit
    np.testing.assert_allclose(get_lsst_gain("g", 30., zp=32.5), get_lsst_gain("g", 30.) / 10)

def test_get_lsst_gain_custom_zeropoints():
    np.testing.assert_allclose(get_lsst_gain("r", 1., zp=30, zp_instrumental={"r": 30.}), 1.)


# tests for the func parse_opsim
def test_parse_opsim(opsim_path):
    data = parse_opsim(opsim_path)
    assert len(data) == 3
    assert list(data["fieldid"]) == [10, 11, 12]
    assert list(data["band"]) == ["lsstr", "lsstr", "lsstg"]
    np.testing.assert_allclose(data["rotskypos"], [0., 30., 90.])
    # 5-sigma depth 24.5 at zp=30: 5 x skynoise = 10**(0.4 x 5.5)
    assert np.isclose(data["skynoise"].iloc[0], 10**(0.4 * 5.5) / 5)
    np.testing.assert_allclose(data["gain"], get_lsst_gain(["r", "r", "g"], [30., 15., 30.]))

def test_parse_opsim_no_rotation(opsim_path):
    assert (parse_opsim(opsim_path, rotate=False)["rotskypos"] == 0).all()

def test_parse_opsim_sql_where(opsim_path):
    assert len(parse_opsim(opsim_path, sql_where="night < 2")) == 2

def test_parse_opsim_dataframe(opsim_path):
    visits = read_opsim(opsim_path, columns=None)
    pandas.testing.assert_frame_equal(parse_opsim(visits), parse_opsim(opsim_path))
    with pytest.warns(UserWarning):
        parse_opsim(visits, sql_where="night < 2")


# tests for the func get_lsst_fields
def test_get_lsst_fields():
    fields = get_lsst_fields([10., 100.], [0., -30.], [0., 90.], fieldid=[5, 7])
    assert list(fields.index) == [5, 7]
    assert fields.index.name == "fieldid"
    assert list(fields.columns[:3]) == ["ra", "dec", "rotskypos"]
    np.testing.assert_allclose([fields.centroid.x[5], fields.centroid.y[5]], [10., 0.], atol=1e-3)


# ============== #
#  LSST survey   #
# ============== #
# tests for the func LSST.from_opsim
def test_from_opsim(survey):
    assert survey.per_ccd, "per_ccd should be the default"
    assert len(survey.data) == 3
    assert survey.fields.index.names == ["fieldid"]
    assert len(survey.fields) == 3
    assert len(survey.ccd_footprints) == 189
    assert survey.footprint.geom_type == "Polygon"

def test_from_opsim_exclude_ccds(opsim_path):
    survey = LSST.from_opsim(opsim_path, exclude_ccds=[94])
    assert len(survey.ccd_footprints) == 188
    with pytest.warns(UserWarning):
        LSST.from_opsim(opsim_path, per_ccd=False, exclude_ccds=[94])


# tests for the func radec_to_fieldid
def test_radec_to_fieldid_boresight(survey):
    # the boresight of visits 10 and 11 is at the center of CCD 94 (R22_S11), whatever the rotation
    fieldids = survey.radec_to_fieldid(pandas.DataFrame({"ra": [10.], "dec": [0.]}))
    assert sorted(fieldids["fieldid"]) == [10, 11]
    assert set(fieldids["ccdid"]) == {94}

def test_radec_to_fieldid_ccd_gap(survey, opsim_path):
    # in the gap between R22_S11 and R22_S12 for visit 10 (no rotation): seen with the outline only.
    # (x>0 is West: ra decreases)
    gap_x = 0.5 * (E2V_SIZE[0] + (42.25 * SCALE - E2V_SIZE[0]))
    radec = pandas.DataFrame({"ra": [10. - gap_x], "dec": [0.]})
    fields_ccd = survey.radec_to_fieldid(radec)
    fields_outline = LSST.from_opsim(opsim_path, per_ccd=False).radec_to_fieldid(radec)
    assert 10 not in set(fields_ccd["fieldid"]), "a position in a gap should not be observed"
    assert 10 in set(fields_outline["fieldid"])

def test_radec_to_fieldid_array(survey):
    fieldids = survey.radec_to_fieldid(np.array([[10., 0.], [200., 50.]]))
    assert sorted(fieldids["fieldid"]) == [10, 11]

def test_radec_to_fieldid_rotation(opsim_path):
    # 1.7 deg East and 1 deg North of the boresight: at rotSkyPos=0, x=-1.7, y=1 (inside the
    # camera, which extends to |x|=1.76); at 30 deg, x=-1.7cos30-1sin30=-1.97 (outside).
    radec = pandas.DataFrame({"ra": [11.70], "dec": [1.0]})
    rotated = LSST.from_opsim(opsim_path, per_ccd=False).radec_to_fieldid(radec)
    not_rotated = LSST.from_opsim(opsim_path, per_ccd=False, rotate=False).radec_to_fieldid(radec)
    assert set(not_rotated["fieldid"]) == {10, 11}, "without rotation, both visits are identical"
    assert set(rotated["fieldid"]) == {10}, "the 30 deg rotated visit should not cover it"


# tests for the func get_nvisits_healpix
def test_get_nvisits_healpix(opsim_path):
    import healpy as hp
    survey = LSST.from_opsim(opsim_path, per_ccd=False) # outline: no CCD gap at the pixel centers
    nvisits = survey.get_nvisits_healpix(nside=32)
    assert len(nvisits) == hp.nside2npix(32)
    # pixel centers are within ~0.9 deg of the boresights, inside the camera (1.76 deg)
    pixel_10 = hp.ang2pix(32, 10., 0., lonlat=True) # visits 10 and 11 (r band)
    pixel_12 = hp.ang2pix(32, 100., -30., lonlat=True) # visit 12 (g band)
    assert nvisits[pixel_10] == 2 and nvisits[pixel_12] == 1
    nvisits_g = survey.get_nvisits_healpix(nside=32, bands="lsstg")
    assert nvisits_g[pixel_10] == 0 and nvisits_g[pixel_12] == 1

def test_get_nvisits_healpix_area(opsim_path):
    # 3 visits x 10.38 deg2 (outline) ~ number of covered pixels x pixel area
    import healpy as hp
    survey = LSST.from_opsim(opsim_path, per_ccd=False)
    nside = 128 # pixels of 0.46 deg
    nvisits = survey.get_nvisits_healpix(nside=nside)
    covered = nvisits.sum() * hp.nside2pixarea(nside, degrees=True)
    assert np.isclose(covered, 3 * get_lsst_footprint(per_ccd=False).area, rtol=0.05) # 5%: pixelisation


# ============== #
#  Plotting      #
# ============== #
@pytest.mark.parametrize("incl_both", [True, False])
@pytest.mark.parametrize("per_ccd", [True, False])
def test_show_footprint(opsim_path, incl_both, per_ccd):
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    fig = LSST.from_opsim(opsim_path, per_ccd=per_ccd).show_footprint(incl_both=incl_both)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)

def test_show_nvisits_map(survey):
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    fig = survey.show_nvisits_map(10., 0., size=2, step=0.1)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
