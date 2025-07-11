# tests/test_lsst_comcam.py

from skysurvey.survey import lsst_comcam

def test_from_dp1_parquet():
    path = test/data/dp1_visits-ECDFS_EDFS_Fornax_LELF.parque
    survey = lsst_comcam.from_dp1_parquet(path)
    assert len(survey.data) > 0
    assert "mjd" in survey.data.columns
    assert all(survey.data["band"].str.startswith("lsst"))
