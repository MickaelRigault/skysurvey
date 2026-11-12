import warnings

import numpy as np
import pandas
import pytest

from skysurvey.survey.core import BaseSurvey


@pytest.fixture
def logs():
    """ tiny observing log, NOT sorted by mjd and in 64-bit precision """
    return pandas.DataFrame({"mjd": [3., 1., 2., 4.],
                             "band": ["g", "r", "g", "r"],
                             "skynoise": [1., 2., 3., 4.],
                             "gain": [1., 1., 1., 1.],
                             "zp": [25., 25., 25., 25.],
                             "fieldid": [10, 10, 20, 30]})

@pytest.fixture
def survey(logs):
    return BaseSurvey(logs)


# tests for the class BaseSurvey
def test_init_none():
    survey = BaseSurvey(None)
    assert survey.data is None
    assert survey.fields is None

def test_of_type(survey):
    assert survey.of_type == "BaseSurvey"

def test_metadata(survey):
    assert survey.metadata == {"type": "BaseSurvey"}

def test_array(survey):
    # numpy representation of the data: (nobs, ncolumns)
    assert np.asarray(survey).shape == (4, 6)


# tests for the func set_data
def test_set_data_sort_mjd(survey):
    assert list(survey.data["mjd"]) == [1., 2., 3., 4.]
    # rows are moved, not only the mjd column
    assert list(survey.data["skynoise"]) == [2., 3., 1., 4.]

def test_set_data_no_sort_mjd(logs):
    survey = BaseSurvey(None)
    survey.set_data(logs, sort_mjd=False)
    assert list(survey.data["mjd"]) == [3., 1., 2., 4.]

def test_set_data_decreasing_mjd_kept(logs):
    # monotonic decreasing data are not considered as needing a sort
    decreasing = logs.sort_values("mjd", ascending=False)
    survey = BaseSurvey(decreasing)
    assert list(survey.data["mjd"]) == [4., 3., 2., 1.]

def test_set_data_lower_precision(survey):
    assert survey.data["mjd"].dtype == "float32"
    assert survey.data["fieldid"].dtype == "int32"

def test_set_data_keep_precision(logs):
    survey = BaseSurvey(None)
    survey.set_data(logs, lower_precision=False)
    assert survey.data["mjd"].dtype == "float64"
    assert survey.data["fieldid"].dtype == "int64"

def test_set_data_does_not_modify_input(logs):
    _ = BaseSurvey(logs)
    assert list(logs["mjd"]) == [3., 1., 2., 4.], "input data have been modified"
    assert logs["mjd"].dtype == "float64", "input data have been modified"

def test_set_data_missing_column(logs):
    with pytest.warns(UserWarning, match="missing"):
        _ = BaseSurvey(logs.drop(columns="skynoise"))

def test_set_data_complete_no_warning(logs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = BaseSurvey(logs)


# tests for the func get_timerange
def test_get_timerange(survey):
    assert list(survey.get_timerange()) == [1., 4.]


# tests for the func date_range
def test_date_range(survey):
    assert survey.date_range == (1., 4.)


# tests for the func nfields
def test_nfields_from_data(survey):
    # no nfields set: max of data["fieldid"] is used, with a warning.
    with pytest.warns(UserWarning):
        nfields = survey.nfields
    assert nfields == 30


# tests for the func radec_to_fieldid
def test_radec_to_fieldid_not_implemented(survey):
    with pytest.raises(NotImplementedError):
        survey.radec_to_fieldid([10, 20])


# tests for the func show
def test_show_not_implemented(survey):
    with pytest.raises(NotImplementedError):
        survey.show()


# tests for the func get_night_boundary
def get_survey_at(times_of_day, ndays=30):
    """ survey observing at the given times of day (fraction of the UT day), every day """
    mjd = (np.arange(ndays)[:, None] + np.asarray(times_of_day)[None, :]).ravel()
    return BaseSurvey(pandas.DataFrame({"mjd": mjd, "band": "g", "skynoise": 1., "gain": 1., "zp": 25.}))

def test_get_night_boundary_from_logs():
    # observing from 1h to 9h UT: the daytime is from 9h to 25h UT, its middle at 17h UT
    survey = get_survey_at(np.linspace(1, 9, 50) / 24)
    assert np.isclose(survey.get_night_boundary(), 17 / 24, atol=2 / 1440) # 2 minutes (1 minute bins)

def test_get_night_boundary_daytime_across_midnight():
    # observing from 14h to 22h UT: the daytime (22h to 14h) crosses 0h UT, its middle at 6h UT
    survey = get_survey_at(np.linspace(14, 22, 50) / 24)
    assert np.isclose(survey.get_night_boundary(), 6 / 24, atol=2 / 1440)

def test_get_night_boundary_longitude(survey):
    # local mean noon: 12h UT at longitude 0, 12h + 70.75/15 h UT at longitude -70.75 (Rubin)
    assert np.isclose(survey.get_night_boundary(longitude=0), 0.5)
    assert np.isclose(survey.get_night_boundary(longitude=-70.75), (12 + 70.75 / 15) / 24)

def test_get_night_boundary_all_day():
    # observations at all times of day (e.g. space telescope): the longitude is needed
    survey = get_survey_at(np.arange(0, 1, 1 / 1440))
    with pytest.raises(ValueError):
        survey.get_night_boundary()
