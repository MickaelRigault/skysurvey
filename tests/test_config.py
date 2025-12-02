import numpy as np

from skysurvey.config import get_band_color


def test_band_color():
    assert get_band_color("ztfr") == "tab:red"
    assert get_band_color(np.str_("ztfr")) == "tab:red"
    assert get_band_color(["ztfr", "ztfg"]) == ["tab:red", "tab:green"]
    assert get_band_color("foo") is None
    assert get_band_color("foo", fill_value="bar") == "bar"
