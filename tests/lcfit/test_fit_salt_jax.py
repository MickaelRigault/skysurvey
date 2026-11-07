"""Tests of skysurvey.lcfit.fit_salt_jax (saltjax wrapper) against fit_salt."""
import numpy as np
import pandas
import pytest

import skysurvey
from skysurvey.lcfit import fit_salt, fit_salt_jax

saltjax = pytest.importorskip("saltjax")
BANDS = ["desg", "desr", "desi"]


@pytest.fixture(scope="module")
def dataset():
    """Small SN Ia dataset observed by a seeded mock survey."""
    from shapely import geometry
    from skysurvey.tools import utils
    rng = np.random.default_rng(12)
    snia = skysurvey.SNeIa.from_draw(size=40, zmax=0.08, tstart=59_000, tstop=59_050,
                                     radec={"ra_range": [200, 220], "dec_range": [-10, 10]})
    size = 8_000
    data = {"gain": 1, "zp": 30, "skynoise": rng.normal(size=size, loc=300, scale=20),
            "mjd": rng.uniform(58_970, 59_120, size=size),
            "band": rng.choice(BANDS, size=size)}
    data["ra"], data["dec"] = utils.random_radec(size=size, ra_range=[200, 220],
                                                 dec_range=[-10, 10], rng=rng)
    survey = skysurvey.Survey.from_pointings(data, footprint=geometry.Point(0, 0).buffer(2))
    dset = skysurvey.DataSet.from_targets_and_survey(snia, survey, seed=12)
    dset.data["zpsys"] = "ab"
    ndet = dset.get_ndetection()
    return dset, list(ndet[ndet >= 7].index[:15])


@pytest.mark.parametrize("modelcov", [False, True])
def test_fit_salt_jax_matches_fit_salt(dataset, modelcov):
    pytest.importorskip("iminuit")
    dset, idx = dataset
    jres = fit_salt_jax(dset, indexes=idx, modelcov=modelcov)
    sres = fit_salt(dset, indexes=idx, modelcov=modelcov, in_scatter={}, warn=False,
                    bounds={"t0": 10, "x1": 4, "c": 0.6}).astype(float)
    assert isinstance(jres, pandas.DataFrame)
    assert set(sres.columns) <= set(jres.columns)
    common = sres.index.intersection(jres.index)
    assert len(common) == len(sres)
    for p in ["t0", "x0", "x1", "c"]:
        pull = (jres.loc[common, p] - sres.loc[common, p]) / sres.loc[common, f"{p}_err"]
        assert pull.abs().max() < 0.05, p
        ratio = jres.loc[common, f"{p}_err"] / sres.loc[common, f"{p}_err"]
        assert (ratio - 1).abs().max() < 0.02, p
    if modelcov:
        assert jres["converged"].all()


def test_unsupported_effect_raises(dataset):
    dset, _ = dataset
    snia = skysurvey.SNeIa.from_draw(size=20, zmax=0.08, tstart=59_000, tstop=59_050,
                                     radec={"ra_range": [200, 220], "dec_range": [-10, 10]},
                                     effect=skysurvey.effects.host_dust)
    dset_host = skysurvey.DataSet.from_targets_and_survey(snia, dset.survey, seed=1)
    with pytest.raises(NotImplementedError):
        fit_salt_jax(dset_host)
