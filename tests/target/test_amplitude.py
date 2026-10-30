import numpy as np
import pytest
import sncosmo
from skysurvey.source import get_sncosmo_sourcenames
from skysurvey.target.snia import SNeIa
from skysurvey.target.sncc import SNeII, SNeIIn, SNeIIb, SNeIb, SNeIc, SNeIcBL

# every (supernova class, template): SALT2 for SNe Ia, all the Vincenzi templates otherwise
TEMPLATES = [(SNeIa, SNeIa._TEMPLATE)] + [
    (sn_class, name)
    for sn_class in (SNeII, SNeIIn, SNeIIb, SNeIb, SNeIc, SNeIcBL)
    for name in get_sncosmo_sourcenames(sn_class._KIND, startswith="v19", endswith="corr")
]

# test that set_source_peakabsmag(magabs) and set_source_peakmag(magobs) give the same amplitude
@pytest.mark.parametrize("sn_class, template", TEMPLATES,
                         ids=[f"{sn_class.__name__}-{name}" for sn_class, name in TEMPLATES])
def test_peakabsmag_equals_peakmag(sn_class, template):
    target = sn_class()
    band, magsys, cosmo = target.peak_absmag_band, target.magsys, target.cosmology
    magabs = -19.3 if sn_class is SNeIa else sn_class._MAGABS[0]

    for z in [0.01, 0.1, 1.0]:
        magobs = target.magabs_to_magobs(z, magabs)

        model_absmag = sncosmo.Model(source=template)
        model_absmag.set(z=z)
        model_absmag.set_source_peakabsmag(magabs, band, magsys, cosmo=cosmo)

        model_peakmag = sncosmo.Model(source=template)
        model_peakmag.set(z=z)
        model_peakmag.set_source_peakmag(magobs, band, magsys)

        # magabs_to_magobs computes distmod with z in float32
        assert np.isclose(model_absmag.get(target.amplitude_name),
                          model_peakmag.get(target.amplitude_name), rtol=1e-6, atol=0)
