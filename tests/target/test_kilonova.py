import numpy as np
import sncosmo
from skysurvey.target.kilonova import read_possis_file, get_kilonova_model, Kilonova, _KILONOVA_MODEL
from skysurvey.tools.utils import random_radec

# tests for the func read_possis_file
def make_fake_possis_file(tmp_path):
    # fake possis file 
    path = tmp_path / "fake_possis.txt" 

    nobs = 3
    nwave = 4
    ntime = 2
    t_i, t_f = 0.0, 10.0
    header = [
        f"{nobs}\n",              # line 1
        f"{nwave}\n",             # line 2
        f"{ntime} {t_i} {t_f}\n"  # line 3
    ]

    # fake data
    rows = []
    for _ in range(nobs):
        for w in range(nwave):
            row = [1000 + w, 1.0, 2.0]
            rows.append(" ".join(map(str, row)) + "\n")

    path.write_text("".join(header + rows))
    return path

def test_read_possis_file(tmp_path):
    file_name = make_fake_possis_file(tmp_path)
    phase, wave, cos_theta, flux = read_possis_file(file_name)

    assert isinstance(phase, np.ndarray)
    assert isinstance(wave, np.ndarray)
    assert isinstance(cos_theta, np.ndarray)
    assert isinstance(flux, np.ndarray)

    assert phase.shape[0] == 2
    assert wave.shape[0] == 4
    assert cos_theta.shape[0] == 3
    assert flux.shape[0] == 2
    assert flux.shape[1] == 4
    assert flux.shape[2] == 3

# tests for the func get_kilonova_model 
def test_get_kilonova_model(monkeypatch):
    def fake_reader(filename):
        phase = np.array([0, 1])
        wave = np.array([4000, 5000])
        cos_theta = np.array([0.0, 0.5, 1.0])
        flux = np.ones((2, 2, 3))
        return phase, wave, cos_theta, flux
    
    import skysurvey.target.kilonova as kilonova
    monkeypatch.setattr(kilonova, "read_possis_file", fake_reader)

    model = get_kilonova_model("fake")

    assert isinstance(model, sncosmo.Model)
    assert model.source.name == "kilonova"

# tests for the class Kilonova
def test_kilonova_instantiation():
    kilonova = Kilonova()

    assert kilonova.__class__.__name__ == "Kilonova"

def test_kilonova_attributes():
    kilonova = Kilonova()

    assert hasattr(kilonova, "_KIND")
    assert kilonova._KIND == "kilonova"

    assert hasattr(kilonova, "_TEMPLATE")
    assert kilonova._TEMPLATE is _KILONOVA_MODEL

    assert hasattr(kilonova, "_RATE")
    assert kilonova._RATE == 1e3 

    assert "_MODEL" in dir(kilonova)

def test_kilonova_model_keys():
    kilonova = Kilonova()
    model = kilonova._MODEL

    model_keys = {"t0", "redshift", "magabs", "magobs", "theta", "radec"}
    assert model_keys.issubset(model.keys())

    t0 = model["t0"]
    assert "func" in t0
    assert callable(t0["func"])
    assert t0["func"].__name__ == "uniform"     
    assert t0["kwargs"] == {"low":56_000, "high":56_200}

    redshift = model["redshift"]
    assert redshift["kwargs"] == {"zmax":0.2}
    assert redshift["as"] == "z"

    magabs = model["magabs"]
    assert "func" in magabs
    assert callable(magabs["func"])
    assert magabs["func"].__name__ == "normal" 
    assert magabs["kwargs"] ==  {"loc": -18, "scale": 1}

    magobs = model["magobs"]
    assert "func" in magobs
    assert magobs["func"] == "magabs_to_magobs"
    assert magobs["kwargs"] == {"z":"@z", "magabs":"@magabs"}

    theta = model["theta"]
    assert "func" in theta
    assert callable(theta["func"])
    assert theta["func"].__name__ == "uniform"     
    assert theta["kwargs"] == {"low":0., "high":90.}

    radec = model["radec"]
    assert "func" in radec
    assert radec["func"] == random_radec
    assert radec["kwargs"] == {}
    assert radec["as"] == ["ra","dec"]