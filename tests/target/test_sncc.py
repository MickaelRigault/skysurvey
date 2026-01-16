import numpy as np
from skysurvey.target.sncc import VincenziModels, SnanaModels, SNeII, SNeIIn, SNeIIb, SNeIb, SNeIc, SNeIcBL
from unittest.mock import patch


# tests for the class VincenziModels
def test_vincenzimodels_attributes():
    class DummyVincenzi(VincenziModels):
        _KIND = "Dummy"

        def set_template(self, templates):
            self._template = templates

    vincenzi = DummyVincenzi()

    assert vincenzi._KIND == "Dummy"
    assert vincenzi._TEMPLATES == "complex"
    assert np.isnan(vincenzi._RATE)

def test_vincenzimodels_template_property(monkeypatch):
    class DummyVincenzi(VincenziModels):
        _KIND = "DummyVincenzi"

        def set_template(self, templates):
            self._template = templates

    vincenzi = DummyVincenzi()

    with patch("skysurvey.target.sncc.get_sncosmo_sourcenames") as mock_get:
        mock_get.return_value = ["vincenzi_template1", "vincenzi_template2"]
        template = vincenzi.template

        mock_get.assert_called_once_with(vincenzi._KIND, startswith="v19", endswith="corr")
        assert template == ["vincenzi_template1", "vincenzi_template2"]
        assert vincenzi.template == ["vincenzi_template1", "vincenzi_template2"]

# tests for the class SnanaModels 
def test_snanamodels_template_property(monkeypatch):
    class DummySnana(SnanaModels):
        _KIND = "DummySnana"

        def set_template(self, templates):
            self._template = templates

    snana = DummySnana()

    with patch("skysurvey.target.sncc.get_sncosmo_sourcenames") as mock_get:
        mock_get.return_value = ["snana_template1", "snana_template2"]
        template = snana.template

        mock_get.assert_called_once_with(snana._KIND, startswith="snana", endswith="")
        assert template == ["snana_template1", "snana_template2"]
        assert snana.template == ["snana_template1", "snana_template2"]

# tests for the class SNeII
def test_sneii_instantiation():
    sneii = SNeII()

    assert sneii.__class__.__name__ == "SNeII"

def test_sneii_attributes():
    sneii = SNeII()
    CC_RATE = 1.0e5

    assert hasattr(sneii, "_KIND")
    assert sneii._KIND == "SN II"

    assert hasattr(sneii, "_RATE")
    assert sneii._RATE == CC_RATE * 0.649 

    assert hasattr(sneii, "_MAGABS")
    assert sneii._MAGABS == (-17.48, 0.7)

# tests for the class SNeIIn
def test_sneiin_instantiation():
    sneiin = SNeIIn()

    assert sneiin.__class__.__name__ == "SNeIIn"

def test_sneiin_attributes():
    sneiin = SNeIIn()
    CC_RATE = 1.0e5

    assert hasattr(sneiin, "_KIND")
    assert sneiin._KIND == "SN IIn"

    assert hasattr(sneiin, "_RATE")
    assert sneiin._RATE == CC_RATE * 0.047

    assert hasattr(sneiin, "_MAGABS")
    assert sneiin._MAGABS == (-18.0, 0.8)

# tests for the class SNeIIb
def test_sneiib_instantiation():
    sneiib = SNeIIb()

    assert sneiib.__class__.__name__ == "SNeIIb"

def test_sneiib_attributes():
    sneiib = SNeIIb()
    CC_RATE = 1.0e5

    assert hasattr(sneiib, "_KIND")
    assert sneiib._KIND == "SN IIb"

    assert hasattr(sneiib, "_RATE")
    assert sneiib._RATE == CC_RATE * 0.109

    assert hasattr(sneiib, "_MAGABS")
    assert sneiib._MAGABS == (-17.45, 0.6)

# tests for the class SNeIb
def test_sneib_instantiation():
    sneib = SNeIb()

    assert sneib.__class__.__name__ == "SNeIb"

def test_sneib_attributes():
    sneib = SNeIb()
    CC_RATE = 1.0e5

    assert hasattr(sneib, "_KIND")
    assert sneib._KIND == "SN Ib"

    assert hasattr(sneib, "_RATE")
    assert sneib._RATE == CC_RATE * 0.108

    assert hasattr(sneib, "_MAGABS")
    assert sneib._MAGABS == (-17.35, 0.53)

# tests for the class SNeIc
def test_sneic_instantiation():
    sneic = SNeIc()

    assert sneic.__class__.__name__ == "SNeIc"

def test_sneic_attributes():
    sneic = SNeIc()
    CC_RATE = 1.0e5

    assert hasattr(sneic, "_KIND")
    assert sneic._KIND == "SN Ic"

    assert hasattr(sneic, "_RATE")
    assert sneic._RATE == CC_RATE * 0.075

    assert hasattr(sneic, "_MAGABS")
    assert sneic._MAGABS == (-17.50, 0.7)

# tests for the class SNeIcBL
def test_sneicbl_instantiation():
    sneicbl = SNeIcBL()

    assert sneicbl.__class__.__name__ == "SNeIcBL"

def test_sneicbl_attributes():
    sneicbl = SNeIcBL()
    CC_RATE = 1.0e5

    assert hasattr(sneicbl, "_KIND")
    assert sneicbl._KIND == "SN Ic-BL"

    assert hasattr(sneicbl, "_RATE")
    assert sneicbl._RATE == CC_RATE * 0.097

    assert hasattr(sneicbl, "_MAGABS")
    assert sneicbl._MAGABS == (-18.12, 0.9)