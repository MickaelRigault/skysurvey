import numpy as np
import pandas
import pytest
import sncosmo

from astropy.cosmology import Planck18

from skysurvey.template import Template
from skysurvey.target.core import Transient
from skysurvey.target.timeserie import TSTransient
from skysurvey.target.collection import (broadcast_mapping, targets_from_collection,
                                         TargetCollection, TransientCollection,
                                         TSTransientCollection)


# Tiny hand-built sncosmo time-series sources: flat spectrum, gaussian in phase
# peaking at phase 0. No download needed.
PHASES = np.linspace(-20, 50, 15)
WAVES = np.linspace(2000, 10000, 50)

def make_source(name):
    flux = np.exp(-0.5 * (PHASES[:, None] / 10)**2) * np.ones((len(PHASES), len(WAVES))) * 1e-15
    return sncosmo.TimeSeriesSource(PHASES, WAVES, flux, name=name)


@pytest.fixture
def collection():
    target_a = TSTransient.from_draw(size=3, template=make_source("fakeA"), magabs=[-18, 0.1])
    target_b = TSTransient.from_draw(size=2, template=make_source("fakeB"), magabs=[-16, 0.1])
    return TransientCollection([target_a, target_b])


# tests for the func targets_from_collection
def test_targets_from_collection_not_implemented():
    with pytest.raises(NotImplementedError):
        targets_from_collection(None)

# tests for the func broadcast_mapping
def test_broadcast_mapping_scalar():
    assert list(broadcast_mapping(3, 4)) == [3, 3, 3, 3]

def test_broadcast_mapping_1d():
    assert list(broadcast_mapping([1, 2], 2)) == [1, 2]

def test_broadcast_mapping_1d_incompatible():
    with pytest.raises(ValueError):
        broadcast_mapping([1, 2], 3)

def test_broadcast_mapping_2d():
    # 2d input: broadcast along the first axis, last axis kept
    values = broadcast_mapping([[-18, 1]], 3)
    assert values.shape == (3, 2)
    assert (values == [[-18, 1]] * 3).all()

# tests for the class TargetCollection
def test_collection_empty():
    collection = TargetCollection()
    assert collection.targets == []
    assert collection.templates == []
    assert collection.template is collection.templates

def test_set_targets(collection):
    assert len(collection.targets) == 2
    assert isinstance(collection.targets[0], TSTransient)

def test_template_names(collection):
    assert collection.template_names == ["fakeA", "fakeB"]

def test_models(collection):
    models = collection.models
    assert len(models) == 2
    assert models[0] is collection.targets[0].model

# tests for the func call_down
def test_call_down_attribute(collection):
    assert collection.call_down("rate") == [0.001, 0.001]

def test_call_down_method_kwargs(collection):
    zmax = collection.call_down("get_model_parameter", entry="redshift", key="zmax")
    assert zmax == [0.05, 0.05]

def test_call_down_no_call(collection):
    methods = collection.call_down("get_rate", allow_call=False)
    assert all(callable(m) for m in methods), "the attribute itself is returned"

# tests for the func get_model_parameters
def test_get_model_parameters(collection):
    assert collection.get_model_parameters("magabs", "loc") == [-18, -16]
    assert collection.get_model_parameters("magabs", "unknown", default=0) == [0, 0]

# tests for the func get_data
def test_get_data(collection):
    data = collection.get_data()

    assert len(data) == 5, "3 + 2 targets"
    assert "kind" in data and "subindex" in data
    assert list(data["subindex"]) == [0, 1, 2, 0, 1], "index of each target's data"
    assert list(data["template"]) == ["fakeA"] * 3 + ["fakeB"] * 2

def test_get_data_colname(collection):
    data = collection.get_data(keys=["a", "b"], colname="origin")
    assert list(data["origin"]) == ["a"] * 3 + ["b"] * 2

def test_get_data_no_keys(collection):
    data = collection.get_data(keys=None)
    assert len(data) == 5
    assert "kind" not in data
    assert list(data.index) == [0, 1, 2, 0, 1], "original indices kept"

def test_data_property(collection):
    assert len(collection.data) == 5
    assert collection.data is collection.data, "stored after first call"

# tests for the func get_target_template
def test_get_target_template(collection):
    # index 3 is the first target of the second (fakeB) transient
    template = collection.get_target_template(3)
    row = collection.data.loc[3]

    assert isinstance(template, Template)
    assert template.source.name == "fakeB"
    assert np.isclose(template.sncosmo_model.get("z"), row["z"])
    assert np.isclose(template.sncosmo_model.get("t0"), row["t0"])

def test_get_target_template_as_model(collection):
    model = collection.get_target_template(0, as_model=True)
    assert isinstance(model, sncosmo.Model)
    assert model.source.name == "fakeA"

def test_get_target_template_set_magabs(collection):
    model = collection.get_target_template(4, as_model=True, set_magabs=True)
    # peak magnitude of the source set to the target's magobs in bessellb
    assert np.isclose(model.source_peakmag("bessellb", "ab"), collection.data.loc[4, "magobs"], atol=1e-4)

def test_get_target_template_set_magabs_no_magobs():
    data = pandas.DataFrame({"z": [0.01], "t0": [0.], "magabs": [-17.]})
    target = TSTransient.from_data(data, template=make_source("fakeA"))
    collection = TransientCollection([target])
    model = collection.get_target_template(0, as_model=True, set_magabs=True)
    assert np.isclose(model.source_peakabsmag("bessellb", "ab", cosmo=Planck18), -17., atol=1e-4)

# tests for the func to_transient
def test_to_transient(collection):
    transient = collection.to_transient()
    assert isinstance(transient, Transient)
    assert len(transient.data) == 5
    assert list(transient.data["template"]) == ["fakeA"] * 3 + ["fakeB"] * 2, "template column kept"

# tests for the func as_targets
def test_as_targets(collection):
    # as_targets rebuilds templates from their names: register them in sncosmo
    for name in ["fakeA", "fakeB"]:
        sncosmo.register(make_source(name), name=f"skysurvey-test-{name}", force=True)

    data = collection.get_data()
    data["template"] = data["template"].map(lambda name: f"skysurvey-test-{name}")
    collection._data = data
    targets = collection.as_targets()

    assert len(targets) == 2
    assert [len(t.data) for t in targets] == [3, 2]
    # the registered sources keep their own .name (registry key != source.name)
    assert [t.template_source.name for t in targets] == ["fakeA", "fakeB"]

def test_as_targets_no_template_column(collection):
    collection._data = pandas.DataFrame({"z": [0.1]})
    with pytest.raises(AttributeError):
        collection.as_targets()

# tests for the class TransientCollection
def test_update_model(collection):
    collection.update_model(t0={"func": np.random.uniform, "kwargs": {"low": 0, "high": 1}})
    assert collection.get_model_parameters("t0", "high") == [1, 1]

# tests for the class TSTransientCollection
def test_tscollection_from_sncosmo():
    collection = TSTransientCollection.from_sncosmo([make_source("fakeA"), make_source("fakeB")],
                                                    rates=[1e3, 2e3])
    assert len(collection.targets) == 2
    assert all(isinstance(t, TSTransient) for t in collection.targets)
    assert collection.template_names == ["fakeA", "fakeB"]
    assert collection.call_down("rate") == [1e3, 2e3]

def test_tscollection_from_sncosmo_broadcast_rate():
    collection = TSTransientCollection.from_sncosmo([make_source("fakeA"), make_source("fakeB")],
                                                    rates=5.)
    assert collection.call_down("rate") == [5., 5.]
