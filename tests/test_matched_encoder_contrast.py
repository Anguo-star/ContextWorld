import importlib.util
from pathlib import Path
import numpy as np


def test_centered_energy_is_query_local_and_permutation_invariant(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'diagnose_matched_encoder_contrast.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('matched_encoder_contrast', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = np.array([[1., 100.], [3., 100.], [9., -50.], [13., -50.]])
    groups = np.array(['a', 'a', 'b', 'b'])
    expected = 2.0 + 8.0
    assert module.centered_energy(values, groups) == expected
    order = np.array([2, 0, 3, 1])
    assert module.centered_energy(values[order], groups[order]) == expected


def test_conditional_variance_decomposes_global_energy(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'diagnose_matched_encoder_contrast.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('matched_encoder_contrast', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = np.array([[1.], [3.], [9.], [13.]])
    got = module.conditional_fraction(values, np.array(['a','a','b','b']))
    assert np.isclose(got['within_query_energy'], 10.)
    assert np.isclose(got['global_centered_energy'], 91.)
    assert np.isclose(got['between_query_energy'], 81.)
    assert np.isclose(got['conditional_variance_fraction'], 10/91)
