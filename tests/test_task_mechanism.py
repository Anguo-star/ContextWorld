"""Small synthetic checks for the cross-task native visual objective."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('task_native_objective', SCRIPTS / 'diagnose_task_native_objective.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


@pytest.mark.parametrize('conditions,horizon', [(2, 3), (3, 3), (11, 7)])
def test_variable_condition_and_horizon_native_loss_gradient_closure(conditions, horizon):
    torch = pytest.importorskip('torch')
    torch.manual_seed(19)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.predictor = torch.nn.Linear(4, 4)
            self.action_encoder = torch.nn.Identity()

        def predict(self, history, action):
            return self.predictor(history + action)

    class Adapter:
        device = 'cpu'

        def __init__(self):
            self.model = Model()

        def _normalize_actions(self, value):
            return value

    class Helper:
        @staticmethod
        def predict_all(adapter, history, future, actions, family, modes):
            assert modes == ('free',)
            with torch.no_grad():
                output = adapter.model.predict(torch.as_tensor(history), torch.as_tensor(actions))
            return {'free': output[:, -1:, :].numpy()}

    adapter = Adapter()
    history = np.random.default_rng(42).normal(size=(conditions, horizon, 4)).astype('float32')
    future = np.random.default_rng(43).normal(size=(conditions, 1, 4)).astype('float32')
    actions = np.random.default_rng(44).normal(size=(conditions, horizon, 4)).astype('float32')
    named = list(adapter.model.predictor.named_parameters())
    row, gradients, endpoint = diagnostic.evaluate(adapter, Helper, 'lewm', history, future, actions, named, 'synthetic')
    assert row['position_count'] == horizon and row['condition_count'] == conditions
    assert endpoint.shape == (conditions, 4)
    assert row['losses']['native_all'] == pytest.approx(
        np.square(endpoint - future[:, 0]).mean()/horizon
        + sum(row['per_position']['native_by_position'][:-1])/horizon,
        abs=1e-6,
    )
    assert row['losses']['query_response'] == pytest.approx(row['per_position']['response_by_position'][-1]/horizon)
    assert row['gradient_reconstruction']['native_all']['relative_to_component_norm_sum'] < 1e-6
    assert row['gradient_reconstruction']['query_native']['relative_to_component_norm_sum'] < 1e-6
    assert diagnostic.generic._tensor_tuple_norm(gradients['native_all']) > 0


def test_selection_hashes_distinct_groups_and_uses_first_manifest_query():
    query_ids = []
    source_groups = []
    for group in range(18):
        for repeat in range(2):
            query = f'g{group}-q{repeat}'
            query_ids.extend([query, query])
            source_groups.extend([f'g{group}', f'g{group}'])

    class Raw:
        files = ['query_ids', 'source_groups']

        def __getitem__(self, key):
            return np.asarray({'query_ids': query_ids, 'source_groups': source_groups}[key])

    manifest = {'splits': {'development': {'query_ids': list(dict.fromkeys(query_ids))}}}
    key, selected, groups, fit_count = diagnostic._select(Raw(), manifest, 'development')
    assert key == 'query_ids' and fit_count == 36
    assert len(selected) == len(groups) == len(set(groups)) == 16
    assert selected == [f'{group}-q0' for group in groups]
    assert diagnostic._select(Raw(), manifest, 'development')[1:3] == (selected, groups)
