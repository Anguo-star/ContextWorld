"""Small semantic checks; no benchmark dataset reads."""
import importlib.util
import unittest
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image
spec = importlib.util.spec_from_file_location('reference', Path(__file__).resolve().parents[1] / 'scripts/validate_expanded_history_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

def image_bytes(color):
    buffer = BytesIO()
    Image.new('RGB', (4, 4), color).save(buffer, format='PNG')
    return buffer.getvalue()

class ReferenceTests(unittest.TestCase):
    def test_actual_pair_groups(self):
        self.assertEqual(reference.group({'pair_id':'pcf-train-strat-p000000-a03'}, 'friction'), 'pcf-train-strat-p000000')
        self.assertNotEqual(reference.group({'pair_id':'pcf-train-strat-p000001-a03'}, 'friction'), 'pcf-train-strat-p000000')
        self.assertEqual(reference.group({'pair_id': 'pmd-train-000000-forward', 'catalog_index': [0.0]}, 'damping'), 'pmd-train-source-000000')
        self.assertEqual(reference.group({'pair_id': 'pmd-train-000001-reverse', 'catalog_index': [1.0]}, 'damping'), 'pmd-train-source-000000')
        self.assertEqual(reference.group({'pair_id': 'pmd-loader-validation-000003-reverse'}, 'damping'), 'pmd-loader-validation-source-000001')
        self.assertEqual(reference.group({'source_episode':42}, 'cube'), '42')

    def test_threshold_both_directions(self):
        for labels in [np.array([0,0,1,1]),np.array([1,1,0,0])]:
            values=np.array([0.1,0.2,0.8,0.9])
            threshold,flip,overlap=reference.fit_threshold(values,labels)
            np.testing.assert_array_equal((values>threshold).astype(int)^flip,labels)
            self.assertFalse(overlap)

    def test_control_theoretical_ceiling(self):
        black=image_bytes('black');white=image_bytes('white')
        rows=[dict(pair='pair',mode=mode,pixels=[black,black,black],actions=np.zeros(30),future=future) for mode,future in [('a',black),('b',white)]]
        controls=reference.controls(rows)
        for name in ['current_only_equal','action_only_equal','current_plus_action_equal']:
            self.assertEqual(controls[name]['deterministic_classifier_theoretical_accuracy_upper_bound_not_trained_score'],0.5)
        self.assertEqual(controls['future_different']['fraction'],1.0)

if __name__ == '__main__':
    unittest.main()
