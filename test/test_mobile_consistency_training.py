import importlib.util
from pathlib import Path
import unittest
import tempfile
import sys
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

spec = importlib.util.spec_from_file_location('mobile_training', Path(__file__).parents[1] / 'util/py/age/train_mobile_consistency.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class TinyAgeModel(torch.nn.Module):
    def __init__(self, in_shape, out_shape, parameters, device):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.AdaptiveAvgPool2d(1))
        self.head = torch.nn.Linear(4, 1)
        self.to(device)

    def forward(self, x):
        return 1 + 94 * torch.sigmoid(self.head(self.features(x).flatten(1)))


class MobileTrainingTest(unittest.TestCase):
    def test_training_writes_metrics_and_preserves_source_checkpoint(self):
        image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (40, 40, 3), dtype=np.uint8))
        dataset = SimpleNamespace(dataset=[{'image': image, 'age': 25} for _ in range(8)])
        train = SimpleNamespace(dataset=dataset, indices=list(range(4)))
        val = SimpleNamespace(dataset=dataset, indices=[4, 5],
                              held_out_test=SimpleNamespace(dataset=dataset, indices=[6, 7]))
        # Production split wrappers need __len__; reuse real Subset instead.
        from torch.utils.data import Subset
        train = Subset(dataset, train.indices)
        validation = Subset(dataset, val.indices)
        validation.held_out_test = Subset(dataset, val.held_out_test.indices)
        def fake_loader(*_):
            return (1,), .01, train, validation
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / 'baseline.pth'
            torch.save(TinyAgeModel(None, None, None, 'cpu').state_dict(), checkpoint)
            before = checkpoint.read_bytes()
            argv = ['train', '--checkpoint', str(checkpoint), '--output', str(root / 'run'),
                    '--epochs', '1', '--workers', '0', '--batch-size', '2', '--device', 'cpu']
            with patch.object(training, 'Net', TinyAgeModel), patch.object(training, 'loader', fake_loader), \
                    patch.object(sys, 'argv', argv):
                training.main()
            self.assertEqual(checkpoint.read_bytes(), before)
            self.assertEqual(json.loads((root / 'run/result.json').read_text())['state'], 'complete')
            history = json.loads((root / 'run/history.json').read_text())
            self.assertEqual(len(history), 1)
            self.assertTrue(np.isfinite(history[0]['loss']))
            self.assertEqual(history[0]['n'], 2)

    def test_identity_crop_and_deterministic_variants(self):
        image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (100, 120, 3), dtype=np.uint8))
        np.testing.assert_array_equal(np.asarray(training.crop_variant(image)), np.asarray(image))
        for variant in training.VARIANTS:
            a = training.tensor(training.crop_variant(image, *variant))
            b = training.tensor(training.crop_variant(image, *variant))
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(a.shape, (3, 224, 224))
            self.assertTrue(torch.isfinite(a).all())

    def test_selection_rejects_accuracy_or_crop_tail_regression(self):
        base = dict(mae=5., mean_crop_range=3., p95_crop_range=8.)
        self.assertTrue(training.eligible(dict(mae=5.1, mean_crop_range=2., p95_crop_range=7.), base))
        self.assertFalse(training.eligible(dict(mae=5.3, mean_crop_range=2., p95_crop_range=7.), base))
        self.assertFalse(training.eligible(dict(mae=5., mean_crop_range=2., p95_crop_range=9.), base))


if __name__ == '__main__':
    unittest.main()
