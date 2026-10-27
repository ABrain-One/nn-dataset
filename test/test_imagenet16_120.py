"""Focused CPU tests; synthetic batches are not benchmark results."""
import hashlib
import importlib
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


class ImageNet16Tests(unittest.TestCase):
    def test_loader_contract(self):
        module = importlib.import_module('ab.nn.loader.imagenet16-120')
        from ab.nn.util.Loader import get_obj
        self.assertIs(get_obj('imagenet16-120', 'loader'), module.loader)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'ImageNet16'
            root.mkdir()
            files = []
            for name in ('train_data_batch_1', 'val_data'):
                pixels = np.zeros((4, 768), dtype=np.uint8)
                pixels[:, :256] = 10
                pixels[:, 256:512] = 20
                pixels[:, 512:] = 30
                payload = pickle.dumps({'data': pixels, 'labels': [1, 120, 121, 1000]}, protocol=2)
                (root / name).write_bytes(payload)
                files.append(((name, hashlib.md5(payload).hexdigest()),))
            calls = []
            def transform_fn(norm):
                calls.append(norm)
                return lambda image: torch.from_numpy(np.array(image).copy()).permute(2, 0, 1)
            with patch.object(module, 'data_dir', Path(directory)), \
                 patch.object(module.ImageNet16, 'train_list', files[0]), \
                 patch.object(module.ImageNet16, 'valid_list', files[1]):
                shape, baseline, train, valid = module.loader(transform_fn, 'img-classification')
                self.assertEqual(shape, (120,))
                self.assertAlmostEqual(baseline, 1 / 120)
                self.assertEqual(len(calls), 1)
                np.testing.assert_allclose(calls[0][0], np.array([122.68, 116.66, 104.01]) / 255)
                np.testing.assert_allclose(calls[0][1], np.array([63.22, 61.26, 65.09]) / 255)
                self.assertEqual(len(train), 2)
                self.assertEqual(len(valid), 2)
                self.assertEqual([train[i][1] for i in range(2)], [0, 119])
                image, _ = train[0]
                self.assertEqual(tuple(image.shape), (3, 16, 16))
                self.assertEqual(image[:, 0, 0].tolist(), [10, 20, 30])
                batch, labels = next(iter(torch.utils.data.DataLoader(train, batch_size=2)))
                self.assertEqual(tuple(batch.shape), (2, 3, 16, 16))
                self.assertEqual(labels.tolist(), [0, 119])
                raw = module.ImageNet16(root, train=False)
                self.assertEqual(raw[0][0].mode, 'RGB')
                self.assertEqual(raw[0][0].size, (16, 16))
                (root / 'val_data').write_bytes(b'corrupt')
                with self.assertRaisesRegex(RuntimeError, 'val_data'):
                    module.loader(transform_fn, 'img-classification')
                (root / 'val_data').unlink()
                with self.assertRaisesRegex(RuntimeError, 'val_data'):
                    module.loader(transform_fn, 'img-classification')


if __name__ == '__main__':
    unittest.main()
