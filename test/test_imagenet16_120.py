"""Focused CPU tests; synthetic images are not benchmark results."""
import hashlib
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import numpy as np
import torch

class ImageNet16Tests(unittest.TestCase):
    def test_loader_contract(self):
        module = importlib.import_module('ab.nn.loader.imagenet16-120')
        from ab.nn.util.Loader import get_obj

        self.assertIs(get_obj('imagenet16-120', 'loader'), module.loader)

        # Acquisition returns HWC uint8 images and zero-based labels.
        pixels = np.empty((2, 16, 16, 3), dtype=np.uint8)
        pixels[:] = [10, 20, 30]
        norms = []

        def transform_fn(norm):
            norms.append(norm)
            return lambda image: torch.from_numpy(
                np.array(image).copy()
            ).permute(2, 0, 1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'ImageNet16'

            with patch.object(module, 'data_dir', Path(directory)), \
                 patch.object(
                     module,
                     '_load_huggingface',
                     return_value=(pixels, [0, 119]),
                 ) as acquire:
                shape, baseline, train, valid = module.loader(
                    transform_fn, 'img-classification'
                )

                self.assertEqual(
                    acquire.call_args_list,
                    [call(root, True), call(root, False)],
                )
                self.assertEqual(shape, (120,))
                self.assertAlmostEqual(baseline, 1 / 120)
                self.assertEqual(len(norms), 1)
                np.testing.assert_allclose(
                    norms[0][0],
                    np.array([122.68, 116.66, 104.01]) / 255,
                )
                np.testing.assert_allclose(
                    norms[0][1],
                    np.array([63.22, 61.26, 65.09]) / 255,
                )

                for dataset in (train, valid):
                    self.assertEqual(len(dataset), 2)
                    self.assertEqual(
                        [dataset[i][1] for i in range(2)], [0, 119]
                    )

                    image, _ = dataset[0]
                    self.assertEqual(tuple(image.shape), (3, 16, 16))
                    self.assertEqual(
                        image[:, 0, 0].tolist(), [10, 20, 30]
                    )

                    batch, labels = next(iter(
                        torch.utils.data.DataLoader(
                            dataset, batch_size=2
                        )
                    ))
                    self.assertEqual(tuple(batch.shape), (2, 3, 16, 16))
                    self.assertEqual(labels.tolist(), [0, 119])

                raw = module.ImageNet16(root, train=False)
                self.assertEqual(raw[0][0].mode, 'RGB')
                self.assertEqual(raw[0][0].size, (16, 16))

    def test_acquisition_failure_propagates(self):
        module = importlib.import_module('ab.nn.loader.imagenet16-120')

        with patch.object(
            module,
            '_load_huggingface',
            side_effect=RuntimeError('Corrupt ImageNet16 file'),
        ):
            with self.assertRaisesRegex(RuntimeError, 'Corrupt ImageNet16'):
                module.loader(lambda norm: None, 'img-classification')

    def test_checksum_and_corrupt_file_preservation(self):
        module = importlib.import_module('ab.nn.loader.imagenet16-120')

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.parquet'
            payload = b'checksum fixture; not a real parquet file'
            path.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()

            module._check_hash(path, expected, 'sha256')

            path.write_bytes(b'corrupt')
            with self.assertRaisesRegex(RuntimeError, 'sha256 mismatch'):
                module._check_hash(path, expected, 'sha256')

            self.assertEqual(path.read_bytes(), b'corrupt')

if __name__ == '__main__':
    unittest.main()
