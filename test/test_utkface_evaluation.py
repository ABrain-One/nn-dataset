"""No downloads: verify the production loader separates train/evaluation inputs."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from ab.nn.loader import utkface
from ab.nn.transform.Resize_ColorJit_Flip_Blur import transform


class EvaluationPreprocessingTest(unittest.TestCase):
    def test_split_inputs_are_deterministic_and_match_deployment(self):
        torch.set_num_threads(2)
        pixels = np.random.default_rng(42).integers(0, 256, (260, 310, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        records = [{"image": image, "age": 25} for _ in range(30)]
        with patch.object(utkface, "load_dataset", return_value=records), \
                patch.object(utkface.os, "makedirs"):
            _, _, train, validation = utkface.loader(transform, "age-regression")

        test = validation.held_out_test
        self.assertIsNot(train.dataset, validation.dataset)
        self.assertEqual(len(train) + len(validation) + len(test), len(records))
        self.assertFalse(set(train.indices) & set(validation.indices))
        self.assertFalse(set(train.indices) & set(test.indices))
        self.assertFalse(set(validation.indices) & set(test.indices))
        expected = np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        expected = (expected - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
        expected = expected.transpose(2, 0, 1)
        for split in (validation, test):
            first, label = split[0]
            self.assertEqual(first.dtype, torch.float32)
            self.assertEqual(tuple(first.shape), (3, 224, 224))
            self.assertEqual(float(label[0]), 25)
            np.testing.assert_array_equal(first.numpy(), expected)
            for _ in range(9):
                self.assertTrue(torch.equal(first, split[0][0]))
        # The fix must retain augmentation for learning.
        torch.manual_seed(42)
        samples = [train[0][0] for _ in range(10)]
        self.assertTrue(any(not torch.equal(samples[0], x) for x in samples[1:]))


if __name__ == "__main__":
    unittest.main()
