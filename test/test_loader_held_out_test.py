"""No downloads: the utkface and wikitext loaders return a held-out test set in the test slot."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from ab.nn.loader import utkface, wikitext


class UTKFaceHeldOutTest(unittest.TestCase):
    def setUp(self):
        image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (16, 16, 3), dtype=np.uint8))
        self.records = [{"image": image, "age": a} for a in range(100) for _ in range(4)]
        with patch.object(utkface, "load_dataset", return_value=self.records), \
                patch.object(utkface.os, "makedirs"):
            self.out = utkface.loader(lambda norm: T.Compose([T.Resize((8, 8)), T.ToTensor()]), "age-regression")

    def test_arity_and_test_slot_is_held_out_subset(self):
        self.assertEqual(len(self.out), 4)
        _, _, train, test = self.out
        ages = [float(r["age"]) for r in self.records]
        train_idx, val_idx, test_idx = utkface._stratified_split_indices(ages, train_ratio=0.70, val_ratio=0.10, seed=42)
        self.assertEqual(list(test.indices), test_idx)
        self.assertEqual(list(train.indices), train_idx)
        self.assertEqual(list(test.val_set.indices), val_idx)
        self.assertGreater(len(test), len(test.val_set))

    def test_test_slot_disjoint_from_train_and_validation(self):
        _, _, train, test = self.out
        val = test.val_set
        self.assertFalse(set(test.indices) & set(train.indices))
        self.assertFalse(set(test.indices) & set(val.indices))
        self.assertFalse(set(train.indices) & set(val.indices))
        self.assertEqual(len(train) + len(val) + len(test), len(self.records))


class WikiTextHeldOutTest(unittest.TestCase):
    TRAIN = ["the quick brown fox jumps over the lazy dog. " * 4, "abc def ghi " * 4]
    TEST = ["a new text with unseen characters: 0123 xyz! " * 4, "Q@#~ zzz " * 4]

    def setUp(self):
        self.splits = []

        def fake_load_dataset(name, config, split):
            self.splits.append(split)
            return {"text": {"train": self.TRAIN, "test": self.TEST}[split]}

        with patch.object(wikitext, "load_dataset", side_effect=fake_load_dataset):
            self.out = wikitext.loader(None, "txt-generation")

    def test_arity_and_splits(self):
        self.assertEqual(len(self.out), 4)
        self.assertEqual(sorted(self.splits), ["test", "train"])
        out_shape, _, train, test = self.out
        self.assertIsNot(train, test)
        self.assertNotEqual(train.text_idx, test.text_idx)
        self.assertEqual(out_shape, (train.vocab_size,))

    def test_test_encoded_with_train_vocabulary_and_unknown_index(self):
        _, _, train, test = self.out
        train_text = "\n".join(self.TRAIN).lower()
        test_text = "\n".join(self.TEST).lower()
        self.assertEqual(train.chars, sorted(set(train_text)))
        self.assertEqual(test.chars, train.chars)
        self.assertEqual(test.char2idx, train.char2idx)
        self.assertEqual(test.unk_idx, len(train.chars))
        self.assertEqual(test.vocab_size, train.vocab_size)
        self.assertEqual(train.vocab_size, len(train.chars) + 1)
        unseen = set(test_text) - set(train_text)
        self.assertTrue(unseen)
        expected = [train.char2idx.get(c, train.unk_idx) for c in test_text]
        self.assertEqual(test.text_idx, expected)
        self.assertEqual(sum(i == train.unk_idx for i in test.text_idx), sum(c in unseen for c in test_text))
        self.assertNotIn(train.unk_idx, train.text_idx)
        self.assertLess(max(test.text_idx), train.vocab_size)

    def test_dataset_interface(self):
        _, _, train, test = self.out
        for ds in (train, test):
            self.assertEqual(len(ds), len(ds.text_idx) - ds.seq_length - 1)
            x, y = ds[0]
            self.assertEqual(x.dtype, torch.long)
            self.assertEqual(tuple(x.shape), (128,))
            self.assertTrue(torch.equal(x[1:], y[:-1]))


if __name__ == "__main__":
    unittest.main()
