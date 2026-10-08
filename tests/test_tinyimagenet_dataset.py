import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from cifar100_datasets import NPZDataset


class NPZDatasetTest(unittest.TestCase):
    def test_loads_chw_uint8_and_converts_dtype(self):
        images = np.zeros((2, 3, 64, 64), dtype=np.uint8)
        images[0, 0] = 255
        labels = np.array([3, 7], dtype=np.int64)
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.npz"
            np.savez(path, images=images, labels=labels)
            dataset = NPZDataset(path, transform=transform)
            image, label = dataset[0]

        self.assertEqual(len(dataset), 2)
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(label.dtype, torch.int64)
        self.assertEqual(int(label), 3)
        self.assertTrue(torch.isfinite(image).all())
        self.assertEqual(float(image.max()), 1.0)

    def test_accepts_hwc_storage(self):
        images = np.zeros((1, 64, 64, 3), dtype=np.uint8)
        labels = np.array([9], dtype=np.int64)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.npz"
            np.savez(path, images=images, labels=labels)
            dataset = NPZDataset(path)
            image, label = dataset[0]

        self.assertEqual(tuple(image.shape), (3, 64, 64))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(int(label), 9)

    def test_rejects_image_label_length_mismatch(self):
        images = np.zeros((2, 3, 64, 64), dtype=np.uint8)
        labels = np.array([1], dtype=np.int64)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.npz"
            np.savez(path, images=images, labels=labels)
            with self.assertRaisesRegex(ValueError, "length mismatch"):
                NPZDataset(path)


if __name__ == "__main__":
    unittest.main()
