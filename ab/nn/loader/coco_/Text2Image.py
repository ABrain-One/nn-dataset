# File: Text2Image.py
# Description: A unified and robust dataloader for text-to-image tasks,
#              compatible with the LEMUR framework and all CVAE models.

import os
import random
from os.path import join
from PIL import Image

import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_and_extract_archive
import torchvision.transforms as T

from ab.nn.util.Const import data_dir

# --- Configuration ---
COCO_ANN_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
COCO_IMG_URL_TEMPLATE = 'http://images.cocodataset.org/zips/{}2017.zip'

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_DEV = (0.5, 0.5, 0.5)

TARGET_CATEGORIES = ['car']
DEFAULT_IMAGE_SIZE = 256  # Fallback size


class Text2Image(Dataset):
    """
    A PyTorch Dataset for loading COCO image-caption pairs, filtered by category.
    Includes robust error handling for missing or corrupt image files.
    """

    def __init__(self, root, split='train', transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.split = split

        ann_dir = join(root, 'annotations')
        if not os.path.exists(ann_dir):
            os.makedirs(root, exist_ok=True)
            download_and_extract_archive(COCO_ANN_URL, root, filename='annotations_trainval2017.zip')

        captions_ann_file = join(ann_dir, f'captions_{split}2017.json')
        self.coco = COCO(captions_ann_file)

        # Filter image IDs to only include those with the target categories
        if TARGET_CATEGORIES:
            instances_ann_file = join(ann_dir, f'instances_{split}2017.json')
            coco_instances = COCO(instances_ann_file)
            cat_ids = coco_instances.getCatIds(catNms=TARGET_CATEGORIES)
            img_ids = coco_instances.getImgIds(catIds=cat_ids)
            caption_img_ids = self.coco.getImgIds()
            self.ids = list(sorted(list(set(img_ids) & set(caption_img_ids))))
        else:
            self.ids = list(sorted(self.coco.imgs.keys()))

        self.img_dir = join(root, f'{split}2017')
        if self.ids and not os.path.exists(join(self.img_dir, self.coco.loadImgs(self.ids[0])[0]['file_name'])):
            download_and_extract_archive(COCO_IMG_URL_TEMPLATE.format(split), root, filename=f'{split}2017.zip')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = join(self.img_dir, img_info['file_name'])

        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, FileNotFoundError):
            # This robust error handling prevents training crashes from bad data
            print(f"Warning: Could not load image {img_path}. Skipping.")
            return self.__getitem__((index + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        # Return (image, text) for training and (image, dummy_tensor) for validation
        if self.split == 'train':
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            captions = [ann['caption'] for ann in anns if 'caption' in ann]
            text_prompt = random.choice(captions) if captions else "an image"
            return image, text_prompt
        else:
            return image, torch.tensor(0)


def loader(transform_fn, task, **kwargs):
    """
    The main entry point for the LEMUR framework. It creates a robust, global
    transform pipeline that works for all text-to-image models.
    """
    if 'txt-image' not in task.strip().lower():
        raise ValueError(f"The task '{task}' is not a text-to-image task for this dataloader.")

    # --- THE ROBUST GLOBAL TRANSFORM ---
    # Intelligently inspect the framework's transform to get the target size
    try:
        example_transform = transform_fn((NORM_MEAN, NORM_DEV))
        resize_step = next((t for t in example_transform.transforms if isinstance(t, T.Resize)), None)
        image_size = resize_step.size if isinstance(resize_step.size, int) else resize_step.size[0]
    except (AttributeError, TypeError, StopIteration):
        print(
            f"Warning: Could not determine image size from transform. Falling back to {DEFAULT_IMAGE_SIZE}x{DEFAULT_IMAGE_SIZE}.")
        image_size = DEFAULT_IMAGE_SIZE

    # Rebuild the transform pipeline correctly, adding the crucial CenterCrop step
    final_transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),  # <-- This is the fix that ensures all images are square
        T.ToTensor(),
        T.Normalize(NORM_MEAN, NORM_DEV)
    ])
    # --- END OF FIX ---

    path = join(data_dir, 'coco')
    train_dataset = Text2Image(root=path, split='train', transform=final_transform)
    val_dataset = Text2Image(root=path, split='val', transform=final_transform)

    # Define shapes for the framework
    out_shape = (3, image_size, image_size)
    in_shape = {'vocab_size': 30000}  # Placeholder, not used by CVAE models
    class_names = None

    return (in_shape, out_shape, class_names), 0.0, train_dataset, val_dataset