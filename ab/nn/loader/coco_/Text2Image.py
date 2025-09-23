
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


class Text2Image(Dataset):
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
            print(f"Warning: Could not load image {img_path}. Skipping.")
            return self.__getitem__((index + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        if self.split == 'train':
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            captions = [ann['caption'] for ann in anns if 'caption' in ann]
            text_prompt = random.choice(captions) if captions else "an image"
            return image, text_prompt
        else:
            return image, torch.tensor(0)


def loader(transform_fn, task, **kwargs):
    if 'txt-image' not in task.strip().lower():
        raise ValueError(f"The task '{task}' is not a text-to-image task for this dataloader.")

    # --- THE DEFINITIVE FIX ---
    # Inspect the transform provided by the framework to get the target image size
    example_transform = transform_fn((NORM_MEAN, NORM_DEV))
    resize_step = example_transform.transforms[0]

    # Extract the integer size (e.g., 256) from the resize step
    image_size = -1
    if hasattr(resize_step, 'size'):
        size_attr = getattr(resize_step, 'size')
        image_size = size_attr if isinstance(size_attr, int) else size_attr[0]

    if image_size == -1:
        raise ValueError("Could not determine image size from the provided transform.")

    # Rebuild the transform pipeline correctly, adding the crucial CenterCrop step
    final_transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),  # <-- THE FIX: Ensures a square image
        T.ToTensor(),
        T.Normalize(NORM_MEAN, NORM_DEV)
    ])
    # --- END OF FIX ---

    path = join(data_dir, 'coco')
    train_dataset = Text2Image(root=path, split='train', transform=final_transform)
    val_dataset = Text2Image(root=path, split='val', transform=final_transform)
    return (None,), 0.0, train_dataset, val_dataset