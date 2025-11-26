import os
from os import makedirs
from os.path import join, exists
import requests
from collections import Counter

import torch
import nltk
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_and_extract_archive

from nltk.tokenize import word_tokenize
from ab.nn.util.Const import data_dir
GLOBAL_CAPTION_VOCAB = {}

coco_ann_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
coco_img_url = 'http://images.cocodataset.org/zips/{}2017.zip'

__norm_mean = (104.01362025, 114.03422265, 119.9165958)
__norm_dev = (73.6027665, 69.89082075, 70.9150767)
MINIMUM_ACCURACY = 0.001

class COCOCaptionDataset(Dataset):
    def __init__(self, transform, root, split='train', word2idx=None, idx2word=None):
        super().__init__()
        nltk.download('punkt_tab')
        valid_splits = ['train', 'val']
        if split not in valid_splits:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

        self.root = root
        self.transform = transform
        self.split = split
        self.word2idx = word2idx
        self.idx2word = idx2word

        ann_dir = os.path.join(root, 'annotations')
        if not os.path.exists(ann_dir):
            print("COCO annotations not found! Downloading...")
            makedirs(root, exist_ok=True)
            download_and_extract_archive(coco_ann_url, root, filename='annotations_trainval2017.zip')
            print("Annotation download complete.")

        ann_file = os.path.join(ann_dir, f'captions_{split}2017.json')
        if not os.path.exists(ann_file):
            raise RuntimeError(f"Missing {ann_file}. Check that 'annotations_trainval2017.zip' was properly extracted.")

        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.img_dir = os.path.join(root, f'{split}2017')
        first_image_info = self.coco.loadImgs(self.ids[0])[0]
        first_file_path = os.path.join(self.img_dir, first_image_info['file_name'])
        if not os.path.exists(first_file_path):
            print(f"COCO {split} images not found! Downloading...")
            download_and_extract_archive(coco_img_url.format(split), root, filename=f'{split}2017.zip')
            print(f"COCO {split} image download complete.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        file_path = os.path.join(self.img_dir, img_info['file_name'])

        for attempt in range(2):  
            try:
                with Image.open(file_path) as img_file:
                    image = img_file.convert('RGB')
                    image = image.copy()
                break
            except Exception as e:
                if attempt == 0:
                    print(f'Image read error ({file_path}). Attempting to download on the fly.')
                    url = img_info.get('coco_url', None)
                    if not url:
                        raise RuntimeError(f"No coco_url found for image id {img_id}")
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                    else:
                        raise RuntimeError(f"Failed to download image: {img_info['file_name']} (status {response.status_code})")
                else:
                    raise RuntimeError(f"Could not load or download image: {file_path}")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        captions = []
        for ann in anns:
            if 'caption' in ann:
                captions.append(ann['caption'])
        if len(captions) == 0:
            captions = [""]

        if self.transform is not None:
            image = self.transform(image)
        
        # Ensure image is resized if it's not already (though transform should handle it)
        # If transform is 'echo', it won't resize. We need to enforce resizing for batching.
        # However, usually the transform passed in includes resizing.
        # The issue is that Optuna picked 'echo'.
        # We should probably enforce a resize here if the tensor size is not consistent.
        # But `image` is a tensor now.
        
        # Actually, let's just force a resize in the __getitem__ using torchvision transforms if we are in this specific mode.
        # Or better, let's update the `loader` to wrap the transform with a Resize.
        
        if image.dim() == 4 and image.size(0) == 1:
            image = image.squeeze(0)
        return image, captions

    @staticmethod
    def collate_fn(batch, word2idx):
        images = []
        all_captions = []
        for img, caps in batch:
            images.append(img)

            tokenized_captions = [
                [word2idx['<SOS>']] +
                [word2idx.get(word, word2idx['<UNK>']) for word in word_tokenize(cap.lower())] +
                [word2idx['<EOS>']]
                for cap in caps
            ]
            all_captions.append(tokenized_captions)

        images = torch.stack(images, dim=0)
        max_len = max(len(cap) for caps in all_captions for cap in caps)
        max_captions = max(len(caps) for caps in all_captions)

        padded_captions = []
        for caps in all_captions:
            padded_caps = [cap + [word2idx['<PAD>']] * (max_len - len(cap)) for cap in caps]
            num_to_pad = max_captions - len(caps)
            for _ in range(num_to_pad):
                padded_caps.append([word2idx['<PAD>']] * max_len)
            padded_captions.append(torch.tensor(padded_caps))

        captions_tensor = torch.stack(padded_captions, dim=0)
        return images, captions_tensor

    def collate(self, batch):
        return self.__class__.collate_fn(batch, self.word2idx)

def build_vocab(dataset, threshold=5):
    counter = Counter()
    for i in range(len(dataset)):
        _, captions = dataset[i]
        for caption in captions:
            tokens = word_tokenize(caption.lower())
            counter.update(tokens)
    specials = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    vocab_words = [word for word, count in counter.items() if count >= threshold]
    vocab_words = sorted(vocab_words)
    vocab = specials + vocab_words
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def robust_transform(img, base_transform):
    # 1. Resize
    from torchvision import transforms
    # img is likely a PIL image here coming from __getitem__
    # Resize works on PIL images too
    img = transforms.Resize((256, 256))(img)
    
    # 2. Apply the Optuna transform
    # The error showed base_transform expects PIL (it called to_tensor).
    # So we should pass the PIL image directly.
    # We remove the manual ToTensor conversion unless base_transform fails otherwise.
    
    try:
        return base_transform(img)
    except TypeError:
        # If it fails, maybe it DID expect a tensor?
        # But the previous error said "Got Tensor, expected PIL".
        # So passing PIL is correct for that case.
        # If this fails with "Expected Tensor", we can try converting.
        img_tensor = transforms.ToTensor()(img)
        return base_transform(img_tensor)

def loader(transform_fn, task):
    if task != 'img-captioning':
        raise Exception(f"The task '{task}' is not implemented in this file.")
    # We need to enforce resizing because Optuna might pick 'echo' (identity)
    # and COCO images vary in size, causing stack errors in collate_fn.
    from torchvision import transforms
    from functools import partial
    
    base_transform = transform_fn((__norm_mean, __norm_dev))
    
    # Use partial to bind base_transform to the module-level function
    transform = partial(robust_transform, base_transform=base_transform)
    
    path = join(data_dir, 'coco')
    # USE SMALL COCO (Validation Set as Training Data)
    # This avoids downloading the 20GB train set.
    full_dataset = COCOCaptionDataset(transform=transform, root=path, split='val')
    
    # Split the 5000 val images into 4500 Train and 500 Val
    # We use a fixed seed for reproducibility
    torch.manual_seed(42)
    train_size = 4500
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # We need to manually attach word2idx/idx2word to the subsets because random_split doesn't copy attributes
    # But first we need to build vocab on the TRAIN subset only to be rigorous
    # However, for simplicity and since it's the same domain, we can build on the full_dataset or just the train subset.
    # Let's build on the train subset.
    
    vocab_path = os.path.join(path, 'vocab_small.pth')
    if os.path.exists(vocab_path):
        vocab_data = torch.load(vocab_path)
        word2idx = vocab_data['word2idx']
        idx2word = vocab_data['idx2word']
    else:
        word2idx, idx2word = build_vocab(train_dataset, threshold=1)
        torch.save({'word2idx': word2idx, 'idx2word': idx2word}, vocab_path)

    # Attach attributes to the underlying dataset so collate_fn works
    # Note: random_split wraps the dataset, so we need to handle that in collate_fn or here.
    # Actually, the dataset class uses self.word2idx. 
    # Since we are using subsets, the underlying dataset is 'full_dataset'.
    full_dataset.word2idx = word2idx
    full_dataset.idx2word = idx2word
    
    # We don't need to re-assign to train_dataset/val_dataset because they access the underlying dataset
    # BUT, our collate_fn is static and takes word2idx as an arg.
    # So we just need to pass the correct word2idx to the partial.
    
    # Vocab already handled above

    GLOBAL_CAPTION_VOCAB['word2idx'] = word2idx
    GLOBAL_CAPTION_VOCAB['idx2word'] = idx2word
    
    from functools import partial
    # train_dataset is a Subset, so it doesn't have word2idx attribute directly.
    # We pass the word2idx dictionary directly.
    train_dataset.collate_fn = partial(COCOCaptionDataset.collate_fn, word2idx=word2idx)
    val_dataset.collate_fn = partial(COCOCaptionDataset.collate_fn, word2idx=word2idx)

    vocab_size = len(word2idx)

    # Set Net class-level attributes for printing sentences
    try:
        from ab.nn.nn.RESNETLSTM import Net as RESNETLSTMNet
        RESNETLSTMNet.idx2word = idx2word
        RESNETLSTMNet.eos_index = word2idx['<EOS>']
    except Exception:
        pass
    
    try:
        from ab.nn.nn.ResNetTransformer import Net as ResNetTransformerNet
        ResNetTransformerNet.word2idx = word2idx
        ResNetTransformerNet.idx2word = idx2word
    except Exception:
        pass

    return (vocab_size,), MINIMUM_ACCURACY, train_dataset, val_dataset
