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
        limit = int(os.environ.get("NN_TRAIN_LIMIT", 0))
        if limit > 0 and self.split == "train": self.ids = self.ids[:limit]

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
                    image.filename = file_path # Preserve filename for cached transforms
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

def gpt2_collate_fn(batch):
    """Surgical Fix: Tokenize captions using GPT2 tokenizer for decoder compatibility."""
    import torch
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    images = torch.stack([item[0] for item in batch], dim=0)
    # COCO items[1] is a list of captions. Use the first one for training.
    raw_captions = [item[1][0] if isinstance(item[1], list) and len(item[1]) > 0 else str(item[1]) for item in batch]

    tokens = tokenizer(
        raw_captions,
        padding=True,
        truncation=True,
        max_length=40,
        return_tensors="pt",
    )
    # Shape: (B, seq_len) -> (B, 1, seq_len) to match existing pipeline expectations if needed, 
    # but Blip2Fast usually handles (B, seq_len) after unsqueeze.
    return images, tokens.input_ids

def loader(transform_fn, task):
    if task != 'img-captioning':
        raise Exception(f"The task '{task}' is not implemented in this file.")

    # --- Cached Feature Mode ---
    probe = transform_fn((__norm_mean, __norm_dev))
    if probe is None:
        import importlib
        mod_name = f"ab.nn.transform.{transform_fn.__module__.split('.')[-1]}"
        transform_module = importlib.import_module(mod_name)
        
        if hasattr(transform_module, 'get_dataset'):
            train_dataset = transform_module.get_dataset(split='train')
            try:
                val_dataset = transform_module.get_dataset(split='val')
            except Exception:
                val_dataset = transform_module.get_dataset(split='train')
            
            # [SPEED FIX] Apply NN_TRAIN_LIMIT to cached dataset
            import os
            from torch.utils.data import Subset
            limit = int(os.environ.get("NN_TRAIN_LIMIT", 0))
            if limit > 0:
                train_dataset = Subset(train_dataset, range(min(limit, len(train_dataset))))
                print(f"  [Caption] Cached Train dataset limited to {len(train_dataset)} samples.")

            val_dataset = Subset(val_dataset, range(min(500, len(val_dataset))))
            
            # Use GPT2 collate for transformer compatibility
            train_dataset.collate_fn = gpt2_collate_fn
            val_dataset.collate_fn = gpt2_collate_fn
            
            return (50257,), MINIMUM_ACCURACY, train_dataset, val_dataset
        else:
            raise ImportError(f"get_dataset() missing in {mod_name}")

    # --- Standard Raw Image Mode (unchanged) ---
    transform = probe
    path = join(data_dir, 'coco')
    train_dataset = COCOCaptionDataset(transform=transform, root=path, split='train')
    val_dataset = COCOCaptionDataset(transform=transform, root=path, split='val')
    # Reduce and Randomize validation set size for fast debugging

    import random
    val_ids = list(sorted(val_dataset.ids))
    random.shuffle(val_ids)
    val_dataset.ids = val_ids[:300]
    
    vocab_path = os.path.join(path, 'vocab.pth')
    if os.path.exists(vocab_path):
        vocab_data = torch.load(vocab_path)
        word2idx = vocab_data['word2idx']
        idx2word = vocab_data['idx2word']
    else:
        word2idx, idx2word = build_vocab(train_dataset, threshold=1)
        torch.save({'word2idx': word2idx, 'idx2word': idx2word}, vocab_path)
    train_dataset.word2idx = word2idx
    train_dataset.idx2word = idx2word
    val_dataset.word2idx = word2idx
    val_dataset.idx2word = idx2word

    GLOBAL_CAPTION_VOCAB['word2idx'] = word2idx
    GLOBAL_CAPTION_VOCAB['idx2word'] = idx2word
    
    train_dataset.collate_fn = lambda batch: train_dataset.__class__.collate_fn(batch, train_dataset.word2idx)
    val_dataset.collate_fn = lambda batch: val_dataset.__class__.collate_fn(batch, val_dataset.word2idx)

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
