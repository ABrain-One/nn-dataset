"""
Cached BLIP-2 Feature Transform for NAS Image Captioning Pipeline.

This transform activates Caption.py cached feature mode by returning None from
transform(). The dataset returns precomputed BLIP-2/Q-Former features with shape:
    (32, 768)

Important rules enforced:
- NO train fallback for validation.
- Validates split metadata inside each shard.
- Uses float16 CPU tensors.
- Fails loudly if expected shards are missing.
- Prints loaded shard names transparently.
"""

import os
import torch
from torch.utils.data import Dataset
from ab.nn.util.Const import cache_dir


_SHARED_CACHE = {}

def _auto_extract_features(split: str):
    import torch
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from transformers import BitsAndBytesConfig, Blip2Model
    from ab.nn.util.Loader import load_dataset
    
    print(f"\n[CACHE-AUTO] Missing cache for '{split}'. Starting automatic extraction to {cache_dir}...")
    os.makedirs(cache_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("[CACHE-AUTO] Loading BLIP-2 Encoder in 4-bit...")
    model = Blip2Model.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    print("[CACHE-AUTO] Loading underlying dataset via 'blip2_processor'...")
    out_shape, min_acc, train_dataset, test_dataset = load_dataset("img-captioning", "coco", "blip2_processor")
    
    target_dataset = train_dataset if split == "train" else test_dataset
    
    def _collate(batch):
        images = torch.stack([item[0] for item in batch])
        labels = [item[1] for item in batch]
        return images, labels
        
    loader = DataLoader(target_dataset, batch_size=32, num_workers=4, shuffle=False, collate_fn=_collate)
    features_list = []
    labels_list = []
    
    print(f"[CACHE-AUTO] Extracting features for {split}...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(loader)):
            images = images.to(device)
            feats = model.get_qformer_features(pixel_values=images).last_hidden_state
            # CRITICAL: Must save as float16 CPU — _load_shared_cache strictly validates this
            features_list.append(feats.cpu().to(torch.float16))
            labels_list.extend(labels)
            
            if (i + 1) % 500 == 0:
                temp_feats = torch.cat(features_list, dim=0)  # already float16
                torch.save({'features': temp_feats, 'labels': labels_list, 'split': split}, f"{cache_dir}/coco_{split}_{i}.pt")
                features_list = []
                labels_list = []
                
    if features_list:
        temp_feats = torch.cat(features_list, dim=0)  # already float16
        torch.save({'features': temp_feats, 'labels': labels_list, 'split': split}, f"{cache_dir}/coco_{split}_final.pt")
        
    print(f"[CACHE-AUTO] Extraction complete for {split}.\n")

def _discover_shards(split: str, auto_extracted: bool = False):
    """
    Strictly discovers shards for the given split.
    If none are found, automatically extracts them once.
    """
    prefix = f"coco_{split}_"
    if os.path.isdir(cache_dir):
        files = sorted(
            f for f in os.listdir(cache_dir)
            if f.startswith(prefix) and f.endswith(".pt")
        )
        if files:
            return files
            
    if not auto_extracted:
        _auto_extract_features(split)
        return _discover_shards(split, auto_extracted=True)

    raise FileNotFoundError(f"[FATAL] Missing strictly required '{split}' cache shards in {cache_dir}. Auto-extraction failed.")


def _load_shared_cache(split: str):
    """
    Load BLIP-2 feature shards once per Python process per split.
    """
    real_dir = os.path.realpath(cache_dir)
    cache_key = f"{real_dir}_{split}"

    if cache_key in _SHARED_CACHE:
        return _SHARED_CACHE[cache_key]

    shard_files = _discover_shards(split=split)

    print(f"\n[CACHE] Pre-loading BLIP-2 float16 CPU cache for split: '{split}'")
    
    all_features = []
    all_labels = []

    for sf in shard_files:
        shard_path = os.path.join(real_dir, sf)
        print(f"  -> Loading shard: {sf}")
        
        data = torch.load(shard_path, weights_only=False, map_location="cpu")

        if "features" not in data or "labels" not in data or "split" not in data:
            raise KeyError(f"Shard must contain 'features', 'labels', and 'split' metadata: {shard_path}")
            
        if data["split"] != split:
            raise ValueError(f"[FATAL] Shard {sf} has split='{data['split']}' but we requested '{split}'. Data Leakage Prevented!")

        # Ensure float16 CPU
        feat = data["features"]
        if feat.dtype != torch.float16 or feat.device.type != "cpu":
            raise ValueError(f"[FATAL] Shard {sf} contains {feat.dtype} on {feat.device}. Must be float16 CPU.")
            
        all_features.append(feat)
        all_labels.extend(data["labels"])
        del data
        
        # Prevent OOM by stopping early if we reached the limit
        limit = int(os.environ.get("NN_TRAIN_LIMIT", 0))
        if limit > 0 and split == "train" and len(all_labels) >= limit:
            print(f"  -> Reached limit {limit}. Truncating shard load.")
            excess = len(all_labels) - limit
            if excess > 0:
                all_labels = all_labels[:-excess]
                all_features[-1] = all_features[-1][:-excess]
            break

    # [FIX] Do NOT call torch.cat or .contiguous() on the full dataset! 
    # This prevents the 12.4GB RAM spike. We keep them as individual shards.
    shard_offsets = [0]
    for f in all_features:
        shard_offsets.append(shard_offsets[-1] + f.size(0))

    _SHARED_CACHE[cache_key] = {
        "features": all_features,  # List of tensors
        "labels": all_labels,
        "offsets": shard_offsets
    }

    print(f"[CACHE] '{split}' loaded successfully! Total length: {len(all_labels)}\n")
    return _SHARED_CACHE[cache_key]


class CachedBlip2Dataset(Dataset):
    def __init__(self, split: str = "train"):
        self.split = split

        # STRICTLY pass the requested split down to prevent data leakage
        cache = _load_shared_cache(split=self.split)
        self._all_features = cache["features"]  # List of Tensors
        self._all_labels = cache["labels"]
        self._offsets = cache["offsets"]
        self._length = len(self._all_labels)
        self._collate_fn = None

    @property
    def collate_fn(self):
        return self._collate_fn

    @collate_fn.setter
    def collate_fn(self, value):
        # Silently ignore external overrides (like from Caption.py) 
        # to preserve our multi-caption logic.
        pass

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        import bisect
        # O(log N) lookup to find exactly which shard this index belongs to
        shard_idx = bisect.bisect_right(self._offsets, idx) - 1
        local_idx = idx - self._offsets[shard_idx]
        
        # Convert float16 to float32 on the fly during training
        feature = self._all_features[shard_idx][local_idx].float()
        label = self._all_labels[idx]
        return feature, label


def get_collate_fn():
    """
    Collate cached BLIP-2 features and tokenize captions using GPT-2 tokenizer.
    """
    tokenizer = None

    def collate_fn(batch):
        nonlocal tokenizer
        if tokenizer is None:
            from transformers import BitsAndBytesConfig, GPT2Tokenizer
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
        features = torch.stack([item[0] for item in batch], dim=0)
        raw_captions = [item[1] if isinstance(item[1], (list, tuple)) else [item[1]] for item in batch]
        
        max_caps = max(len(caps) for caps in raw_captions)
        if max_caps == 0: max_caps = 1
            
        flat_captions = []
        for caps in raw_captions:
            padded_caps = list(caps) + [""] * (max_caps - len(caps))
            for cap in padded_caps:
                flat_captions.append(str(cap).strip() + tokenizer.eos_token)

        tokens = tokenizer(
            flat_captions, padding=True, truncation=True, max_length=60, return_tensors="pt"
        )
        
        batch_size = len(batch)
        seq_len = tokens.input_ids.size(1)
        labels = tokens.input_ids.view(batch_size, max_caps, seq_len)

        return features, labels

    return collate_fn


def transform(norm):
    # Return a dummy callable so Caption.py's native cache-probe logic triggers correctly.
    # Caption.py checks if `probe is None`, but if it is None, it falls back to raw mode or a specific hack branch.
    # Wait, the upstream Caption.py says:
    #     probe = transform_fn((__norm_mean, __norm_dev))
    #     if probe is None:
    #         # ... cached mode logic
    # Ah! Upstream Caption.py EXPECTS `probe is None` for cached mode!
    # So returning `None` here is ALREADY the correct upstream behavior.
    return None

def get_dataset(split: str = "train") -> CachedBlip2Dataset:
    dataset = CachedBlip2Dataset(split)
    dataset._collate_fn = get_collate_fn()
    return dataset

def get_vocab_size():
    return (50257,)
