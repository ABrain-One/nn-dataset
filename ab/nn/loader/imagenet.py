import os
from torch.utils.data import Dataset
from ab.nn.util.Const import data_dir

__norm_mean = (0.485, 0.456, 0.406)
__norm_dev  = (0.229, 0.224, 0.225)
__class_quantity = 1000
MINIMUM_ACCURACY = 1.0 / __class_quantity

class _ImageNet(Dataset):
    def __init__(self, split, transform):
        self.data = split
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        item = self.data[i]
        return self.transform(item['image'].convert('RGB')), item['label']

def loader(transform_fn, task):
    if not os.environ.get('HF_TOKEN'):
        raise EnvironmentError(
            "HF_TOKEN not set. To fix:\n"
            "  1. Register at huggingface.co\n"
            "  2. Accept ImageNet terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k\n"
            "  3. Get token at huggingface.co/settings/tokens\n"
            "  4. export HF_TOKEN=hf_xxxxxxxxxxxx")
    from datasets import load_dataset
    transform = transform_fn((__norm_mean, __norm_dev))
    ds = load_dataset('ILSVRC/imagenet-1k', cache_dir=str(data_dir / 'imagenet'))
    train_set = _ImageNet(ds['train'],      transform)
    test_set  = _ImageNet(ds['validation'], transform)
    return (__class_quantity,), MINIMUM_ACCURACY, train_set, test_set