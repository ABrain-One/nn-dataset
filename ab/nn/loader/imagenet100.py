from torchvision.datasets import ImageFolder
from ab.nn.util.Const import data_dir

__norm_mean = (0.485, 0.456, 0.406)
__norm_dev  = (0.229, 0.224, 0.225)
__class_quantity = 100
MINIMUM_ACCURACY = 1.0 / __class_quantity
__dataset_dir = data_dir / 'imagenet100'

def __download():
    import kagglehub, shutil
    path = kagglehub.dataset_download("ambityga/imagenet100")
    shutil.copytree(path, __dataset_dir, dirs_exist_ok=True)
    shutil.rmtree(path)

def __merge_train(merged_dir):
    merged_dir.mkdir(exist_ok=True)
    for src in sorted(__dataset_dir.glob('train.X*')):
        for cls in src.iterdir():
            if cls.is_dir():
                link = merged_dir / cls.name
                if not link.exists():
                    link.symlink_to(cls.resolve())

def loader(transform_fn, task):
    transform = transform_fn((__norm_mean, __norm_dev))
    if not __dataset_dir.exists() or not any(__dataset_dir.iterdir()):
        __download()

    train_dir = __dataset_dir / 'train'
    val_dirs  = sorted(__dataset_dir.glob('val.X*'))

    if not val_dirs:
        raise FileNotFoundError(f"No val.X* dir found in {__dataset_dir}")

    if not train_dir.exists():
        __merge_train(train_dir)

    train_set = ImageFolder(root=str(train_dir),   transform=transform)
    test_set  = ImageFolder(root=str(val_dirs[0]), transform=transform)

    return (__class_quantity,), MINIMUM_ACCURACY, train_set, test_set