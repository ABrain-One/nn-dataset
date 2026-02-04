import os
from pathlib import Path

nn_dataset = 'nn-dataset'

def get_version(version_file='version'):
    """Reads the version from the VERSION file located in the project root."""
    try:
        from importlib.metadata import version
        ver = version(nn_dataset)
        if ver:
            return ver
    except Exception:
        pass

    # Start from the directory containing this file and resolve to absolute path
    current_dir = Path(__file__).parent.resolve()

    # Traverse up to find the project root (where both version and pyproject.toml exist)
    max_iterations = 20  # Prevent infinite loops
    for _ in range(max_iterations):
        if (current_dir / version_file).exists() and (current_dir / "pyproject.toml").exists():
            break
        parent = current_dir.parent
        if parent == current_dir:  # Reached filesystem root
            break
        current_dir = parent

    version_path = current_dir / version_file
    if not version_path.is_file():
        raise FileNotFoundError(f"{version_file} not found in the project directory. Searched up to: {current_dir}")

    with open(version_path, "r") as f:
        version_str = f.read().strip()

    if not version_str:
        raise ValueError(f"Version file {version_path} is empty")

    return version_str


def add_version(nm: str) -> str:
    ver = get_version()
    if not ver:
        raise ValueError(f"get_version() returned None or empty string")
    return nm + '-' + ver


default_config = ''
default_epochs = 1
default_trials = -1  # one more trial
default_min_batch_power = 0
default_max_batch_power = 12
default_min_lr = 1e-5
default_max_lr = 1.0
default_min_momentum = 0.0
default_max_momentum = 1.0
default_min_dropout = 0.0
default_max_dropout = 0.5
default_pretrained = None
default_transform = None
default_train_missing_pipelines = None
default_nn_hyperparameters = {}

default_nn_fail_attempts = 30
default_num_workers = 2
default_random_config_order = False
default_save_pth_weights = False
default_save_onnx_weights = False

default_epoch_limit_minutes = None # minutes


base_module = 'ab'
to_nn = (base_module, 'nn')

def nn_mod(module_type: str, module_name: str) -> str:
    """
    Build a fully qualified module path for nn components.
    
    Args:
        module_type: Type of module ('nn', 'metric', 'loader', 'transform')
        module_name: Name of the specific module
    
    Returns:
        Full module path string (e.g., 'ab.nn.nn.RLFN')
    """
    return f"{base_module}.nn.{module_type}.{module_name}"

config_splitter = '_'


def nn_path(dr):
    """
    Defines path to ab/nn directory.
    """
    import ab.nn.util.__init__ as init_file
    return Path(init_file.__file__).parent.parent.absolute() / dr


metric_dir = nn_path('metric')
nn_dir = nn_path('nn')


def model_script(name):
    return nn_dir / f'{name}.py'


default_nn_name = 'AlexNet'
default_nn_path = model_script(default_nn_name)
transform_dir = nn_path('transform')
stat_dir = nn_path('stat')
stat_train_dir = stat_dir / 'train'
stat_run_dir = stat_dir / 'run'
stat_nn_dir = stat_dir / 'nn'

code_folders = (nn_dir, metric_dir)  # transform_dir,
gen_folders = (stat_dir,)


def __project_root_path():
    """
    Defines a path to the project root directory.
    """
    project_root = Path().absolute()
    current_dir = project_root
    while True:
        if (current_dir / base_module).exists() and (current_dir / 'README.md').exists():
            project_root = current_dir
            break
        if not current_dir.parent or current_dir.parent == current_dir:
            break
        current_dir = current_dir.parent.absolute()
    return project_root


ab_root_path = __project_root_path()
print(f"LEMUR root {ab_root_path}")
out = 'out'
out_dir = ab_root_path / out
ckpt_dir = out_dir / 'ckpt'
data_dir = ab_root_path / 'data'
db_dir = ab_root_path / 'db'
demo_dir = ab_root_path / 'demo'
db_file = db_dir / 'ab.nn.db'
zst_db_file = db_dir / add_version('ab.nn.zst')

onnx_dir = out_dir / 'onnx'
onnx_file = onnx_dir / 'nn.onnx'

main_tables = ('stat',)
main_columns = ('task', 'dataset', 'metric', 'nn')
main_columns_ext = main_columns + ('epoch',)
code_tables = ('nn', 'transform', 'metric')
param_tables = ('prm',)
dependent_tables = code_tables + param_tables
all_tables = main_tables + dependent_tables
index_colum = ('task', 'dataset') + dependent_tables
extra_main_columns = ('duration', 'accuracy')
nn_code_minhash_table = "nn_code_minhash"

# Mobile analytics (runtime) table
run_table = 'run'
# optional columns follow similar naming style; allow NULLs where data is not available
run_main_index = ('task', 'dataset', 'metric', 'nn')
run_extra_columns = (
    'device_type', 'os_version', 'valid', 'emulator', 'error_message',
    'duration', 'device_analytics_json')

# NN statistics table
nn_stat_table = 'nn_stat'
HF_NN = 'NN-Dataset'
tmp_data = "tmp_data"
STAT_TABLE = "stat"
NN_SIM_TABLE = "nn_similarity"
WORK_TABLE = "tmp_data"

core_nn_cls = (
    # img-classification
    'AirNet',
    'AirNext',
    'AlexNet',
    'BagNet',
    'ComplexNet',
    'BayesianNet-1',
    'ConvNeXt',
    'ConvNeXtTransformer',
    'DPN107',
    'DPN131',
    'DPN68',
    'DarkNet',
    'DenseNet',
    'Diffuser',
    'EfficientNet',
    'FractalNet',
    'GoogLeNet',
    'ICNet',
    'InceptionV3-1',
    'MNASNet',
    'MaxVit',
    'MoE-hetero4-Alex-Dense-Air-Bag',
    'MobileNetV2',
    'MobileNetV3',
    'RegNet',
    'ResNet',
    'ShuffleNet',
    'SqueezeNet-1',
    'SwinTransformer',
    'UNet2D',
    'VGG',
    'VisionTransformer',
    'RLFN',
    'SwinIR')

core_nn = core_nn_cls + (
    # img-segmentation
    'DeepLabV3-1',
    'FCN8s',
    'FCN16s',
    'FCN32s-1',
    'LRASPP',
    'UNet-1',
    # obj-detection
    'FasterRCNN',
    'FCOS',
    'RetinaNet',
    'SSDLite',
    # txt-generation
    'LSTM',
    'RNN',
    # img-captioning
    'RESNETLSTM',
    'ResNetTransformer')
