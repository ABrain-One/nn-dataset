from .psnr import Net as MaiPsnr
from .mae import Net as Mae

# This is the Master List the framework uses to find your code
METRICS = {
    'mai_psnr': MaiPsnr,
    'mae': Mae
}
