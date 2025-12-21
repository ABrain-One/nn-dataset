from .psnr import Net as MaiPsnr

# This is the Master List the framework uses to find your code
METRICS = {
    'mai_psnr': MaiPsnr
}
