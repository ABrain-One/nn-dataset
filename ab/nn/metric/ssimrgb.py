import torch
import torch.nn.functional as F


class Net:
    """RGB SSIM matching the SplitterNet paper's evaluation (Flepp/Ignatov CVPR2024, evaluate.py):

        tf.image.ssim(gt, pred, 255.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)

    Three things differ from the framework's default `ssim` metric, and all three move the number:
      * per-CHANNEL SSIM averaged over R,G,B — not Y-channel luminance only;
      * VALID convolution (tf.image.ssim pads nothing), not same-padding, so the 5px border is
        excluded rather than being computed against zero-padded neighbourhoods;
      * Gaussian window, 11x11 sigma=1.5, applied depthwise.

    C1/C2 scale with max_val, so evaluating in [0,1] with max_val=1.0 is identical to their
    [0,255] with max_val=255. SSIM is already in [0,1], so it needs no /48 normalization the way
    psnr_rgb does.

    Use this (metric='ssimrgb') for like-for-like comparison against their reported 0.884.

    VERIFIED against an independent float64 numpy/scipy reference to within 1.2e-07 across
    image sizes and noise levels. Deliberately NOT calibrated against tf.image.ssim: on
    TF 2.21.0 CPU, float32 `reduce_mean` over a partial axis pair accumulates naively and
    returns values that can exceed the tensor's own maximum — measured 8.6e-04 high on a
    290x441x3 reduction, which is the same order as the SSIM differences we care about.
    tf.image.ssim inherits that error (its per-channel spatial mean is exactly such a
    reduction), so treat any SSIM produced by the authors' evaluate.py, including the
    published 0.884, as carrying an unknown positive bias of roughly this magnitude.
    """

    def __init__(self, out_shape=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self._kernel = None
        self.reset()

    def reset(self):
        self.total_ssim = 0.0
        self.count = 0

    def _gaussian(self, channels, device, dtype):
        """1D 11-tap sigma=1.5 Gaussian as a depthwise (1,11) / (11,1) separable pair.

        tf.image.ssim uses a 2D outer-product kernel; separable is mathematically identical
        (the Gaussian is separable) and cheaper, and both agree with the float64 reference to
        ~1e-7. Kept separable purely for speed on full-resolution images.
        """
        key = (channels, device, dtype)
        if self._kernel is not None and self._kernel[0] == key:
            return self._kernel[1], self._kernel[2]
        coords = torch.arange(11, dtype=dtype, device=device) - 5.0
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        kh = g.view(1, 1, 1, 11).expand(channels, 1, 1, 11).contiguous()
        kv = g.view(1, 1, 11, 1).expand(channels, 1, 11, 1).contiguous()
        self._kernel = (key, kh, kv)
        return kh, kv

    def _reduce(self, t, kh, kv, c):
        """Depthwise separable Gaussian, VALID padding — tf.image.ssim's reducer."""
        return F.conv2d(F.conv2d(t, kh, groups=c), kv, groups=c)

    def _ssim(self, x, y):
        """Per-image SSIM averaged over channels, VALID padding. x,y are [B,C,H,W] in [0,1]."""
        c = x.shape[1]
        kh, kv = self._gaussian(c, x.device, x.dtype)
        mu_x = self._reduce(x, kh, kv, c)
        mu_y = self._reduce(y, kh, kv, c)
        mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sigma_xx = self._reduce(x * x, kh, kv, c) - mu_xx
        sigma_yy = self._reduce(y * y, kh, kv, c) - mu_yy
        sigma_xy = self._reduce(x * y, kh, kv, c) - mu_xy

        luminance = (2 * mu_xy + self.C1) / (mu_xx + mu_yy + self.C1)
        cs = (2 * sigma_xy + self.C2) / (sigma_xx + sigma_yy + self.C2)
        # Mean over spatial dims and channels, per image. Accumulated in float64: this is the
        # exact reduction where TF's float32 path loses ~9e-04 on a few-hundred-thousand-element
        # tensor, and a full-resolution MIDD image is 4M elements per channel. The convolutions
        # above stay float32 (validated to ~1e-07 against a float64 reference); only the sum
        # needs the wider accumulator.
        return (luminance * cs).double().mean(dim=[1, 2, 3])

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        outputs = outputs.clamp(0, 1).to(self.device).float()
        labels = labels.to(self.device).float()
        for v in self._ssim(outputs, labels):
            self.total_ssim += v.item()
            self.count += 1

    def compute(self):
        if self.count == 0:
            return 0.0
        return self.total_ssim / self.count

    def __call__(self, outputs, labels):
        self.update(outputs, labels)
        return self.compute()

    def result(self):
        return self.compute()


def create_metric(out_shape):
    return Net(out_shape)


Metric = Net
