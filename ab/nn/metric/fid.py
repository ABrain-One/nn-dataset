# File: fid.py
# Location: ab/nn/metric/
# Description: This file defines the FIDMetric class, with an updated result
# method to be compatible with the framework's maximization objective.

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from scipy import linalg
from torchvision.models import inception_v3
import torchvision.transforms as transforms

# A global cache for the InceptionV3 model to avoid reloading it on every call.
_inception_model_cache = {}

class InceptionV3FeatureExtractor(nn.Module):
    """
    A wrapper for the InceptionV3 model to extract features from the
    final average pooling layer, which is standard for FID calculation.
    """

    def __init__(self):
        super().__init__()
        # Load a pre-trained InceptionV3 model
        self.inception_v3 = inception_v3(weights='Inception_V3_Weights.DEFAULT', aux_logits=False)
        # The FID calculation uses features from the last pooling layer
        self.inception_v3.fc = nn.Identity()

    def forward(self, x):
        return self.inception_v3(x)


def _get_inception_model(device):
    """ Helper function to load or retrieve the cached InceptionV3 model. """
    if 'model' not in _inception_model_cache:
        print(f"\n[FIDMetric] Caching InceptionV3 model for the first time...")
        model = InceptionV3FeatureExtractor().to(device)
        _inception_model_cache['model'] = model
        print("[FIDMetric] Model cached successfully.")

    _inception_model_cache['model'].to(device)
    return _inception_model_cache['model']


class FIDMetric:
    """
    A stateful metric class for calculating the Fréchet Inception Distance (FID).
    """

    def __init__(self, out_shape=None, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.inception_model = _get_inception_model(self.device)
        self.reset()

    def reset(self):
        """ Clears the accumulated features for a new evaluation phase. """
        self.real_features = []
        self.fake_features = []

    def __call__(self, preds, labels):
        """
        Processes a batch of predictions (fake images) and labels (real images).
        """
        if not preds or labels is None:
            return

        self.inception_model.eval()
        real_images_unnorm = (labels.to(self.device) + 1) / 2
        real_images_transformed = torch.stack(
            [self.transform(transforms.ToPILImage()(img)) for img in real_images_unnorm])
        fake_images_transformed = torch.stack([self.transform(img) for img in preds])

        with torch.no_grad():
            real_feats = self.inception_model(real_images_transformed.to(self.device))
            fake_feats = self.inception_model(fake_images_transformed.to(self.device))
            self.real_features.append(real_feats.cpu().numpy())
            self.fake_features.append(fake_feats.cpu().numpy())

    def _calculate_fid(self, mu1, sigma1, mu2, sigma2):
        """ NumPy implementation of the Frechet Distance. """
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def result(self):
        """
        Calculates the final score. Lower FID is better, so we return the
        negative value for the framework to maximize.
        """
        if not self.real_features or not self.fake_features:
            return -float('inf')  # Return a very bad score if no data

        real_features_all = np.concatenate(self.real_features, axis=0)
        fake_features_all = np.concatenate(self.fake_features, axis=0)
        mu_real, sigma_real = np.mean(real_features_all, axis=0), np.cov(real_features_all, rowvar=False)
        mu_fake, sigma_fake = np.mean(fake_features_all, axis=0), np.cov(fake_features_all, rowvar=False)

        fid_score = self._calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)

        # --- FIX as per supervisor's notes ---
        # The framework maximizes the result, so we return the negative FID score.
        return -float(fid_score)

    def get_all(self):
        """ Returns a dictionary of all computed metrics. """
        # We calculate the real FID score here for logging purposes.
        fid_score = -self.result() if (self.real_features and self.fake_features) else float('inf')
        return {'FID_Score': fid_score}


def create_metric(out_shape=None, device=None):
    """ Factory function used by the LEMUR framework. """
    return FIDMetric(out_shape=out_shape, device=device)

