# File: ConditionalVAE-2.py
# Description: A refactored and enhanced version that aligns with supervisor
#              feedback. This model is stateless and uses a powerful CLIP text
#              encoder for improved performance.

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import math
import os

# --- UPDATE: Import CLIP-specific components from transformers ---
from transformers import CLIPTextModel, CLIPTokenizer


def supported_hyperparameters():
    """Returns the hyperparameters supported by this model."""
    return {'lr', 'momentum', 'version'}


class PerceptualLoss(nn.Module):
    """A loss function that compares high-level features from a VGG network."""

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:23].eval()
        self.vgg = nn.Sequential(*vgg)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.l1 = nn.L1Loss()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, y_hat, y):
        y_hat_norm = self.normalize(y_hat)
        y_norm = self.normalize(y)
        vgg_y_hat = self.vgg(y_hat_norm)
        vgg_y = self.vgg(y_norm)
        return self.l1(vgg_y_hat, vgg_y)


class SelfAttention(nn.Module):
    """A Self-Attention layer to help the model learn spatial relationships."""

    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        attention = torch.bmm(query, key).softmax(dim=-1)
        value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        return self.gamma * out + x


class Net(nn.Module):
    # --- PERFORMANCE ENHANCEMENT: Replaced DistilBERT with CLIP Text Encoder ---
    class TextEncoder(nn.Module):
        def __init__(self, out_size=128):
            super().__init__()
            model_name = "openai/clip-vit-base-patch32"
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.text_model = CLIPTextModel.from_pretrained(model_name)

            # CLIP's native output dimension is 512. We add a linear layer
            # to project this down to the embedding size our CVAE expects.
            self.text_linear = nn.Linear(512, out_size)

            # Freeze the pre-trained weights to save memory and training time
            for param in self.text_model.parameters():
                param.requires_grad = False

        def forward(self, text):
            device = self.text_linear.weight.device
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.text_model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device)
            )
            # Use the pooler_output for a single summary vector of the text
            return self.text_linear(outputs.pooler_output)

    class CVAE(nn.Module):
        def __init__(self, latent_dim=512, text_embedding_dim=128, image_channels=3, image_size=256):
            super().__init__()
            self.latent_dim = latent_dim
            self.encoder_conv = nn.Sequential(
                nn.Conv2d(image_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 1024, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(1024, 1024, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)
            )
            with torch.no_grad():
                dummy_input = torch.zeros(1, image_channels, image_size, image_size)
                dummy_output = self.encoder_conv(dummy_input)
                self.final_feature_dim = dummy_output.view(-1).shape[0]
                self.final_conv_shape = dummy_output.shape
            combined_dim = self.final_feature_dim + text_embedding_dim
            self.fc_mu = nn.Linear(combined_dim, latent_dim)
            self.fc_log_var = nn.Linear(combined_dim, latent_dim)
            self.decoder_input = nn.Linear(latent_dim + text_embedding_dim, self.final_feature_dim)
            self.decoder_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(1024, 512, kernel_size=3, padding=1), nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.LeakyReLU(0.2, inplace=True),
                SelfAttention(256),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(64, image_channels, kernel_size=3, padding=1), nn.Tanh()
            )

        def encode(self, image, text_embedding):
            x = self.encoder_conv(image)
            x = torch.flatten(x, start_dim=1)
            combined = torch.cat([x, text_embedding], dim=1)
            return self.fc_mu(combined), self.fc_log_var(combined)

        def reparameterize(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z, text_embedding):
            combined = torch.cat([z, text_embedding], dim=1)
            x = self.decoder_input(combined)
            x = x.view(-1, *self.final_conv_shape[1:])
            return self.decoder_conv(x)

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.prm = prm or {}
        self.text_embedding_dim = 128
        self.latent_dim = 512
        self.epoch_counter = 0  # This model is stateless and resets with each run

        image_channels, image_size = in_shape[1], in_shape[2]
        self.text_encoder = self.TextEncoder(out_size=self.text_embedding_dim).to(device)
        self.cvae = self.CVAE(self.latent_dim, self.text_embedding_dim, image_channels, image_size).to(device)

        lr = self.prm.get('lr', 1e-4)
        beta1 = self.prm.get('momentum', 0.9)
        self.optimizer = torch.optim.Adam(self.cvae.parameters(), lr=lr, betas=(beta1, 0.999))
        self.reconstruction_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss().to(device)

    def train_setup(self, prm):
        pass

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        kld_warmup_epochs = 25
        max_kld_weight = 0.000025

        if self.epoch_counter < kld_warmup_epochs:
            kld_weight = max_kld_weight * (self.epoch_counter / kld_warmup_epochs)
        else:
            kld_weight = max_kld_weight

        for batch in train_data:
            real_images, text_prompts = batch
            real_images = real_images.to(self.device)
            self.optimizer.zero_grad()
            text_embeddings = self.text_encoder(text_prompts)
            mu, log_var = self.cvae.encode(real_images, text_embeddings)
            z = self.cvae.reparameterize(mu, log_var)
            reconstructed_images = self.cvae.decode(z, text_embeddings)
            recon_loss = self.reconstruction_loss(reconstructed_images, real_images)
            perc_loss = self.perceptual_loss(reconstructed_images, real_images)
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + 0.5 * perc_loss + (kld_weight * kld_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cvae.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()

        self.epoch_counter += 1
        return total_loss / len(train_data) if train_data else 0.0

    @torch.no_grad()
    def generate(self, text_prompts):
        self.eval()
        num_images = len(text_prompts)
        z = torch.randn(num_images, self.latent_dim, device=self.device)
        text_embeddings = self.text_encoder(text_prompts)
        generated_images = self.cvae.decode(z, text_embeddings)
        generated_images = (generated_images + 1) / 2
        return [T.ToPILImage()(img.cpu()) for img in generated_images]

    @torch.no_grad()
    def forward(self, images, **kwargs):
        # Prompts are now expected to be passed in from the evaluation script
        prompts_to_use = kwargs.get('prompts')

        # Fallback for safety if no prompts are provided
        if not prompts_to_use:
            batch_size = images.size(0)
            default_prompts = ["a photo of a car"]
            prompts_to_use = [default_prompts[i % len(default_prompts)] for i in range(batch_size)]

        return self.generate(prompts_to_use), prompts_to_use
