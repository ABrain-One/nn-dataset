# File: ConditionalVAE_GAN_Final_Optimized.py
# Description: This is the final, optimized version of the VAE-GAN, incorporating all
#              stability improvements including a weaker discriminator, robust checkpointing
#              with score reset, asymmetrical learning rates, and re-balanced loss weights.

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import math
import os
import glob

from transformers import CLIPTextModel, CLIPTokenizer


def supported_hyperparameters():
    """Returns the hyperparameters supported by this model."""
    return {'lr', 'momentum', 'version', 'lr_g', 'lr_d'}


class PerceptualLoss(nn.Module):
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
    class TextEncoder(nn.Module):
        def __init__(self, out_size=128):
            super().__init__()
            model_name = "openai/clip-vit-base-patch32"
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.text_model = CLIPTextModel.from_pretrained(model_name)
            self.text_linear = nn.Linear(512, out_size)
            for param in self.text_model.parameters():
                param.requires_grad = False

        def forward(self, text):
            device = self.text_linear.weight.device
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.text_model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device)
            )
            return self.text_linear(outputs.pooler_output)

    class CVAE(nn.Module):
        class UpsampleBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
                self.pixel_shuffle = nn.PixelShuffle(2)
                self.lrelu = nn.LeakyReLU(0.2, inplace=True)

            def forward(self, x):
                x = self.conv(x)
                x = self.pixel_shuffle(x)
                x = self.lrelu(x)
                return x

        def __init__(self, latent_dim=512, text_embedding_dim=128, image_channels=3, image_size=256):
            super().__init__()
            self.latent_dim = latent_dim
            self.encoder_conv = nn.Sequential(
                nn.Conv2d(image_channels, 32, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 512, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)
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
                self.UpsampleBlock(512, 512),
                self.UpsampleBlock(512, 256),
                self.UpsampleBlock(256, 128),
                SelfAttention(128),
                self.UpsampleBlock(128, 64),
                self.UpsampleBlock(64, 32),
                self.UpsampleBlock(32, 16),
                nn.Conv2d(16, image_channels, kernel_size=3, padding=1),
                nn.Tanh()
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

    class Discriminator(nn.Module):
        def __init__(self, image_channels=3):
            super().__init__()
            # --- Weakened Discriminator Architecture ---
            self.model = nn.Sequential(
                nn.Conv2d(image_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 1, 4, 1, 0)
            )

        def forward(self, x):
            return self.model(x)

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.prm = prm or {}
        self.text_embedding_dim = 128
        self.latent_dim = 512
        self.model_name = "ConditionalVAE_GAN"

        self.register_buffer('epoch_counter', torch.tensor(0))
        self.register_buffer('best_score_so_far', torch.tensor(-1.0))

        self.checkpoint_dir = os.path.join("checkpoints", self.model_name)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        image_channels, image_size = in_shape[1], in_shape[2]
        self.text_encoder = self.TextEncoder(out_size=self.text_embedding_dim).to(device)
        self.cvae = self.CVAE(self.latent_dim, self.text_embedding_dim, image_channels, image_size).to(device)
        self.discriminator = self.Discriminator(image_channels).to(device)

        # Asymmetrical learning rates for Generator and Discriminator
        lr_g = self.prm.get('lr', 2e-6)
        lr_d = self.prm.get('min_learning_rate', 2e-7)
        beta1 = self.prm.get('momentum', 0.5)

        print(f"--- Initializing VAE-GAN with G_LR: {lr_g} | D_LR: {lr_d} | Beta1: {beta1} ---")

        self.optimizer_g = torch.optim.Adam(self.cvae.parameters(), lr=lr_g, betas=(beta1, 0.999))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(beta1, 0.999))

        self.reconstruction_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss().to(device)
        self.adversarial_loss = nn.BCEWithLogitsLoss()

        # Robust resume logic with score reset
        resume_flag = os.getenv('RESUME_TRAINING', 'true').lower()
        source_checkpoint_path = os.path.join("checkpoints", "ConditionalVAE3", "best_model.pth")
        gan_checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")

        if resume_flag == 'true' and os.path.exists(gan_checkpoint_path):
            print(f"Resuming GAN training. Loading checkpoint from: {gan_checkpoint_path}")
            self.load_state_dict(torch.load(gan_checkpoint_path, map_location=device))
            print(f"Resumed from Epoch {self.epoch_counter.item() + 1}. Best score so far: {self.best_score_so_far.item():.4f}")
        elif resume_flag == 'true' and os.path.exists(source_checkpoint_path):
            print(f"No GAN checkpoint found. Initializing with VAE weights from: {source_checkpoint_path}")
            self.load_state_dict(torch.load(source_checkpoint_path, map_location=device), strict=False)

            print(f"Loaded pre-trained VAE score of {self.best_score_so_far.item():.4f}. Resetting for new GAN training.")
            self.best_score_so_far = torch.tensor(-1.0)

            print(f"Starting GAN fine-tuning from Epoch {self.epoch_counter.item() + 1}. New best score tracker initialized.")
        else:
            print("No checkpoint found. Starting a fresh training run.")

    def train_setup(self, prm):
        pass

    def learn(self, train_data):
        self.train()
        total_g_loss = 0.0
        total_d_loss = 0.0

        # --- Final, conservative loss weights for stability ---
        recon_weight = 10.0
        perc_weight = 1.0
        kld_weight = 0.0000025
        adversarial_weight = 0.0001

        for batch in train_data:
            real_images, text_prompts = batch
            real_images = real_images.to(self.device)

            # --- (1) Train the Discriminator ---
            self.optimizer_d.zero_grad()

            with torch.no_grad():
                text_embeddings = self.text_encoder(text_prompts)
                mu, log_var = self.cvae.encode(real_images, text_embeddings)
                z = self.cvae.reparameterize(mu, log_var)
                reconstructed_images = self.cvae.decode(z, text_embeddings)

            real_output = self.discriminator(real_images)

            real_labels = torch.ones_like(real_output, device=self.device)
            fake_labels = torch.zeros_like(real_output, device=self.device)

            d_loss_real = self.adversarial_loss(real_output, real_labels)
            fake_output = self.discriminator(reconstructed_images.detach())
            d_loss_fake = self.adversarial_loss(fake_output, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            self.optimizer_d.step()
            total_d_loss += d_loss.item()

            # --- (2) Train the VAE (Generator) ---
            self.optimizer_g.zero_grad()

            text_embeddings = self.text_encoder(text_prompts)
            mu, log_var = self.cvae.encode(real_images, text_embeddings)
            z = self.cvae.reparameterize(mu, log_var)
            reconstructed_images_for_g = self.cvae.decode(z, text_embeddings)

            recon_loss = self.reconstruction_loss(reconstructed_images_for_g, real_images)
            perc_loss = self.perceptual_loss(reconstructed_images_for_g, real_images)
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            fake_output_for_g = self.discriminator(reconstructed_images_for_g)
            g_loss_adv = self.adversarial_loss(fake_output_for_g, real_labels)

            g_loss = (recon_weight * recon_loss) + (perc_weight * perc_loss) + (kld_weight * kld_loss) + (adversarial_weight * g_loss_adv)

            g_loss.backward()
            self.optimizer_g.step()
            total_g_loss += g_loss.item()

        avg_g_loss = total_g_loss / len(train_data) if train_data else 0.0
        avg_d_loss = total_d_loss / len(train_data) if train_data else 0.0

        print(f"Epoch {self.epoch_counter.item()} - G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}")
        return avg_g_loss

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
        self.epoch_counter += 1
        prompts_to_use = kwargs.get('prompts')
        if not prompts_to_use:
            batch_size = images.size(0)
            default_prompts = ["a photo of a car"]
            prompts_to_use = [default_prompts[i % len(default_prompts)] for i in range(batch_size)]

        return self.generate(prompts_to_use), prompts_to_use

    def save_if_best(self, current_score):
        if current_score > self.best_score_so_far.item():
            self.best_score_so_far = torch.tensor(current_score)
            best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            print(f"\n--- New best score: {current_score:.4f} at epoch {self.epoch_counter.item()}! Saving checkpoint to {best_checkpoint_path} ---")
            torch.save(self.state_dict(), best_checkpoint_path)