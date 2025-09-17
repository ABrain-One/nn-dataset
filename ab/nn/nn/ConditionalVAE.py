
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import math
import os
import glob

from transformers import AutoTokenizer, AutoModel


def supported_hyperparameters():
    """Returns the hyperparameters supported by this model."""
    return {'lr', 'beta1', 'beta2', 'kld_warmup_epochs'}


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
            model_name = "distilbert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_model = AutoModel.from_pretrained(model_name)
            self.text_linear = nn.Linear(768, out_size)
            for param in self.text_model.parameters():
                param.requires_grad = False

        def forward(self, text):
            device = self.text_linear.weight.device
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.text_model(input_ids=inputs.input_ids.to(device),
                                      attention_mask=inputs.attention_mask.to(device))
            return self.text_linear(outputs.last_hidden_state.mean(dim=1))

    class CVAE(nn.Module):
        def __init__(self, latent_dim=512, text_embedding_dim=128, image_channels=3, image_size=256):
            super().__init__()
            self.latent_dim = latent_dim

            # This architecture is for 256x256 images
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
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(1024, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                SelfAttention(256),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(64, image_channels, kernel_size=3, padding=1),
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

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.prm = prm or {}
        self.text_embedding_dim = 128
        self.latent_dim = 512
        self.epoch_counter = 0
        self.model_name = "ConditionalVAE_Final"

        image_channels, image_size = in_shape[1], in_shape[2]
        self.text_encoder = self.TextEncoder(out_size=self.text_embedding_dim).to(device)
        self.cvae = self.CVAE(self.latent_dim, self.text_embedding_dim, image_channels, image_size).to(device)

        lr = self.prm['lr']
        beta1 = self.prm['beta1']
        beta2 = self.prm['beta2']

        self.optimizer = torch.optim.Adam(self.cvae.parameters(), lr=lr, betas=(beta1, beta2))
        self.reconstruction_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss().to(device)

        # resume_flag = os.getenv('RESUME_TRAINING', 'true').lower()
        #
        # print("------------------------------------------")
        # print(f"DEBUG: RESUME_TRAINING flag is set to '{resume_flag}'")
        #
        # if resume_flag == 'true':
        #     print(f"DEBUG: Searching for checkpoints in: {self.checkpoint_dir}")
        #     list_of_files = glob.glob(os.path.join(self.checkpoint_dir, f'{self.model_name}_epoch_*.pth'))
        #     print(f"DEBUG: Found checkpoint files: {list_of_files}")
        #
        #     if list_of_files:
        #         latest_file = max(list_of_files, key=os.path.getctime)
        #         print(f"DEBUG: Latest file found: {latest_file}")
        #         print(f"RESUME_TRAINING=true. Loading checkpoint: {latest_file}")
        #         self.cvae.load_state_dict(torch.load(latest_file, map_location=self.device))
        #         self.epoch_counter = int(os.path.basename(latest_file).split('_')[-1].split('.')[0])
        #         print(f"DEBUG: Setting epoch counter to: {self.epoch_counter}")
        #     else:
        #         print("RESUME_TRAINING=true, but no checkpoint found. Starting from scratch.")
        # else:
        #     print("RESUME_TRAINING=false. Starting a fresh training run.")
        # print("------------------------------------------")

    def train_setup(self, prm):
        pass

    def learn(self, train_data):
        self.train()
        total_loss = 0.0

        kld_warmup_epochs = int(50 * self.prm['kld_warmup_epochs'])
        max_kld_weight = 0.000025

        current_epoch = self.epoch_counter
        if current_epoch < kld_warmup_epochs:
            kld_weight = max_kld_weight * (current_epoch / kld_warmup_epochs)
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

        # --- THE FIX: Increment the counter here, after the training epoch is complete ---
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
        batch_size = images.size(0)
        fixed_prompts_for_eval = ["a car",
                                  "a car on a road",
                                  "a car parked"]
        prompts_to_use = [fixed_prompts_for_eval[i % len(fixed_prompts_for_eval)] for i in range(batch_size)]

        # output_dir = os.path.join("output_images", self.model_name)
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        # custom_prompts_to_generate = ["a red car",
        #                               "a blue car"]
        # if custom_prompts_to_generate:
        #     custom_images = self.generate(custom_prompts_to_generate)
        #     for i, img in enumerate(custom_images):
        #         # Use the current epoch_counter for saving images
        #         save_path = os.path.join(output_dir,
        #                                  f"{self.model_name}_output_epoch_{self.epoch_counter}_image_{i + 1}.png")
        #         img.save(save_path)

        #  The counter is no longer incremented here
        # checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_epoch_{self.epoch_counter}.pth")
        # torch.save(self.cvae.state_dict(), checkpoint_path)

        return self.generate(prompts_to_use), prompts_to_use
