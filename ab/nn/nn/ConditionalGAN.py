

import torch
import torch.nn as nn
import os
import torchvision.utils as vutils
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms.functional import to_pil_image
from torch.nn.utils import spectral_norm

try:
    from transformers import CLIPTokenizer
except ImportError:
    raise ImportError("Please install transformers: pip install transformers")


def supported_hyperparameters():
    """Returns the set of hyperparameters supported by this model."""
    return {'lr', 'beta1'}


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class Net(nn.Module):
    class Generator(nn.Module):
        def __init__(self, noise_dim, embed_dim, hidden_dim, vocab_size, img_channels, feature_maps):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            input_dim = noise_dim + hidden_dim

            self.l1 = nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(input_dim, feature_maps * 16, 4, 1, 0, bias=False)),
                nn.BatchNorm2d(feature_maps * 16), nn.ReLU(True)
            )
            self.l2 = nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(feature_maps * 16, feature_maps * 8, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(feature_maps * 8), nn.ReLU(True)
            )
            self.l3 = nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(feature_maps * 4), nn.ReLU(True)
            )
            self.attn1 = Self_Attn(feature_maps * 4)
            self.l4 = nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(feature_maps * 2), nn.ReLU(True)
            )
            self.l5 = nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(feature_maps), nn.ReLU(True)
            )
            self.attn2 = Self_Attn(feature_maps)
            self.l6 = nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1, bias=False)),
                nn.Tanh()
            )

        def forward(self, noise, text_tokens):
            embeddings = self.embedding(text_tokens)
            _, (hidden, _) = self.lstm(embeddings)
            text_conditioning = hidden.squeeze(0)
            x = torch.cat([noise, text_conditioning], dim=1)
            x = self.l1(x.unsqueeze(2).unsqueeze(3))
            x = self.l2(x)
            x = self.l3(x)
            x = self.attn1(x)
            x = self.l4(x)
            x = self.l5(x)
            x = self.attn2(x)
            x = self.l6(x)
            return x

    class Discriminator(nn.Module):
        def __init__(self, embed_dim, hidden_dim, vocab_size, img_channels, feature_maps):
            super().__init__()
            self.embedding = spectral_norm(nn.Embedding(vocab_size, embed_dim))
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

            self.image_path = nn.Sequential(
                spectral_norm(nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.text_path = nn.Sequential(
                spectral_norm(nn.Linear(hidden_dim, feature_maps * 2)), nn.ReLU()
            )

            self.combined_path1 = nn.Sequential(
                spectral_norm(nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.attn = Self_Attn(feature_maps * 8)

            self.combined_path2 = nn.Sequential(
                spectral_norm(nn.Conv2d(feature_maps * 8, feature_maps * 16, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(feature_maps * 16, 1, kernel_size=8, stride=1, padding=0, bias=False))
            )

        def forward(self, image, text_tokens):
            image_features = self.image_path(image)
            embeddings = self.embedding(text_tokens)
            _, (hidden, _) = self.lstm(embeddings)
            text_conditioning = hidden.squeeze(0)
            text_features = self.text_path(text_conditioning)
            _, _, H, W = image_features.shape
            text_features_replicated = text_features.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
            combined_features = torch.cat([image_features, text_features_replicated], dim=1)
            x = self.combined_path1(combined_features)
            x = self.attn(x)
            x = self.combined_path2(x)
            return x.view(-1)

    def __init__(self, shape_a, shape_b, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.prm = prm
        self.vocab_size = 49408
        img_channels = 3
        self.noise_dim = 100
        embed_dim, hidden_dim, feature_maps = 64, 128, 48
        self.save_interval = 1
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.max_length = 16
        self.generator = self.Generator(
            self.noise_dim, embed_dim, hidden_dim, self.vocab_size, img_channels, feature_maps
        ).to(device)
        self.discriminator = self.Discriminator(
            embed_dim, hidden_dim, self.vocab_size, img_channels, feature_maps
        ).to(device)
        self.fixed_noise = torch.randn(8, self.noise_dim, device=device)
        self.fixed_prompts = [
            "a red car parked on the street", "a blue car driving on a highway",
            "a photo of a black car in a garage", "a white car covered in snow",
            "a yellow sports car", "an old rusty car in a field",
            "a shiny new green car", "a silver car at night"
        ]
        self.fixed_tokenized_prompts = self.tokenizer(
            self.fixed_prompts, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )['input_ids'].to(device)
        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.gen_checkpoint_path = os.path.join(self.checkpoint_dir, 'generator.pth')
        self.disc_checkpoint_path = os.path.join(self.checkpoint_dir, 'discriminator.pth')
        self.opt_gen_checkpoint_path = os.path.join(self.checkpoint_dir, 'optimizer_g.pth')
        self.opt_disc_checkpoint_path = os.path.join(self.checkpoint_dir, 'optimizer_d.pth')
        self.epochs_trained = 0
        self.r1_penalty_weight = 10.0

    def train_setup(self, prm):
        self.to(self.device)
        lr = float(prm.get('lr', 0.0002))
        beta1 = float(prm.get('beta1', 0.5))


        lr_g = lr / 4.0
        lr_d = lr

        print(f"--- Setting Up Optimizers --- G_lr: {lr_g}, D_lr: {lr_d}")

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(beta1, 0.999))

        # --- FIX: Align scheduler with longer training run ---
        total_epochs = 150

        def lr_lambda(epoch):
            # Keep LR constant for the first half, then decay linearly
            if epoch < total_epochs / 2:
                return 1.0
            else:
                return 1.0 - (epoch - total_epochs / 2) / (total_epochs / 2)

        self.scheduler_G = LambdaLR(self.optimizer_G, lr_lambda=lr_lambda)
        self.scheduler_D = LambdaLR(self.optimizer_D, lr_lambda=lr_lambda)

        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        torch.backends.cudnn.benchmark = True

        if os.path.exists(self.gen_checkpoint_path):
            try:
                self.generator.load_state_dict(torch.load(self.gen_checkpoint_path, map_location=self.device))
                self.discriminator.load_state_dict(torch.load(self.disc_checkpoint_path, map_location=self.device))
                self.optimizer_G.load_state_dict(torch.load(self.opt_gen_checkpoint_path, map_location=self.device))
                self.optimizer_D.load_state_dict(torch.load(self.opt_disc_checkpoint_path, map_location=self.device))
                print("--- Checkpoints loaded. Resuming training. ---")
            except Exception as e:
                print(f"Could not load checkpoints, starting from scratch. Error: {e}")

    def learn(self, train_data, current_epoch=0):
        epoch_to_run = self.epochs_trained
        print(f"--- Starting Epoch {epoch_to_run} ---")
        for i, data_batch in enumerate(train_data):
            self.generator.train()
            self.discriminator.train()
            real_images, raw_text_prompts = data_batch

            tokenized_prompts = self.tokenizer(
                list(raw_text_prompts), padding='max_length', truncation=True,
                max_length=self.max_length, return_tensors="pt"
            )
            text_tokens = tokenized_prompts['input_ids'].to(self.device)
            real_images = real_images.to(self.device)
            b_size = real_images.size(0)

            real_target = torch.full((b_size,), 0.9, device=self.device)
            fake_target = torch.full((b_size,), 0.1, device=self.device)


            for _ in range(2):
                self.optimizer_D.zero_grad()
                real_images.requires_grad = True

                # Train with real images
                output_real = self.discriminator(real_images, text_tokens)
                loss_d_real = self.criterion(output_real, real_target)
                grad_real = torch.autograd.grad(outputs=output_real.sum(), inputs=real_images, create_graph=True)[0]
                grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                r1_penalty = self.r1_penalty_weight / 2 * grad_penalty

                # Train with fake images
                with torch.no_grad():
                    noise = torch.randn(b_size, self.noise_dim, device=self.device)
                    fake_images = self.generator(noise, text_tokens).detach()
                output_fake = self.discriminator(fake_images, text_tokens)
                loss_d_fake = self.criterion(output_fake, fake_target)

                # Total discriminator loss
                loss_d = loss_d_real + loss_d_fake + r1_penalty
                loss_d.backward()
                self.optimizer_D.step()
                real_images.requires_grad = False

            # Update Generator once
            self.optimizer_G.zero_grad()
            generator_real_target = torch.full((b_size,), 1.0, device=self.device)
            noise_g = torch.randn(b_size, self.noise_dim, device=self.device)
            fake_images_for_g = self.generator(noise_g, text_tokens)
            output_g = self.discriminator(fake_images_for_g, text_tokens)
            loss_g = self.criterion(output_g, generator_real_target)
            loss_g.backward()
            self.optimizer_G.step()

            if i % 100 == 0:
                print(
                    f'[{epoch_to_run}][{i}/{len(train_data)}] Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f} R1 Penalty: {r1_penalty.item():.4f}')

        self.scheduler_G.step()
        self.scheduler_D.step()

        if (epoch_to_run + 1) % self.save_interval == 0:
            with torch.no_grad():
                self.generator.eval()
                fake_samples = self.generator(self.fixed_noise, self.fixed_tokenized_prompts).detach().cpu()
                torch.save(self.generator.state_dict(), self.gen_checkpoint_path)
                torch.save(self.discriminator.state_dict(), self.disc_checkpoint_path)
                torch.save(self.optimizer_G.state_dict(), self.opt_gen_checkpoint_path)
                torch.save(self.optimizer_D.state_dict(), self.opt_disc_checkpoint_path)
                os.makedirs('samples', exist_ok=True)
                vutils.save_image(fake_samples, f'samples/epoch_{epoch_to_run:03d}.png', normalize=True, nrow=4)
                vutils.save_image(fake_samples, f'samples/latest_samples.png', normalize=True, nrow=4)
                print(f"--- Checkpoint saved for epoch {epoch_to_run}. ---")
        else:
            print(f"--- Epoch {epoch_to_run} complete. No checkpoint saved. ---")
        self.epochs_trained += 1
        return loss_g.item()

    def forward(self, input_tensor: torch.Tensor, text_prompts: list = None) -> torch.Tensor:
        self.generator.eval()

        # --- FIX: Robust handling of text_prompts to prevent "ambiguous boolean" error ---
        prompts_to_use = self.fixed_prompts
        if text_prompts is not None:
            valid_prompts = [p for p in text_prompts if isinstance(p, str) and p.strip()]
            if valid_prompts:
                prompts_to_use = valid_prompts

        tokenized_prompts = self.tokenizer(
            prompts_to_use, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )['input_ids'].to(self.device)

        noise = torch.randn(len(prompts_to_use), self.noise_dim, device=self.device)

        with torch.no_grad():
            generated_tensors = self.generator(noise, tokenized_prompts)
            generated_tensors = generated_tensors * 0.5 + 0.5  # Denormalize
            return (generated_tensors, prompts_to_use)