
import torch
import torch.nn as nn
import numpy as np
import os
import glob
from PIL import Image
import itertools

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import AutoTokenizer, AutoModel

# Optional import for 8-bit optimizer
try:
    import bitsandbytes as bnb

    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False


def supported_hyperparameters():
    """Returns the hyperparameters supported by this model."""
    return {'lr', 'beta1', 'beta2', 'steps_per_epoch'}


class Net(nn.Module):
    """
    The main Net class that holds the Diffusion components and implements the
    framework's training and evaluation logic.
    """

    class TextEncoder(nn.Module):
        def __init__(self, out_size=768):
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
            outputs = self.text_model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device)
            )
            return self.text_linear(outputs.last_hidden_state)

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.prm = prm or {}
        self.epoch_counter = 0
        self.model_name = "CLDiffusion"

        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        self.vae.requires_grad_(False)

        self.text_encoder = self.TextEncoder(out_size=prm.get('cross_attention_dim', 768)).to(device)

        latent_size = in_shape[2] // 8

        self.unet = UNet2DConditionModel(
            sample_size=latent_size,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            block_out_channels=(128, 256, 512),
            cross_attention_dim=prm.get('cross_attention_dim', 768)
        ).to(device)

        #  Enable Memory-Efficient Attention (xFormers) if available
        try:
            self.unet.enable_xformers_memory_efficient_attention()
            print("xFormers memory-efficient attention enabled.")
        except Exception:
            print("xFormers not available. Using standard attention.")

        # Gradient checkpointing is already enabled, which is great for memory saving.
        self.unet.enable_gradient_checkpointing()
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

        #  Setup for Mixed-Precision Training
        self.scaler = torch.cuda.amp.GradScaler()

        # self.checkpoint_dir = os.path.join("checkpoints", self.model_name)
        # if not os.path.exists(self.checkpoint_dir):
        #     os.makedirs(self.checkpoint_dir)
        # self.load_checkpoint()

    # def load_checkpoint(self):
    #     # (omitted for brevity - no changes from previous version)
    #     unet_files = glob.glob(os.path.join(self.checkpoint_dir, f'{self.model_name}_unet_epoch_*.pth'))
    #     text_encoder_files = glob.glob(os.path.join(self.checkpoint_dir, f'{self.model_name}_text_encoder_epoch_*.pth'))
    #
    #     if unet_files and text_encoder_files:
    #         latest_unet = max(unet_files, key=os.path.getctime)
    #         latest_text_encoder = max(text_encoder_files, key=os.path.getctime)
    #         print(f"Loading UNet checkpoint: {latest_unet}")
    #         print(f"Loading Text Encoder checkpoint: {latest_text_encoder}")
    #         self.unet.load_state_dict(torch.load(latest_unet, map_location=self.device))
    #         self.text_encoder.load_state_dict(torch.load(latest_text_encoder, map_location=self.device))
    #         try:
    #             self.epoch_counter = int(os.path.basename(latest_unet).split('_')[-1].split('.')[0])
    #         except (ValueError, IndexError):
    #             self.epoch_counter = 0
    #     else:
    #         print("No checkpoint found, starting from scratch.")

    def train_setup(self, prm):
        trainable_params = list(self.unet.parameters()) + list(self.text_encoder.text_linear.parameters())
        lr = prm['lr']
        beta1 = prm['beta1']
        beta2 = prm['beta2']

        #  Optional 8-bit Optimizer
        # To use, ensure 'bitsandbytes' is installed and uncomment the following lines.
        # if BITSANDBYTES_AVAILABLE:
        #     print("Using 8-bit AdamW optimizer.")
        #     self.optimizer = bnb.optim.AdamW8bit(trainable_params, lr=lr, betas=(beta1, 0.999))
        # else:
        #     print("Using standard AdamW optimizer.")
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(beta1, beta2))

        self.criterion = nn.MSELoss()
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=50, factor=0.5)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0

        if not hasattr(self, 'infinite_data_loader'):
            self.infinite_data_loader = itertools.cycle(train_data)

        num_steps = int(self.prm['steps_per_epoch'] * 400)

        if num_steps == 0:
            print("Warning: 'steps_per_epoch' is zero. Skipping training for this epoch.")
            return 0.0

        for i in range(num_steps):
            batch = next(self.infinite_data_loader)
            images, text_prompts = batch
            self.optimizer.zero_grad()

            with torch.no_grad():
                latents = self.vae.encode(images.to(self.device)).latent_dist.sample() * 0.18215

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                      device=self.device)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            text_embeddings = self.text_encoder(text_prompts)

            # --- NEW: Mixed-Precision Training Context ---
            with torch.cuda.amp.autocast():
                noise_pred = self.unet(sample=noisy_latents, timestep=timesteps,
                                       encoder_hidden_states=text_embeddings).sample
                loss = self.criterion(noise_pred, noise)

            #  Scale loss and update weights ---
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        self.epoch_counter += 1

        # unet_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_unet_epoch_{self.epoch_counter}.pth")
        # text_encoder_path = os.path.join(self.checkpoint_dir,
        #                                  f"{self.model_name}_text_encoder_epoch_{self.epoch_counter}.pth")
        # torch.save(self.unet.state_dict(), unet_path)
        # torch.save(self.text_encoder.state_dict(), text_encoder_path)
        # print(f"\nCompleted epoch {self.epoch_counter}. Saved checkpoint to {unet_path} and {text_encoder_path}")

        return total_loss / num_steps

    @torch.no_grad()
    def generate(self, text_prompts, num_inference_steps=50):
        # (omitted for brevity)
        self.eval()
        text_embeddings = self.text_encoder(text_prompts)
        latents = torch.randn((len(text_prompts), self.unet.config.in_channels, self.unet.config.sample_size,
                               self.unet.config.sample_size), device=self.device)
        self.noise_scheduler.set_timesteps(num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.unet(sample=latents, timestep=t, encoder_hidden_states=text_embeddings).sample
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents
        images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        return [Image.fromarray((img * 255).astype(np.uint8)) for img in images]

    @torch.no_grad()
    def forward(self, images, **kwargs):
        # (omitted for brevity - no changes from previous version)
        batch_size = images.size(0)
        fixed_prompts_for_eval = [
            "a photo of a dog", "a painting of a car", "a smiling person"
        ]
        prompts_to_use = [fixed_prompts_for_eval[i % len(fixed_prompts_for_eval)] for i in range(batch_size)]

        output_dir = os.path.join("output_images", self.model_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # custom_prompts_to_generate = [
        #     "a smiling woman with blond hair",
        #     "a man wearing eyeglasses"
        # ]
        # if custom_prompts_to_generate:
        #     print(f"\n[Inference] Generating {len(custom_prompts_to_generate)} custom image(s)...")
        #     custom_images = self.generate(custom_prompts_to_generate)
        #     for i, img in enumerate(custom_images):
        #         save_path = os.path.join(output_dir,
        #                                  f"{self.model_name}_output_epoch_{self.epoch_counter}_image_{i + 1}.png")
        #         img.save(save_path)
        #         print(f"[Inference] Saved custom image to {save_path}")

        eval_images = self.generate(prompts_to_use)

        return eval_images, prompts_to_use
