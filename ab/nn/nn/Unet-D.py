import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2 import ToPILImage
from tqdm import tqdm
import os
import glob
import math
import random
from torch.utils.checkpoint import checkpoint
from copy import deepcopy

#To solve Hugging Face Tokenizer Warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#Performance Optimization: Enable TF32
torch.set_float32_matmul_precision('high')

try:
    from transformers import CLIPTextModel, CLIPTokenizer
except ImportError:
    raise ImportError("Please install 'transformers' for the Diffusion model: pip install transformers")

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def supported_hyperparameters():
    return {'lr': (float, 1e-4), 'timesteps': (int, 1000), 'model_channels': (int, 64), 'epochs': (int, 100)}


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = deepcopy(self.model.state_dict())

    def update(self):
        model_params = self.model.state_dict()
        for name, param in model_params.items():
            self.shadow[name].data = (self.shadow[name].data * self.decay +
                                      param.data * (1 - self.decay))

    def apply_shadow(self):
        self.original_params = deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.shadow, strict=True)

    def restore(self):
        self.model.load_state_dict(self.original_params, strict=True)


class SiLU(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)


class CrossAttention(nn.Module):
    def __init__(self, channels, context_dim):
        super().__init__()
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(context_dim, channels)
        self.value = nn.Linear(context_dim, channels)
        self.out = nn.Linear(channels, channels)

    def forward(self, x, context):
        batch, channels, height, width = x.shape
        x_reshaped = x.view(batch, channels, height * width).transpose(1, 2)
        q = self.query(x_reshaped)
        k = self.key(context)
        v = self.value(context)
        sim = torch.bmm(q, k.transpose(1, 2)) * (channels ** -0.5)
        attn = F.softmax(sim, dim=-1)
        out = torch.bmm(attn, v)
        out = self.out(out).transpose(1, 2).view(batch, channels, height, width)
        return x + out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.emb_proj = nn.Linear(emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = SiLU()

    def forward(self, x, emb):
        h = self.act(self.norm1(x));
        h = self.conv1(h)
        emb_cond = self.emb_proj(self.act(emb)).unsqueeze(-1).unsqueeze(-1)
        h = h + emb_cond;
        h = self.act(self.norm2(h));
        h = self.conv2(h)
        return h + self.shortcut(x)


class AttnResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, context_dim):
        super().__init__()
        self.res_block = ResBlock(in_channels, out_channels, emb_dim)
        self.attention = CrossAttention(out_channels, context_dim)

    def forward(self, x, emb, context):
        h = self.res_block(x, emb)
        return self.attention(h, context)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim

    def forward(self, time):
        device = time.device;
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, context_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            AttnResBlock(in_channels, out_channels, emb_dim, context_dim),
            AttnResBlock(out_channels, out_channels, emb_dim, context_dim),
        ])
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

    def forward(self, x, emb, context):
        for block in self.blocks:
            x = checkpoint(block, x, emb, context, use_reentrant=False)
        return self.downsample(x), x


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, emb_dim, context_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        self.blocks = nn.ModuleList([
            AttnResBlock(out_channels + skip_channels, out_channels, emb_dim, context_dim),
            AttnResBlock(out_channels, out_channels, emb_dim, context_dim),
        ])

    def forward(self, x, skip, emb, context):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        for block in self.blocks:
            x = checkpoint(block, x, emb, context, use_reentrant=False)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, model_channels=64, context_dim=512):
        super().__init__()
        self.time_emb_dim = model_channels * 4
        mc = model_channels

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(mc),
            nn.Linear(mc, self.time_emb_dim), SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        self.conv_in = nn.Conv2d(in_channels, mc, 3, 1, 1)

        self.down1 = DownBlock(mc, mc, self.time_emb_dim, context_dim)
        self.down2 = DownBlock(mc, mc * 2, self.time_emb_dim, context_dim)
        self.down3 = DownBlock(mc * 2, mc * 4, self.time_emb_dim, context_dim)

        self.mid1 = AttnResBlock(mc * 4, mc * 8, self.time_emb_dim, context_dim)
        self.mid2 = AttnResBlock(mc * 8, mc * 4, self.time_emb_dim, context_dim)

        self.up1 = UpBlock(mc * 4, mc * 4, mc * 2, self.time_emb_dim, context_dim)
        self.up2 = UpBlock(mc * 2, mc * 2, mc, self.time_emb_dim, context_dim)
        self.up3 = UpBlock(mc, mc, mc, self.time_emb_dim, context_dim)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(32, mc), SiLU(),
            nn.Conv2d(mc, in_channels, 3, 1, 1)
        )

    def forward(self, x, time, text_context):
        emb = self.time_embed(time)
        h = self.conv_in(x)

        h1_d, h1_s = self.down1(h, emb, text_context)
        h2_d, h2_s = self.down2(h1_d, emb, text_context)
        h3_d, h3_s = self.down3(h2_d, emb, text_context)

        h_mid = self.mid1(h3_d, emb, text_context)
        h_mid = self.mid2(h_mid, emb, text_context)

        h = self.up1(h_mid, h3_s, emb, text_context)
        h = self.up2(h, h2_s, emb, text_context)
        h = self.up3(h, h1_s, emb, text_context)

        return self.out_conv(h)


class Net(nn.Module):
    class TextEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME, use_fast=True)
            self.text_model = CLIPTextModel.from_pretrained(CLIP_MODEL_NAME)
            self.text_model.requires_grad_(False)

        def forward(self, text, device):
            if isinstance(text, tuple): text = list(text)
            inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=77)
            return self.text_model(**{k: v.to(device) for k, v in inputs.items()}).last_hidden_state

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device, self.prm, self.current_epoch = device, prm, 0
        self.image_size = prm.get('image_size', 64)
        image_channels = in_shape[1] if len(in_shape) > 1 else 3
        self.text_encoder = self.TextEncoder().to(device)

        unet = UNet(in_channels=image_channels, model_channels=prm.get('model_channels', 64), context_dim=512).to(device)
        self.unet = torch.compile(unet)
        self.ema = EMA(self.unet)

        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=prm.get('lr', 1e-4), weight_decay=1e-2)
        self.base_lr = prm.get('lr', 1e-4)
        self.warmup_epochs = 5
        total_epochs = prm.get('epochs', 100)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_epochs - self.warmup_epochs, eta_min=1e-6)

        self.loss_fn = nn.MSELoss()
        self.num_timesteps = int(prm.get('timesteps', 1000))

        betas = cosine_beta_schedule(self.num_timesteps).to(device)
        self.alphas_cumprod = torch.cumprod(1. - betas, axis=0)
        self.tensor_to_pil = ToPILImage()
        self.checkpoint_dir = "checkpoints"; os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.validated_this_epoch = False

        print("Pre-compiling and warming up the U-Net...")
        try:
            dummy_image = torch.randn(2, image_channels, self.image_size, self.image_size, device=device)
            dummy_time = torch.tensor([0, 1], device=device)
            dummy_text_context = torch.randn(2, 77, 512, device=device)
            self.unet(dummy_image, dummy_time, dummy_text_context)
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Warning: Model warm-up failed. Training will be slower. Error: {e}")

        list_of_files = glob.glob(os.path.join(self.checkpoint_dir, 'unet_epoch_*.pth'))
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
            print(f"Resuming from checkpoint: {latest_file}")
            self.unet.load_state_dict(torch.load(latest_file, map_location=device), strict=False)
            self.ema = EMA(self.unet)
            self.current_epoch = int(os.path.basename(latest_file).split('_')[-1].split('.')[0])

        self.scaler = torch.amp.GradScaler('cuda')
        self.null_text_context = self.text_encoder([""], self.device)

    def _extract(self, arr, t, x_shape):
        b, *_ = t.shape;
        out = arr.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1))).to(t.device)

    def train_setup(self, trial): pass

    def learn(self, train_loader):
        self.train();
        self.current_epoch += 1
        self.validated_this_epoch = False

        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
            for pg in self.optimizer.param_groups: pg['lr'] = lr
        else:
            self.scheduler.step()

        accumulation_steps = 4
        self.optimizer.zero_grad()
        cfg_drop_prob = 0.1

        for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {self.current_epoch}")):
            images, texts = batch
            images = images.to(self.device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                text_context = self.text_encoder(texts, self.device)
                mask = torch.rand(text_context.shape[0], device=self.device) < cfg_drop_prob
                if mask.any():
                    text_context[mask] = self.null_text_context

                t = torch.randint(0, self.num_timesteps, (images.shape[0],), device=self.device).long()
                noise = torch.randn_like(images)
                sqrt_alphas_cumprod_t = self._extract(torch.sqrt(self.alphas_cumprod), t, images.shape)
                sqrt_one_minus_alphas_cumprod_t = self._extract(torch.sqrt(1. - self.alphas_cumprod), t, images.shape)
                noisy_images = sqrt_alphas_cumprod_t * images + sqrt_one_minus_alphas_cumprod_t * noise
                predicted_noise = self.unet(noisy_images, t, text_context)
                loss = self.loss_fn(predicted_noise, noise) / accumulation_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.ema.update()

        torch.save(self.unet._orig_mod.state_dict(), os.path.join(self.checkpoint_dir, f"unet_epoch_{self.current_epoch}.pth"))
        return 0.0

    @torch.no_grad()
    def generate(self, text_prompts, num_inference_steps=None):
        self.ema.apply_shadow()
        self.eval()

        timesteps_to_iterate = num_inference_steps if num_inference_steps is not None else self.num_timesteps
        guidance_scale = self.prm.get('guidance_scale', 7.5)
        text_context = self.text_encoder(text_prompts, self.device)
        uncond_context = self.null_text_context.repeat(len(text_prompts), 1, 1)
        image = torch.randn((len(text_prompts), 3, self.image_size, self.image_size), device=self.device)

        for i in tqdm(reversed(range(timesteps_to_iterate)), desc="Sampling", total=timesteps_to_iterate, leave=False):
            t = torch.full((len(text_prompts),), i, device=self.device, dtype=torch.long)
            noise_pred_uncond = self.unet(image, t, uncond_context)
            noise_pred_cond = self.unet(image, t, text_context)
            predicted_noise = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            alpha_t = self._extract(self.alphas_cumprod, t, image.shape)
            alpha_t_prev_idx = torch.clamp(t - 1, min=0)
            alpha_t_prev = self._extract(self.alphas_cumprod, alpha_t_prev_idx, image.shape)

            predicted_original_sample = (image - torch.sqrt(1. - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            variance = (1. - alpha_t_prev) / (1. - alpha_t) * (1 - alpha_t / alpha_t_prev)
            std = torch.sqrt(variance.clamp(min=1e-20))
            direction_pointing_to_xt = torch.sqrt((1. - alpha_t_prev - variance).clamp(min=0.0)) * predicted_noise
            mean = torch.sqrt(alpha_t_prev) * predicted_original_sample + direction_pointing_to_xt
            image = mean + (std * torch.randn_like(image) if i > 0 else 0)

        image = (image.clamp(-1, 1) + 1) / 2
        self.ema.restore()
        return [self.tensor_to_pil(img.cpu()) for img in image]

    def forward(self, images, **kwargs):
        if not self.validated_this_epoch:
            fixed_prompts = [
                "a car",
                "a red car",
                "a blue car",
                "a car on a road"
            ]
            output_dir = "generated_images"; os.makedirs(output_dir, exist_ok=True)
            high_quality_sample = self.generate(text_prompts=[fixed_prompts[0]], num_inference_steps=250)
            if high_quality_sample:
                high_quality_sample[0].save(os.path.join(output_dir, f"epoch_{self.current_epoch}_sample.png"))

            batch_size = images.shape[0]
            prompts_for_metric = [random.choice(fixed_prompts) for _ in range(batch_size)]
            images_for_metric = self.generate(text_prompts=prompts_for_metric, num_inference_steps=50)

            try:
                onnx_path = os.path.join(self.checkpoint_dir, f"unet_epoch_{self.current_epoch}.onnx")
                print(f"Exporting model to ONNX: {onnx_path}")
                dummy_image = torch.randn(1, 3, self.image_size, self.image_size, device=self.device)
                dummy_time = torch.tensor([0], device=self.device, dtype=torch.long)
                dummy_text_context = torch.randn(1, 77, 512, device=self.device)

                self.ema.apply_shadow()
                self.unet._orig_mod.eval()

                torch.onnx.export(
                    self.unet._orig_mod,
                    (dummy_image, dummy_time, dummy_text_context),
                    onnx_path,
                    input_names=['noisy_image', 'time', 'text_context'],
                    output_names=['predicted_noise'],
                    dynamic_axes={
                        'noisy_image': {0: 'batch_size'},
                        'time': {0: 'batch_size'},
                        'text_context': {0: 'batch_size'},
                        'predicted_noise': {0: 'batch_size'}
                    },
                    opset_version=14
                )
                self.ema.restore()
                print("ONNX export successful.")
            except Exception as e:
                print(f"ONNX export failed: {e}")

            self.validated_this_epoch = True
            return (images_for_metric, prompts_for_metric)

        return ([], [])


def create_net(in_shape, out_shape, prm, device):
    return Net(in_shape, out_shape, prm, device)