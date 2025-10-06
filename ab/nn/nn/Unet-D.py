#Working model file
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
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')

try:
    from transformers import CLIPTextModel, CLIPTokenizer
except ImportError:
    raise ImportError("Please install 'transformers' for the Diffusion model: pip install transformers")

# ======================================================================================
# --- MODEL DEFINITION ---
# ======================================================================================

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def supported_hyperparameters():
    return {'lr': (float, 1e-4), 'timesteps': (int, 1000), 'model_channels': (int, 128), 'epochs': (int, 100),
            'onnx': (bool, False)}


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1;
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class EMA:
    def __init__(self, model, decay=0.999): self.model = model; self.decay = decay; self.shadow = deepcopy(
        self.model.state_dict())

    def update(self):
        model_params = self.model.state_dict()
        for name, param in model_params.items(): self.shadow[name].data = (
                self.shadow[name].data * self.decay + param.data * (1 - self.decay))

    def apply_shadow(self): self.original_params = deepcopy(self.model.state_dict()); self.model.load_state_dict(
        self.shadow, strict=True)

    def restore(self): self.model.load_state_dict(self.original_params, strict=True)


class SiLU(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__();
        self.group_norm = nn.GroupNorm(32, channels);
        self.query = nn.Linear(channels, channels);
        self.key = nn.Linear(channels, channels);
        self.value = nn.Linear(channels, channels);
        self.out = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, h, w = x.shape;
        x_norm = self.group_norm(x);
        x_r = x_norm.view(b, c, h * w).transpose(1, 2);
        q = self.query(x_r);
        k = self.key(x_r);
        v = self.value(x_r);
        sim = torch.bmm(q, k.transpose(1, 2)) * (c ** -0.5);
        attn = F.softmax(sim, dim=-1);
        out = torch.bmm(attn, v);
        out = self.out(out).transpose(1, 2).view(b, c, h, w);
        return x + out


class CrossAttention(nn.Module):
    def __init__(self, channels, context_dim):
        super().__init__();
        self.query = nn.Linear(channels, channels);
        self.key = nn.Linear(context_dim, channels);
        self.value = nn.Linear(context_dim, channels);
        self.out = nn.Linear(channels, channels)

    def forward(self, x, context):
        b, c, h, w = x.shape;
        x_r = x.view(b, c, h * w).transpose(1, 2)
        q = self.query(x_r);
        k = self.key(context);
        v = self.value(context)
        sim = torch.bmm(q, k.transpose(1, 2)) * (c ** -0.5);
        attn = F.softmax(sim, dim=-1)
        out = torch.bmm(attn, v);
        out = self.out(out).transpose(1, 2).view(b, c, h, w)
        return x + out


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, emb_dim):
        super().__init__();
        self.norm1 = nn.GroupNorm(32, in_c);
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1);
        self.emb_proj = nn.Linear(emb_dim, out_c);
        self.norm2 = nn.GroupNorm(32, out_c);
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1);
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity();
        self.act = SiLU()

    def forward(self, x, emb):
        h = self.act(self.norm1(x));
        h = self.conv1(h);
        h += self.emb_proj(self.act(emb)).unsqueeze(-1).unsqueeze(-1);
        h = self.act(self.norm2(h));
        h = self.conv2(h)
        return h + self.shortcut(x)


class ResBlockWithAttention(nn.Module):
    def __init__(self, in_c, out_c, emb_dim, ctx_dim=None, use_self_attn=False):
        super().__init__();
        self.res_block = ResBlock(in_c, out_c, emb_dim)
        if ctx_dim is not None:
            self.attn = CrossAttention(out_c, ctx_dim)
        elif use_self_attn:
            self.attn = SelfAttention(out_c)
        else:
            self.attn = nn.Identity()

    def forward(self, x, emb, ctx=None):
        x = self.res_block(x, emb)
        if isinstance(self.attn, CrossAttention):
            x = self.attn(x, ctx)
        else:
            x = self.attn(x)
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim

    def forward(self, time):
        device = time.device;
        half_dim = self.dim // 2;
        emb = math.log(10000) / (half_dim - 1);
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb);
        emb = time[:, None] * emb[None, :];
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class UNet(nn.Module):
    def __init__(self, in_c=3, mc=128, ctx_d=512):
        super().__init__()
        self.time_emb_dim = mc * 4
        self.time_embed = nn.Sequential(SinusoidalPositionEmbeddings(mc), nn.Linear(mc, self.time_emb_dim), SiLU(),
                                        nn.Linear(self.time_emb_dim, self.time_emb_dim))
        self.global_text_mlp = nn.Sequential(nn.Linear(ctx_d, self.time_emb_dim), SiLU(),
                                             nn.Linear(self.time_emb_dim, self.time_emb_dim))
        self.conv_in = nn.Conv2d(in_c, mc, 3, 1, 1)
        self.down1 = nn.ModuleList([ResBlockWithAttention(mc, mc * 2, self.time_emb_dim, ctx_d),
                                    ResBlockWithAttention(mc * 2, mc * 2, self.time_emb_dim, ctx_d)])
        self.down2 = nn.ModuleList([ResBlockWithAttention(mc * 2, mc * 4, self.time_emb_dim, ctx_d),
                                    ResBlockWithAttention(mc * 4, mc * 4, self.time_emb_dim, use_self_attn=True)])
        self.down3 = nn.ModuleList([ResBlockWithAttention(mc * 4, mc * 8, self.time_emb_dim, ctx_d),
                                    ResBlockWithAttention(mc * 8, mc * 8, self.time_emb_dim, use_self_attn=True)])
        self.pool = nn.AvgPool2d(2)
        self.mid1 = ResBlockWithAttention(mc * 8, mc * 8, self.time_emb_dim, ctx_d)
        self.mid2 = ResBlockWithAttention(mc * 8, mc * 8, self.time_emb_dim, use_self_attn=True)
        self.up1 = nn.ModuleList([ResBlockWithAttention(mc * 12, mc * 4, self.time_emb_dim, ctx_d),
                                  ResBlockWithAttention(mc * 4, mc * 4, self.time_emb_dim, use_self_attn=True)])
        self.up2 = nn.ModuleList([ResBlockWithAttention(mc * 6, mc * 2, self.time_emb_dim, ctx_d),
                                  ResBlockWithAttention(mc * 2, mc * 2, self.time_emb_dim, ctx_d)])
        self.up3 = nn.ModuleList([ResBlockWithAttention(mc * 3, mc, self.time_emb_dim, ctx_d),
                                  ResBlockWithAttention(mc, mc, self.time_emb_dim, ctx_d)])
        self.up_conv1 = nn.ConvTranspose2d(mc * 8, mc * 4, 2, 2)
        self.up_conv2 = nn.ConvTranspose2d(mc * 4, mc * 2, 2, 2)
        self.up_conv3 = nn.ConvTranspose2d(mc * 2, mc, 2, 2)
        self.out_conv = nn.Sequential(nn.GroupNorm(32, mc), SiLU(), nn.Conv2d(mc, in_c, 3, 1, 1))

    def forward(self, x, time, ctx, global_text_emb):
        time_emb = self.time_embed(time);
        text_emb = self.global_text_mlp(global_text_emb);
        emb = time_emb + text_emb
        h = self.conv_in(x)
        h1 = self.down1[0](h, emb, ctx);
        h1 = self.down1[1](h1, emb, ctx)
        h2 = self.down2[0](self.pool(h1), emb, ctx);
        h2 = self.down2[1](h2, emb)
        h3 = self.down3[0](self.pool(h2), emb, ctx);
        h3 = self.down3[1](h3, emb)
        h_mid = self.pool(h3);
        h_mid = self.mid1(h_mid, emb, ctx);
        h_mid = self.mid2(h_mid, emb)

        h = self.up_conv1(h_mid);
        h = F.interpolate(h, size=h3.shape[2:], mode='bilinear', align_corners=False)
        h = torch.cat([h, h3], dim=1);
        h = self.up1[0](h, emb, ctx);
        h = self.up1[1](h, emb)

        h = self.up_conv2(h);
        h = F.interpolate(h, size=h2.shape[2:], mode='bilinear', align_corners=False)
        h = torch.cat([h, h2], dim=1);
        h = self.up2[0](h, emb, ctx);
        h = self.up2[1](h, emb, ctx)

        h = self.up_conv3(h);
        h = F.interpolate(h, size=h1.shape[2:], mode='bilinear', align_corners=False)
        h = torch.cat([h, h1], dim=1);
        h = self.up3[0](h, emb, ctx);
        h = self.up3[1](h, emb, ctx)

        return self.out_conv(h)


class Net(nn.Module):
    class TextEncoder(nn.Module):
        def __init__(self):
            super().__init__();
            self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME, use_fast=True);
            self.text_model = CLIPTextModel.from_pretrained(CLIP_MODEL_NAME);
            self.text_model.requires_grad_(False)

        def forward(self, text, device):
            txt = list(text) if isinstance(text, tuple) else text;
            inp = self.tokenizer(txt, return_tensors="pt", padding='max_length', truncation=True, max_length=77);
            return self.text_model(**{k: v.to(device) for k, v in inp.items()}).last_hidden_state

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__();
        self.device, self.prm, self.current_epoch = device, prm, 0
        self.last_validated_epoch = -1  # <-- ADDED: Tracker for one-time validation
        self.image_size = prm.get('image_size', 128)
        image_channels = 3
        self.export_onnx = prm.get('onnx', False)
        self.text_encoder = self.TextEncoder().to(device);
        unet = UNet(in_c=image_channels, mc=prm.get('model_channels', 128), ctx_d=512).to(device);
        self.unet = torch.compile(unet);
        self.ema = EMA(self.unet);
        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=prm.get('lr', 1e-4), weight_decay=1e-2);
        self.base_lr = prm.get('lr', 1e-4);
        self.warmup_epochs = 5;
        total_epochs = prm.get('epochs', 100);
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    T_max=total_epochs - self.warmup_epochs,
                                                                    eta_min=1e-6);
        self.loss_fn = nn.L1Loss()

        timesteps_param = prm.get('timesteps', 1000)
        if isinstance(timesteps_param, float) and 0.0 <= timesteps_param <= 1.0:
            self.num_timesteps = int(100 + timesteps_param * 900)
        else:
            self.num_timesteps = int(timesteps_param)
        if self.num_timesteps <= 0: self.num_timesteps = 100

        betas = cosine_beta_schedule(self.num_timesteps).to(device);
        self.alphas_cumprod = torch.cumprod(1. - betas, axis=0);
        self.tensor_to_pil = ToPILImage();
        self.checkpoint_dir = "checkpoints";
        os.makedirs(self.checkpoint_dir, exist_ok=True);
        self.best_train_loss = float('inf')
        print("Pre-compiling and warming up the U-Net...");
        list_of_files = glob.glob(os.path.join(self.checkpoint_dir, 'latest_model.pth'))
        if not list_of_files:
            list_of_files = glob.glob(os.path.join(self.checkpoint_dir, 'best_model.pth'))

        if list_of_files:
            print(f"Resuming from checkpoint: {list_of_files[0]}");
            self.unet._orig_mod.load_state_dict(torch.load(list_of_files[0], map_location=device), strict=True);
            self.ema = EMA(self.unet)
        self.scaler = torch.amp.GradScaler('cuda');
        self.null_text_context = self.text_encoder([""], self.device)

    def _extract(self, arr, t, x_shape):
        b, *_ = t.shape;
        out = arr.gather(-1, t);
        return out.reshape(b, *((1,) * (len(x_shape) - 1))).to(t.device)

    def train_setup(self, trial):
        pass

    def learn(self, train_loader):
        self.train();
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs);
            for pg in self.optimizer.param_groups: pg['lr'] = lr
        else:
            self.scheduler.step()
        acc_steps = 16;
        self.optimizer.zero_grad();
        cfg_drop_prob = 0.1;
        epoch_losses = []
        for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {self.current_epoch}")):
            images, texts = batch;
            images = images.to(self.device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                text_ctx = self.text_encoder(texts, self.device);
                global_text_emb = text_ctx.mean(dim=1)
                mask = torch.rand(text_ctx.shape[0], device=self.device) < cfg_drop_prob
                if mask.any(): text_ctx[mask] = self.null_text_context; global_text_emb[mask] = 0
                t = torch.randint(0, self.num_timesteps, (images.shape[0],), device=self.device).long();
                noise = torch.randn_like(images);
                sqrt_a_t = self._extract(torch.sqrt(self.alphas_cumprod), t, images.shape);
                sqrt_1ma_t = self._extract(torch.sqrt(1. - self.alphas_cumprod), t, images.shape);
                noisy_imgs = sqrt_a_t * images + sqrt_1ma_t * noise
                pred_noise = self.unet(noisy_imgs, t, text_ctx, global_text_emb);
                loss = self.loss_fn(pred_noise, noise) / acc_steps
            epoch_losses.append(loss.item() * acc_steps);
            self.scaler.scale(loss).backward()
            if (i + 1) % acc_steps == 0 or i + 1 == len(train_loader):
                self.scaler.unscale_(self.optimizer);
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0);
                self.scaler.step(self.optimizer);
                self.scaler.update();
                self.optimizer.zero_grad();
                self.ema.update()

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')

        print(f"\nEpoch {self.current_epoch} finished. Average Loss: {avg_epoch_loss:.4f}")
        if avg_epoch_loss < self.best_train_loss:
            self.best_train_loss = avg_epoch_loss;
            print(f"New best training loss: {self.best_train_loss:.4f}. Saving best model checkpoint...");
            torch.save(self.unet._orig_mod.state_dict(), os.path.join(self.checkpoint_dir, "best_model.pth"))
            if self.export_onnx:
                print("Exporting best model to ONNX format...")
                try:
                    dummy_image = torch.randn(1, 3, self.image_size, self.image_size, device=self.device)
                    dummy_time = torch.tensor([500], device=self.device, dtype=torch.long)
                    dummy_ctx = torch.randn(1, 77, 512, device=self.device)
                    dummy_global_emb = torch.randn(1, 512, device=self.device)
                    onnx_path = os.path.join(self.checkpoint_dir, "best_model.onnx")

                    torch.onnx.export(
                        self.unet._orig_mod,
                        (dummy_image, dummy_time, dummy_ctx, dummy_global_emb),
                        onnx_path,
                        input_names=['image', 'time', 'context', 'global_text_emb'],
                        output_names=['noise_pred'],
                        dynamic_axes={
                            'image': {0: 'batch_size'}, 'context': {0: 'batch_size'},
                            'global_text_emb': {0: 'batch_size'}, 'noise_pred': {0: 'batch_size'}
                        },
                        opset_version=14
                    )
                    print(f"Model successfully exported to {onnx_path}")
                except Exception as e:
                    print(f"Failed to export model to ONNX: {e}")
        else:
            print(f"Training loss did not improve from {self.best_train_loss:.4f}.")

        print("\nSaving latest model checkpoint (end of epoch)...")
        try:
            latest_pth_path = os.path.join(self.checkpoint_dir, "latest_model.pth")
            torch.save(self.unet._orig_mod.state_dict(), latest_pth_path)
            print(f"Latest PyTorch checkpoint saved to {latest_pth_path}")
        except Exception as e:
            print(f"An error occurred during final checkpointing: {e}")

        return 0.0

    @torch.no_grad()
    def generate(self, text_prompts, num_inference_steps=None):
        self.ema.apply_shadow();
        self.eval();
        timesteps_to_iterate = num_inference_steps if num_inference_steps is not None else self.num_timesteps;
        guidance_scale = 7.5;
        text_ctx = self.text_encoder(text_prompts, self.device);
        uncond_ctx = self.null_text_context.repeat(len(text_prompts), 1, 1);
        global_text_emb = text_ctx.mean(dim=1);
        global_uncond_emb = uncond_ctx.mean(dim=1);
        image = torch.randn((len(text_prompts), 3, self.image_size, self.image_size), device=self.device)
        for i in tqdm(reversed(range(timesteps_to_iterate)), desc="Sampling", total=timesteps_to_iterate, leave=False):
            t = torch.full((len(text_prompts),), i, device=self.device, dtype=torch.long);
            noise_pred_uncond = self.unet(image, t, uncond_ctx, global_uncond_emb);
            noise_pred_cond = self.unet(image, t, text_ctx, global_text_emb);
            pred_noise = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond);
            alpha_t = self._extract(self.alphas_cumprod, t, image.shape);
            t_prev_idx = torch.clamp(t - 1, min=0);
            alpha_t_prev = self._extract(self.alphas_cumprod, t_prev_idx, image.shape);
            pred_orig_sample = (image - torch.sqrt(1. - alpha_t) * pred_noise) / torch.sqrt(alpha_t);
            variance = (1. - alpha_t_prev) / (1. - alpha_t) * (1 - alpha_t / alpha_t_prev);
            std = torch.sqrt(variance.clamp(min=1e-20));
            dir_xt = torch.sqrt((1. - alpha_t_prev - variance).clamp(min=0.0)) * pred_noise;
            mean = torch.sqrt(alpha_t_prev) * pred_orig_sample + dir_xt;
            image = mean + (std * torch.randn_like(image) if i > 0 else 0)
        image = (image.clamp(-1, 1) + 1) / 2;
        self.ema.restore();
        return [self.tensor_to_pil(img.cpu()) for img in image]

    def forward(self, images, **kwargs):
        # --- FIX: Ensure validation generation only runs ONCE per epoch ---
        if self.current_epoch > self.last_validated_epoch:
            self.eval()
            self.last_validated_epoch = self.current_epoch

            print(f"\nRunning validation for Epoch {self.current_epoch}...")

            eval_prompts = [
                "a red car on a sunny day",
                "a photograph of a blue sports car",
                "a vintage black car",
                "a white car driving on a highway",
                "a silver car parked in a garage",
                "a green car in a forest",
                "an orange car at sunset",
                "a yellow taxi cab"
            ]

            batch_size = images.shape[0]
            prompts_to_use = (eval_prompts * (batch_size // len(eval_prompts) + 1))[:batch_size]

            if not prompts_to_use:
                return ([], [])

            generated_images = self.generate(text_prompts=prompts_to_use, num_inference_steps=50)

            return (generated_images, prompts_to_use)

        return ([], [])


def create_net(in_shape, out_shape, prm, device):
    return Net(in_shape, out_shape, prm, device)


# ======================================================================================
# --- STAND-ALONE GENERATION BLOCK ---
#    python ab/nn/nn/Unet-D.py --prompt "text prompt here"
#Ex: python ab/nn/nn/Unet-D.py --prompt "a red car"
# ======================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt using a trained U-Net model.")
    parser.add_argument("--prompt", type=str, default="a red car", help="The text prompt for image generation.")
    parser.add_argument("--steps", type=int, default=250, help="Number of denoising steps for generation.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest_model.pth",
                        help="Path to the model checkpoint.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # we have to get the same mode channels as we used for training, when we want to generate
    prm = {'model_channels': 96}
    model = Net(in_shape=None, out_shape=None, prm=prm, device=device).to(device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}. Please train the model first.")

    print(f"Loading checkpoint from {args.checkpoint}...")
    model.unet._orig_mod.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print("Checkpoint loaded successfully.")

    print(f"Generating image for prompt: '{args.prompt}'")
    images = model.generate(text_prompts=[args.prompt], num_inference_steps=args.steps)

    if images:
        safe_prompt = "".join([c if c.isalnum() else "_" for c in args.prompt]).strip("_")
        output_path = f"{safe_prompt}.png"
        images[0].save(output_path)
        print(f"Image saved to {output_path}")