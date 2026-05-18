import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Blip2Model, GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def supported_hyperparameters():
    return {'lr', 'batch', 'dropout', 'num_layers', 'num_heads', 'ff_dim', 'decoder_dim', 'load_in_4bit', 'freeze_gpt2'}


def safe_prm(prm, key, default=None):
    if prm is None:
        prm = {}
    if default is None:
        lk = key.lower()
        if any(x in lk for x in ['dim', 'size', 'hidden']):
            default = 768
        elif any(x in lk for x in ['head', 'layer', 'num']):
            default = 8
        elif 'dropout' in lk:
            default = 0.1
        elif 'lr' in lk:
            default = 1e-4
        else:
            default = 0
    val = prm.get(key, default) if isinstance(prm, dict) else default
    if val is None:
        return default
    try:
        if isinstance(default, bool):
            return bool(val)
        if isinstance(default, int):
            return int(val)
        if isinstance(default, float):
            return float(val)
    except Exception:
        pass
    return val


class FrozenBlip2Encoder(nn.Module):
    def __init__(self, device, load_in_4bit=True):
        super().__init__()
        self.device = device
        self.hidden_size = 768
        self.blip2 = None
        model_id = 'Salesforce/blip2-opt-2.7b'

        # Lazy-compatible design:
        # If cached_blip2 transform is used, input is already (B,32,768),
        # so this class will not need to run BLIP2 forward.
        try:
            self.blip2 = Blip2Model.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                load_in_4bit=bool(load_in_4bit),
                device_map={'': device},
            )
        except TypeError:
            self.blip2 = Blip2Model.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map={'': device},
            )
        except Exception as e:
            # Cached feature mode can still work without loading BLIP2 here.
            print(f'[FrozenBlip2Encoder WARN] BLIP2 load skipped/failed: {e}')
            self.blip2 = None

        if self.blip2 is not None:
            for param in self.blip2.parameters():
                param.requires_grad = False
            self.blip2.eval()
            try:
                self.hidden_size = self.blip2.config.qformer_config.hidden_size
            except Exception:
                self.hidden_size = 768

        self.eval()

    def train(self, mode=True):
        super().train(False)
        if self.blip2 is not None:
            self.blip2.eval()
        return self

    def forward(self, pixel_values):
        # Fast cached feature path: cached_blip2 gives (B,32,768).
        if torch.is_tensor(pixel_values):
            if pixel_values.dim() == 3 and pixel_values.size(-1) == self.hidden_size:
                return pixel_values.to(self.device).float()
            if pixel_values.dim() == 2 and pixel_values.size(-1) == self.hidden_size:
                return pixel_values.to(self.device).float().unsqueeze(1)

        if self.blip2 is None:
            raise RuntimeError('BLIP2 encoder is not loaded and input is not cached features (B,T,768). Use --transform cached_blip2 or fix BLIP2 loading.')

        self.blip2.eval()
        with torch.no_grad():
            outputs = self.blip2.get_qformer_features(pixel_values=pixel_values.to(self.device))
        if torch.is_tensor(outputs):
            return outputs.float()
        return outputs.last_hidden_state.float()


# === LLM-GENERATED BRIDGE ===
class CrossModalBridge(nn.Module):
    def __init__(self, prm):
        super().__init__()
        d = int(safe_prm(prm, 'ff_dim', 512))
        self.norm = nn.LayerNorm(768)
        self.fc = nn.Linear(768, d)
        self.gate = nn.Linear(768, d)
        self.out = nn.Linear(d, 768)
    def forward(self, x, captions=None):
        r = self.norm(x)
        return self.out(self.fc(r) * torch.sigmoid(self.gate(r)))

# === END LLM-GENERATED BRIDGE ===


class CaptionDecoder(nn.Module):
    def __init__(self, q_former_hidden, device, prm):
        super().__init__()
        self.device = device
        self.prm = prm if isinstance(prm, dict) else {}
        gpt2_id = 'gpt2'

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        config = GPT2Config.from_pretrained(gpt2_id)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_id, config=config).to(device)
        self.gpt2_hidden = int(config.n_embd)

        freeze_gpt2 = bool(self.prm.get('freeze_gpt2', False))
        if freeze_gpt2:
            for p in self.gpt2.parameters():
                p.requires_grad = False

        self.bridge = CrossModalBridge(self.prm).to(device)

    def _normalize_visual_embeds(self, visual_embeds, batch_size):
        if not torch.is_tensor(visual_embeds):
            raise TypeError('CrossModalBridge must return a torch.Tensor')

        visual_embeds = visual_embeds.to(self.device).float()

        if visual_embeds.dim() == 4:
            visual_embeds = visual_embeds.reshape(visual_embeds.size(0), -1, visual_embeds.size(-1))
        elif visual_embeds.dim() == 2:
            visual_embeds = visual_embeds.unsqueeze(1)
        elif visual_embeds.dim() > 4:
            visual_embeds = visual_embeds.reshape(visual_embeds.size(0), -1, visual_embeds.size(-1))

        if visual_embeds.dim() != 3:
            visual_embeds = visual_embeds.reshape(batch_size, -1, visual_embeds.size(-1))

        # Fix accidental (T,B,H) output.
        if visual_embeds.size(0) != batch_size:
            if visual_embeds.size(1) == batch_size:
                visual_embeds = visual_embeds.transpose(0, 1).contiguous()
            else:
                visual_embeds = visual_embeds.reshape(batch_size, -1, visual_embeds.size(-1))

        # Keep prefix length under control. cached_blip2 normally uses 32 tokens.
        if visual_embeds.size(1) > 32:
            visual_embeds = visual_embeds[:, :32, :]

        hidden = visual_embeds.size(-1)
        if hidden > self.gpt2_hidden:
            visual_embeds = visual_embeds[..., :self.gpt2_hidden]
        elif hidden < self.gpt2_hidden:
            visual_embeds = F.pad(visual_embeds, (0, self.gpt2_hidden - hidden))

        return visual_embeds

    def forward(self, visual_features, caption_ids=None):
        batch_size = visual_features.size(0)
        visual_features = visual_features.to(self.device).float()
        visual_embeds = self.bridge(visual_features, caption_ids)
        visual_embeds = self._normalize_visual_embeds(visual_embeds, batch_size)

        if caption_ids is not None:
            if caption_ids.dim() == 3:
                caption_ids = caption_ids[:, 0, :]
            caption_ids = caption_ids.long().to(self.device)
            caption_ids = caption_ids.clamp(min=0, max=self.gpt2.config.vocab_size - 1)

            start_tokens = torch.full(
                (batch_size, 1),
                self.tokenizer.eos_token_id,
                dtype=torch.long,
                device=self.device,
            )
            caption_ids = torch.cat([start_tokens, caption_ids], dim=1)

            text_embeds = self.gpt2.transformer.wte(caption_ids)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

            ignore_labels = torch.full(
                (batch_size, visual_embeds.shape[1] + 1),
                -100,
                dtype=torch.long,
                device=self.device,
            )
            labels = torch.cat([ignore_labels, caption_ids[:, 1:]], dim=1)

            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            visual_mask = torch.ones((batch_size, visual_embeds.shape[1]), dtype=torch.long, device=self.device)
            text_mask = (caption_ids != pad_id).long().to(self.device)
            attention_mask = torch.cat([visual_mask, text_mask], dim=1)
            return self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss

        # Inference: SOTA-safe greedy generation with per-sequence EOS mask.
        start_token = torch.full(
            (batch_size, 1),
            self.tokenizer.eos_token_id,
            dtype=torch.long,
            device=self.device,
        )
        start_embed = self.gpt2.transformer.wte(start_token)
        outputs_embeds = torch.cat([visual_embeds, start_embed], dim=1)

        generated = []
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(40):
            out = self.gpt2(
                inputs_embeds=outputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Once a sample is finished, keep forcing EOS for that sample.
            next_token[finished] = self.tokenizer.eos_token_id
            generated.append(next_token)

            finished |= (next_token.squeeze(-1) == self.tokenizer.eos_token_id)
            if finished.all():
                break

            past_key_values = out.past_key_values
            outputs_embeds = self.gpt2.transformer.wte(next_token)

        if not generated:
            return torch.empty((batch_size, 0), dtype=torch.long, device=self.device)
        return torch.cat(generated, dim=1)


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.prm = prm if isinstance(prm, dict) else {}
        self.vocab_size = int(out_shape[0]) if out_shape and len(out_shape) > 0 else 50257
        print(f'!!! [MODEL SETUP] Hyperparameters: {self.prm}')

        load_in_4bit = bool(self.prm.get('load_in_4bit', True))
        self.encoder = FrozenBlip2Encoder(device, load_in_4bit=load_in_4bit)
        self.decoder = CaptionDecoder(self.encoder.hidden_size, device, self.prm)

        # Generic Train.py sometimes expects criterion. Captioning uses decoder loss.
        self.criterion = lambda o, l: torch.tensor(0.0, device=self.device, requires_grad=True)
        self.idx2word = None
        self.optimizer = None
        self._print_param_stats()

    def _print_param_stats(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = (100.0 * trainable / total) if total else 0.0
        print(f'   Total params: {total:,} | Trainable: {trainable:,} ({pct:.1f}%)')

    def _ensure_vocab(self):
        if self.idx2word is not None:
            return
        try:
            from ab.nn.loader.coco_.Caption import GLOBAL_CAPTION_VOCAB
            self.idx2word = GLOBAL_CAPTION_VOCAB.get('idx2word', {})
        except Exception:
            self.idx2word = {}

    def forward(self, pixel_values, captions=None):
        self.encoder.eval()
        visual_features = self.encoder(pixel_values)
        if captions is not None:
            self._ensure_vocab()
            if captions.dim() == 3:
                captions = captions[:, 0, :]
            return self.decoder(visual_features, captions)
        return self.decoder(visual_features, None)

    def train_setup(self, prm):
        if isinstance(prm, dict):
            self.prm.update(prm)
        trainable_params = [p for p in self.decoder.parameters() if p.requires_grad]
        lr = float(self.prm.get('lr', 1e-4))
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    def learn(self, train_data):
        self.encoder.eval()
        self.decoder.train()
        self._ensure_vocab()
        if self.optimizer is None:
            self.train_setup(self.prm)

        total_loss = 0.0
        n = 0

        for images, captions in train_data:
            if isinstance(images, list):
                images = torch.stack(images)
            if isinstance(captions, list):
                captions = torch.stack(captions)

            images = images.to(self.device)
            captions = captions.to(self.device)
            if captions.dim() == 3:
                captions = captions[:, 0, :]

            self.optimizer.zero_grad(set_to_none=True)
            loss = self.forward(images, captions)
            if not torch.is_tensor(loss):
                loss = torch.tensor(float(loss), device=self.device, requires_grad=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
            self.optimizer.step()

            total_loss += float(loss.detach().item())
            n += 1

        return 0.0, total_loss / max(n, 1)

# Trial ID: e4022efa1b10222c8cb0d52a882ae2b3
