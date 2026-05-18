import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_prm(prm, key, default=None):
    if prm is None:
        return default
    val = prm.get(key, default) if isinstance(prm, dict) else default
    return val if val is not None else (default if default is not None else 256)


def supported_hyperparameters():
    return {"lr", "batch", "dropout", "decoder_dim", "num_heads", "num_layers", "ff_dim", "max_length"}


# === LLM-GENERATED BRIDGE ===
class CrossModalBridge(nn.Module):
    def __init__(self, prm):
        super().__init__()
        nl = int(safe_prm(prm, 'num_layers', 2))
        layers = []
        for _ in range(nl):
            layers += [nn.LayerNorm(768), nn.Linear(768, 768), nn.GELU()]
        self.mixer = nn.Sequential(*layers)
    def forward(self, x, captions=None):
        return self.mixer(x)

# === END LLM CODE ===


class FrozenBlip2Encoder(nn.Module):
    def __init__(self, device, load_in_4bit=True):
        super().__init__()
        self.device = device
        self.load_in_4bit = load_in_4bit
        self._blip2 = None
        self.hidden_size = 768

    @property
    def blip2(self):
        if self._blip2 is None:
            from transformers import Blip2Model
            kw = dict(torch_dtype=torch.float16, device_map={"": self.device})
            if self.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    kw["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                    )
                except Exception:
                    pass
            self._blip2 = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", **kw)
            for p in self._blip2.parameters():
                p.requires_grad = False
        return self._blip2

    def forward(self, pixel_values):
        if pixel_values.dim() == 3 and pixel_values.size(-1) == self.hidden_size:
            return pixel_values
        if pixel_values.dim() == 2 and pixel_values.size(-1) == self.hidden_size:
            return pixel_values.unsqueeze(1)
        pixel_values = pixel_values.to(self.device)
        if pixel_values.dtype != torch.float16:
            pixel_values = pixel_values.half()
        self.blip2.eval()
        with torch.no_grad():
            out = self.blip2.get_qformer_features(pixel_values=pixel_values)
        return out.last_hidden_state.float()


class CaptionDecoder(nn.Module):
    def __init__(self, encoder_hidden_size, device, prm):
        super().__init__()
        self.device = device
        self.prm = prm if isinstance(prm, dict) else {}
        self.bridge = CrossModalBridge(self.prm)
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.gpt2_hidden = self.gpt2.config.hidden_size

    def _normalize_visual_embeds(self, visual_embeds, batch_size):
        visual_embeds = visual_embeds.to(self.device).float()
        if visual_embeds.dim() == 4:
            visual_embeds = visual_embeds.reshape(visual_embeds.size(0), -1, visual_embeds.size(-1))
        elif visual_embeds.dim() == 2:
            visual_embeds = visual_embeds.unsqueeze(1)
        elif visual_embeds.dim() > 4:
            visual_embeds = visual_embeds.reshape(visual_embeds.size(0), -1, visual_embeds.size(-1))
        if visual_embeds.dim() != 3:
            visual_embeds = visual_embeds.reshape(batch_size, -1, visual_embeds.size(-1))
        if visual_embeds.size(0) != batch_size:
            if visual_embeds.size(1) == batch_size:
                visual_embeds = visual_embeds.transpose(0, 1).contiguous()
            else:
                visual_embeds = visual_embeds.reshape(batch_size, -1, visual_embeds.size(-1))
        hidden = visual_embeds.size(-1)
        if hidden > self.gpt2_hidden:
            visual_embeds = visual_embeds[..., :self.gpt2_hidden]
        elif hidden < self.gpt2_hidden:
            visual_embeds = F.pad(visual_embeds, (0, self.gpt2_hidden - hidden))
        return visual_embeds

    def forward(self, visual_features, caption_ids=None):
        batch_size = visual_features.size(0)
        visual_embeds = self.bridge(visual_features, caption_ids)
        visual_embeds = self._normalize_visual_embeds(visual_embeds, batch_size)
        if caption_ids is not None:
            if caption_ids.dim() == 3:
                caption_ids = caption_ids[:, 0, :]
            caption_ids = caption_ids.long().to(self.device)
            caption_ids = caption_ids.clamp(min=0, max=self.gpt2.config.vocab_size - 1)
            text_embeds = self.gpt2.transformer.wte(caption_ids)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            visual_mask = torch.ones((batch_size, visual_embeds.shape[1]), dtype=torch.long, device=self.device)
            text_mask = (caption_ids != pad_id).long().to(self.device)
            attention_mask = torch.cat([visual_mask, text_mask], dim=1)
            ignore_labels = torch.full((batch_size, visual_embeds.shape[1]), -100, dtype=torch.long, device=self.device)
            labels_text = caption_ids.clone()
            labels_text[labels_text == pad_id] = -100
            labels = torch.cat([ignore_labels, labels_text], dim=1)
            return self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss
        # Generation mode
        start_token = torch.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=torch.long, device=self.device)
        start_embed = self.gpt2.transformer.wte(start_token)
        outputs_embeds = torch.cat([visual_embeds, start_embed], dim=1)
        
        generated = []
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        max_new_tokens = int(self.prm.get("max_length", 30)) if isinstance(self.prm, dict) else 30
        max_new_tokens = max(8, min(max_new_tokens, 40))
        repetition_penalty = 1.15

        for step in range(max_new_tokens):
            out = self.gpt2(inputs_embeds=outputs_embeds, past_key_values=past_key_values, use_cache=True)
            logits = out.logits[:, -1, :]
            
            if step < 5:
                logits[:, self.tokenizer.eos_token_id] = -float("inf")

            if generated:
                prev_tokens = torch.cat(generated, dim=1)
                for b in range(batch_size):
                    for tok in set(prev_tokens[b].tolist()):
                        if 0 <= tok < logits.size(-1):
                            if logits[b, tok] > 0:
                                logits[b, tok] /= repetition_penalty
                            else:
                                logits[b, tok] *= repetition_penalty

            next_token = logits.argmax(dim=-1, keepdim=True)
            next_token[finished] = self.tokenizer.eos_token_id
            generated.append(next_token)
            finished |= next_token.squeeze(-1).eq(self.tokenizer.eos_token_id)
            
            if finished.all():
                break

            past_key_values = out.past_key_values
            outputs_embeds = self.gpt2.transformer.wte(next_token)

        if generated:
            return torch.cat(generated, dim=1)
        return torch.empty((batch_size, 0), dtype=torch.long, device=self.device)


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.prm = prm if isinstance(prm, dict) else {}
        load_in_4bit = bool(self.prm.get("load_in_4bit", True))
        self.encoder = FrozenBlip2Encoder(device, load_in_4bit=load_in_4bit)
        self.decoder = CaptionDecoder(self.encoder.hidden_size, device, self.prm)
        self.criterion = lambda o, l: torch.tensor(0.0, device=self.device, requires_grad=True)
        self.optimizer = None

    def forward(self, pixel_values, captions=None):
        self.encoder.eval()
        visual_features = self.encoder(pixel_values)
        if captions is not None:
            if captions.dim() == 3:
                captions = captions[:, 0, :]
            return self.decoder(visual_features, captions)
        return self.decoder(visual_features, None)

    def train_setup(self, prm):
        if isinstance(prm, dict):
            self.prm.update(prm)
        trainable_params = [p for p in self.decoder.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=float(self.prm.get("lr", 1e-4)))

    def learn(self, train_data):
        self.encoder.eval()
        self.decoder.train()
        if self.optimizer is None:
            self.train_setup(self.prm)
        total_loss, n = 0.0, 0
        for images, captions in train_data:
            if isinstance(images, list): images = torch.stack(images)
            if isinstance(captions, list): captions = torch.stack(captions)
            images, captions = images.to(self.device), captions.to(self.device)
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


# Trial ID: 6367c490244b39502be2b45717973bb3
