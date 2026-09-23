"""Blip2Fast - Cached BLIP-2 visual features with a trainable GPT-2 decoder.

This is the production GPT-2-based image captioning model. BLIP-2 vision/Q-Former
features (32, 768) remain precomputed in validated cache shards; the trainable
part is a normalized visual bridge plus GPT-2.
"""

from __future__ import annotations

import torch

from ab.nn.util.captioning.blip2.text import caption_training_batch
import torch.nn as nn

from ab.nn.util.captioning.blip2.contract import FEATURE_SHAPE
from ab.nn.util.captioning.blip2.gpt2 import GPT2_VOCAB_SIZE, gpt2_runtime_paths

from transformers import AutoModelForCausalLM, AutoTokenizer


def supported_hyperparameters():
    # The NN trainer searches batch size through its dedicated categorical
    # range. Declaring it here would make the generic fallback suggest a
    # float in [0, 1] before that batch-size logic runs.
    return {"lr"}


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = torch.device(device)
        self.prm = dict(prm or {})
        seed = int(self.prm.get("seed", 42))
        self._reference_generator = torch.Generator(device="cpu")
        self._reference_generator.manual_seed(seed)

        if in_shape and tuple(in_shape[-2:]) != FEATURE_SHAPE:
            raise ValueError(f"Expected cached input ending in {FEATURE_SHAPE}, got {in_shape}.")
        if out_shape and int(out_shape[0]) != GPT2_VOCAB_SIZE:
            raise ValueError(f"This model requires GPT-2 token IDs ({GPT2_VOCAB_SIZE}).")
        if self.device.type != "cuda" and not bool(self.prm.get("allow_cpu", False)):
            raise RuntimeError("Blip2Fast requires CUDA unless allow_cpu=true.")
        if self.device.type == "cuda":
            total_gib = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            if total_gib < 12:
                raise RuntimeError(
                    f"Trainable GPT-2 requires at least 12 GiB VRAM; found {total_gib:.1f} GiB."
                )

        decoder_path, tokenizer_path = gpt2_runtime_paths(self.prm.get("cache_dir"))
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), use_fast=True, local_files_only=True
        )
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        # Keep trainable GPT-2 in float32.
        self.gpt2 = AutoModelForCausalLM.from_pretrained(
            str(decoder_path), low_cpu_mem_usage=True, local_files_only=True
        ).to(self.device, dtype=torch.float32)
        if int(self.gpt2.config.vocab_size) != GPT2_VOCAB_SIZE:
            raise RuntimeError("Portable GPT-2 decoder has an incompatible vocabulary.")
        hidden = int(self.gpt2.get_input_embeddings().embedding_dim)
        # Initialize the trainable bridge reproducibly without changing the
        # process-wide RNG used by models trained before or after this one.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.visual_projection = nn.Sequential(
                nn.Linear(FEATURE_SHAPE[1], hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            ).to(self.device, dtype=torch.float32)
        self.optimizer = None
        self.max_text_length = int(self.prm.get("max_text_length", 50))
        self.max_new_tokens = int(self.prm.get("max_new_tokens", 24))
        self.num_beams = int(self.prm.get("num_beams", 3))
        self.prompt = str(self.prm.get("caption_prompt", "a photo of "))

    def _features(self, value):
        if value.ndim != 3 or tuple(value.shape[1:]) != FEATURE_SHAPE:
            raise ValueError(f"Expected (B,{FEATURE_SHAPE[0]},{FEATURE_SHAPE[1]}) features.")
        if not torch.isfinite(value).all():
            raise ValueError("Cached features contain NaN or Inf.")
        return value.to(self.device, dtype=torch.float32, non_blocking=True)

    def _select_reference_labels(self, labels):
        if labels.ndim != 3:
            return labels
        valid = (labels >= 0).any(dim=2)
        if not valid.any(dim=1).all():
            raise ValueError("At least one sample has no caption reference.")
        if self.training:
            chosen = []
            for row in valid.detach().cpu():
                choices = torch.nonzero(row, as_tuple=False).flatten()
                position = torch.randint(
                    len(choices), (1,), generator=self._reference_generator
                ).item()
                chosen.append(int(choices[position]))
            indices = torch.tensor(chosen, device=labels.device)
        else:
            indices = valid.long().argmax(dim=1)
        return labels[torch.arange(labels.shape[0], device=labels.device), indices]

    @staticmethod
    def _caption_targets(ids, attention_mask, prompt_length):
        targets = ids.clone()
        targets[attention_mask == 0] = -100
        targets[:, : min(int(prompt_length), targets.shape[1])] = -100
        return targets

    def _reference_text(self, labels):
        selected = self._select_reference_labels(labels)
        clean = selected.detach().cpu().clone()
        clean[clean < 0] = self.gpt2_tokenizer.eos_token_id
        return self.gpt2_tokenizer.batch_decode(clean, skip_special_tokens=True)

    def _loss(self, features, labels):
        ids, mask, targets = caption_training_batch(
            self.gpt2_tokenizer, self.prompt, self._reference_text(labels),
            self.max_text_length, self.device, add_special_tokens=False,
        )
        visual = self.visual_projection(features)
        text = self.gpt2.get_input_embeddings()(ids)
        embeddings = torch.cat((visual, text), dim=1)
        prefix_labels = torch.full(visual.shape[:2], -100, dtype=torch.long, device=self.device)
        prefix_mask = torch.ones(visual.shape[:2], dtype=mask.dtype, device=self.device)
        return self.gpt2(
            inputs_embeds=embeddings,
            attention_mask=torch.cat((prefix_mask, mask), dim=1),
            labels=torch.cat((prefix_labels, targets), dim=1),
            use_cache=False,
        ).loss

    def _generate(self, features):
        visual = self.visual_projection(features)
        prompt = self.gpt2_tokenizer(
            self.prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device).expand(features.shape[0], -1)
        text = self.gpt2.get_input_embeddings()(prompt)
        embeddings = torch.cat((visual, text), dim=1)
        mask = torch.ones(embeddings.shape[:2], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            return self.gpt2.generate(
                inputs_embeds=embeddings,
                attention_mask=mask,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                eos_token_id=self.gpt2_tokenizer.eos_token_id,
                pad_token_id=self.gpt2_tokenizer.eos_token_id,
            )

    def forward(self, features, captions=None):
        features = self._features(features)
        return self._loss(features, captions) if captions is not None else self._generate(features)

    def compute_loss(self, features, labels):
        """Return the scalar teacher-forcing loss expected by the NN trainer."""
        return self(features, labels)

    def train_setup(self, prm):
        self.prm.update(prm or {})
        learning_rate = float(self.prm.get("lr", 1e-4))
        if not 0 < learning_rate <= 1:
            raise ValueError("lr must be in (0, 1].")
        parameters = list(self.visual_projection.parameters()) + list(self.gpt2.parameters())
        self.optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)

    def learn(self, train_data):
        if self.optimizer is None:
            raise RuntimeError("train_setup() must be called before learn().")
        self.train(True)
        total, batches = 0.0, 0
        for features, labels in train_data:
            self.optimizer.zero_grad(set_to_none=True)
            loss = self(features, labels.to(self.device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite BLIP-2/GPT-2 loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.visual_projection.parameters()) + list(self.gpt2.parameters()), 1.0
            )
            self.optimizer.step()
            total += float(loss.detach())
            batches += 1
        return 0.0, total / max(batches, 1)


__all__ = ["Net", "supported_hyperparameters"]
