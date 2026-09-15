"""Crash-resistant BLIP-2 captioner trained from cached Q-Former features.

The cache builder owns the expensive vision/Q-Former model.  This module loads
only the frozen OPT decoder and the small pretrained BLIP-2 language projection.
"""

from __future__ import annotations

import os
import random

import torch
import torch.nn as nn

from ab.nn.captioning.blip2.environment import validate_environment

validate_environment()

from transformers import AutoModelForCausalLM, AutoTokenizer

from ab.nn.captioning.blip2.contract import (
    FEATURE_SHAPE,
    OPT_DIR_NAME,
    OPT_TOKENIZER_DIR_NAME,
    OPT_VOCAB_SIZE,
    PROJECTION_NAME,
    read_manifest,
    resolve_cache_dir,
    resolve_projection_path,
    sha256_file,
    validate_runtime,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def supported_hyperparameters():
    # Only numeric/searchable training parameters belong here. Portable
    # projection selection is configured explicitly through prm/environment;
    # listing string paths here makes Optuna invent invalid float values.
    return {"lr", "batch"}


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = torch.device(device)
        self.prm = dict(prm or {})
        self.optimizer = None
        seed = int(self.prm.get("seed", 42))
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if in_shape and tuple(in_shape[-2:]) != FEATURE_SHAPE:
            raise ValueError(f"Expected cached input ending in {FEATURE_SHAPE}, got {in_shape}.")
        if out_shape and int(out_shape[0]) != OPT_VOCAB_SIZE:
            raise ValueError(
                f"Blip2Cached requires OPT token IDs ({OPT_VOCAB_SIZE}) from its matching transform."
            )
        if self.device.type != "cuda" and not bool(self.prm.get("allow_cpu", False)):
            raise RuntimeError(
                "Blip2Cached refused to load OPT-2.7B on CPU because it can exhaust "
                "system RAM. Use a CUDA GPU or explicitly set allow_cpu=true."
            )
        if self.device.type == "cuda":
            total_gib = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            if total_gib < 8:
                raise RuntimeError(
                    f"Blip2Cached requires at least 8 GiB VRAM; detected {total_gib:.1f} GiB."
                )

        cache_dir = resolve_cache_dir(self.prm.get("cache_dir"))
        manifest = read_manifest(cache_dir)
        runtime_dir = validate_runtime(cache_dir, manifest)
        projection_path = resolve_projection_path(
            cache_dir, self.prm.get("projection_path")
        )
        projection_record = manifest.get("projection", {})
        if not projection_path.is_file():
            raise RuntimeError(f"BLIP-2 projection is missing: {projection_path}")
        default_projection = (cache_dir / PROJECTION_NAME).resolve()
        expected_sha256 = self.prm.get("projection_sha256") or os.environ.get(
            "BLIP2_PROJECTION_SHA256"
        )
        if projection_path == default_projection:
            expected_sha256 = projection_record.get("sha256")
        if expected_sha256 and sha256_file(projection_path) != expected_sha256:
            raise RuntimeError("BLIP-2 projection checksum mismatch.")
        self.projection_path = projection_path

        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.opt_tokenizer = AutoTokenizer.from_pretrained(
            str(runtime_dir / OPT_TOKENIZER_DIR_NAME),
            use_fast=False,
            local_files_only=True,
        )
        self.opt_tokenizer.pad_token = self.opt_tokenizer.eos_token

        self.opt = AutoModelForCausalLM.from_pretrained(
            str(runtime_dir / OPT_DIR_NAME),
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(self.device)
        self.opt.requires_grad_(False)
        self.opt.eval()
        if bool(self.prm.get("gradient_checkpointing", True)):
            self.opt.gradient_checkpointing_enable()
        hidden = int(self.opt.get_input_embeddings().embedding_dim)
        self.projection = nn.Linear(FEATURE_SHAPE[1], hidden).to(self.device, dtype=torch.float32)
        state = torch.load(projection_path, map_location="cpu", weights_only=True)
        self.projection.load_state_dict(state, strict=True)
        self.max_text_length = int(self.prm.get("max_text_length", 50))
        self.max_new_tokens = int(self.prm.get("max_new_tokens", 24))
        self.num_beams = int(self.prm.get("num_beams", 3))
        self.prompt = str(self.prm.get("caption_prompt", "a photo of "))

    def train(self, mode=True):
        super().train(mode)
        self.opt.eval()
        return self

    def _features(self, value):
        if value.ndim != 3 or tuple(value.shape[1:]) != FEATURE_SHAPE:
            raise ValueError(f"Expected (B,{FEATURE_SHAPE[0]},{FEATURE_SHAPE[1]}) features.")
        if not torch.isfinite(value).all():
            raise ValueError("Cached features contain NaN or Inf.")
        return value.to(self.device, dtype=torch.float32, non_blocking=True)

    def _reference_text(self, labels):
        if labels.ndim == 3:
            valid = (labels >= 0).any(dim=2)
            if not valid.any(dim=1).all():
                raise ValueError("At least one sample has no caption reference.")
            indices = valid.long().argmax(dim=1)
            labels = labels[torch.arange(labels.shape[0], device=labels.device), indices]
        clean = labels.detach().cpu().clone()
        clean[clean < 0] = self.opt_tokenizer.pad_token_id
        return self.opt_tokenizer.batch_decode(clean, skip_special_tokens=True)

    def _loss(self, features, labels):
        texts = [self.prompt + text.strip() for text in self._reference_text(labels)]
        encoded = self.opt_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        ids = encoded.input_ids.to(self.device)
        mask = encoded.attention_mask.to(self.device)
        targets = ids.clone()
        targets[mask == 0] = -100
        visual = self.projection(features).to(self.dtype)
        text = self.opt.get_input_embeddings()(ids).to(self.dtype)
        embeddings = torch.cat((visual, text), dim=1)
        prefix_labels = torch.full(visual.shape[:2], -100, dtype=torch.long, device=self.device)
        joined_labels = torch.cat((prefix_labels, targets), dim=1)
        prefix_mask = torch.ones(visual.shape[:2], dtype=mask.dtype, device=self.device)
        return self.opt(
            inputs_embeds=embeddings,
            attention_mask=torch.cat((prefix_mask, mask), dim=1),
            labels=joined_labels,
            use_cache=False,
        ).loss

    def _generate(self, features):
        visual = self.projection(features).to(self.dtype)
        prompt = self.opt_tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        prompt = prompt.expand(features.shape[0], -1)
        text = self.opt.get_input_embeddings()(prompt).to(self.dtype)
        embeddings = torch.cat((visual, text), dim=1)
        mask = torch.ones(embeddings.shape[:2], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            generated = self.opt.generate(
                inputs_embeds=embeddings,
                attention_mask=mask,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                eos_token_id=self.opt_tokenizer.eos_token_id,
                pad_token_id=self.opt_tokenizer.pad_token_id,
            )
        texts = self.opt_tokenizer.batch_decode(generated, skip_special_tokens=True)
        cleaned = []
        for text in texts:
            # OPT occasionally emits a second caption after a newline. Keep the
            # first non-empty caption instead of merging unrelated descriptions.
            lines = [" ".join(line.split()) for line in str(text).splitlines()]
            normalized = next((line for line in lines if line), "")
            if normalized.lower().startswith(self.prompt.strip().lower()):
                normalized = normalized[len(self.prompt.strip()):].strip()
            words = normalized.split()
            # Stop obvious repeated halves produced by greedy/beam decoding.
            for size in range(1, len(words) // 2 + 1):
                if words[:size] == words[size:2 * size]:
                    words = words[:size]
                    break
            cleaned.append(" ".join(words[:24]) or "image")
        encoded = self.opt_tokenizer(
            cleaned,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        return encoded.input_ids.to(self.device)

    def forward(self, features, captions=None):
        features = self._features(features)
        return self._loss(features, captions) if captions is not None else self._generate(features)

    def train_setup(self, prm):
        self.prm.update(prm or {})
        learning_rate = float(self.prm.get("lr", 1e-5))
        if not 0 < learning_rate <= 1:
            raise ValueError("lr must be in (0, 1].")
        self.optimizer = torch.optim.AdamW(self.projection.parameters(), lr=learning_rate, weight_decay=0.01)

    def learn(self, train_data):
        if self.optimizer is None:
            raise RuntimeError("train_setup() must be called before learn().")
        total = 0.0
        batches = 0
        for features, labels in train_data:
            self.optimizer.zero_grad(set_to_none=True)
            loss = self(features, labels.to(self.device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite BLIP-2 loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.projection.parameters(), 1.0)
            self.optimizer.step()
            total += float(loss.detach())
            batches += 1
        return 0.0, total / max(batches, 1)
