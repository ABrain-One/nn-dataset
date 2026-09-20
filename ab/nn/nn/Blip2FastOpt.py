"""Blip2FastOpt - Cached BLIP-2 visual features with a frozen OPT decoder.

This is the production OPT-based image captioning model. It uses precomputed
BLIP-2 visual query features (32, 768), a trainable pretrained language
projection, diverse multi-reference label sampling during training, and a
frozen OPT decoder.
"""

from __future__ import annotations

import torch

from ab.nn.util.captioning.blip2.text import caption_training_batch

from ab.nn.nn.Blip2Cached import Net as BaselineNet
from ab.nn.nn.Blip2Cached import supported_hyperparameters


class Net(BaselineNet):
    def train_setup(self, prm):
        super().train_setup(prm)

    def __init__(self, in_shape, out_shape, prm, device):
        options = dict(prm or {})
        # OPT is frozen, but gradients still pass through it to the projection.
        options.setdefault("gradient_checkpointing", False)
        super().__init__(in_shape, out_shape, options, device)
        self._reference_generator = torch.Generator(device="cpu")
        self._reference_generator.manual_seed(int(options.get("seed", 42)))

    def _select_reference_labels(self, labels):
        if labels.ndim != 3:
            return labels
        valid = (labels >= 0).any(dim=2)
        if not valid.any(dim=1).all():
            raise ValueError("At least one sample has no caption reference.")
        if self.training:
            selected = []
            for row in valid.detach().cpu():
                choices = torch.nonzero(row, as_tuple=False).flatten()
                position = torch.randint(
                    len(choices),
                    (1,),
                    generator=self._reference_generator,
                ).item()
                selected.append(int(choices[position]))
            indices = torch.tensor(selected, device=labels.device)
        else:
            indices = valid.long().argmax(dim=1)
        rows = torch.arange(labels.shape[0], device=labels.device)
        return labels[rows, indices]

    def _reference_text(self, labels):
        labels = self._select_reference_labels(labels)
        clean = labels.detach().cpu().clone()
        clean[clean < 0] = self.opt_tokenizer.pad_token_id
        return self.opt_tokenizer.batch_decode(clean, skip_special_tokens=True)

    @staticmethod
    def _caption_targets(ids, attention_mask, prompt_length):
        targets = ids.clone()
        targets[attention_mask == 0] = -100
        targets[:, : min(int(prompt_length), targets.shape[1])] = -100
        return targets

    def _loss(self, features, labels):
        ids, mask, targets = caption_training_batch(
            self.opt_tokenizer, self.prompt, self._reference_text(labels),
            self.max_text_length, self.device, add_special_tokens=True,
        )
        visual = self.projection(features).to(self.dtype)
        text = self.opt.get_input_embeddings()(ids).to(self.dtype)
        embeddings = torch.cat((visual, text), dim=1)
        prefix_labels = torch.full(
            visual.shape[:2], -100, dtype=torch.long, device=self.device
        )
        joined_labels = torch.cat((prefix_labels, targets), dim=1)
        prefix_mask = torch.ones(
            visual.shape[:2], dtype=mask.dtype, device=self.device
        )
        return self.opt(
            inputs_embeds=embeddings,
            attention_mask=torch.cat((prefix_mask, mask), dim=1),
            labels=joined_labels,
            use_cache=False,
        ).loss


__all__ = ["Net", "supported_hyperparameters"]
