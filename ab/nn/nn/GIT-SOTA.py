"""
GIT (Microsoft) - State-of-the-Art Image Captioning
License: MIT
Expected: 44-45% BLEU-4

Uses microsoft/git-large-coco pretrained weights.
Vision encoder (CLIP ViT) is frozen; text decoder is fine-tuned.
Transform: git_processor (COCO-normalized → denormalized in _prep_images)
"""

import re
import torch
import torch.nn as nn
from transformers import GitProcessor, GitForCausalLM
import os

# Suppress HuggingFace Tokenizers fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def supported_hyperparameters():
    return {'lr'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape[0])
        self.prm = prm

        model_id = "microsoft/git-large-coco"
        print(f"Loading GIT Model: {model_id}")

        self.processor = GitProcessor.from_pretrained(model_id, use_fast=False)
        self.model = GitForCausalLM.from_pretrained(model_id).to(device)

        for p in self.model.parameters():
            p.requires_grad = False

        print("✅ GIT loaded successfully.")

        self.idx2word = None
        self.word2idx = None
        self.optimizer = None

    # ------------------------------------------------------------------
    # Vocab helpers
    # ------------------------------------------------------------------
    def _ensure_vocab(self):
        if self.idx2word is not None:
            return
        from ab.nn.loader.coco_.Caption import GLOBAL_CAPTION_VOCAB
        self.idx2word = GLOBAL_CAPTION_VOCAB.get('idx2word', {})
        self.word2idx = GLOBAL_CAPTION_VOCAB.get('word2idx', {})

    def _ids_to_text(self, caption_ids):
        """COCO token IDs → raw English text (for training loss computation)."""
        self._ensure_vocab()
        texts = []
        for row in caption_ids:
            words = []
            for token in row:
                tid = token.item()
                if tid == 0:
                    continue
                w = self.idx2word.get(tid, '')
                if w == '<SOS>':
                    continue
                if w == '<EOS>':
                    break
                if w:
                    words.append(w)
            texts.append(" ".join(words))
        return texts

    def _text_to_coco_ids(self, text: str):
        """
        Generated English text → COCO vocab IDs.
        KEY FIX: strip punctuation so 'area.' → 'area' matches COCO vocab.
        """
        self._ensure_vocab()
        try:
            from nltk.tokenize import word_tokenize
            words = word_tokenize(text.lower())
        except Exception:
            words = text.lower().split()

        # Strip any non-alphanumeric characters (dots, commas, apostrophes etc.)
        # This is the critical fix: COCO vocab has no punctuation tokens
        words = [re.sub(r'[^a-z0-9]', '', w) for w in words]
        words = [w for w in words if w]  # remove empty strings after strip

        sos = self.word2idx.get('<SOS>', 1)
        eos = self.word2idx.get('<EOS>', 2)
        unk = self.word2idx.get('<UNK>', 0)
        return [sos] + [self.word2idx.get(w, unk) for w in words] + [eos]

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    def _prep_images(self, images):
        """
        Prepare images for HuggingFace GitProcessor.

        git_processor.py applies: Resize(224) → ToTensor → [0,1] (no custom normalize)
        We simply clamp to ensure [0,1] and pass to the processor.
        do_rescale=False tells HuggingFace NOT to divide by 255 again.
        """
        pixel_values = torch.clamp(images.float(), 0.0, 1.0)
        inputs = self.processor(images=pixel_values, return_tensors="pt", do_rescale=False)
        return inputs.pixel_values.to(self.device)


    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, images, captions=None):
        self._ensure_vocab()
        pixel_values = self._prep_images(images)
        B = images.shape[0]

        if captions is not None:
            # ---- Training: COCO IDs → text → HuggingFace native loss ----
            if captions.dim() == 3:
                captions = captions[:, 0, :]

            raw_texts = self._ids_to_text(captions)
            text_inputs = self.processor(
                text=raw_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=40,
            )
            # Explicit .long() ensures int64 — HuggingFace models require this
            input_ids      = text_inputs.input_ids.to(self.device).long()
            attention_mask = text_inputs.attention_mask.to(self.device).long()
            labels = input_ids.clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            return outputs.loss

        else:
            # ---- Inference: generate → strip punct → COCO ID logits ----
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values=pixel_values,
                    max_new_tokens=25,
                    num_beams=4,           # beam search for better quality
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                )

            generated_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            batch_token_ids = []
            max_len = 1
            for text in generated_texts:
                token_ids = self._text_to_coco_ids(text)  # punctuation stripped
                batch_token_ids.append(token_ids)
                max_len = max(max_len, len(token_ids))

            # Build one-hot logits tensor expected by the framework
            logits = torch.zeros(B, max_len, self.vocab_size, device=self.device)
            for b, token_ids in enumerate(batch_token_ids):
                for t, tid in enumerate(token_ids):
                    if 0 < tid < self.vocab_size:
                        logits[b, t, tid] = 100.0
            return logits

    # ------------------------------------------------------------------
    # Train setup
    # ------------------------------------------------------------------
    def train_setup(self, prm):
        self.prm = prm
        lr = prm.get('lr', 1e-5)
        # Store once — avoids re-scanning all parameters on every epoch
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(self.trainable_params) > 0:
            self.optimizer = torch.optim.AdamW(self.trainable_params, lr=lr, weight_decay=1e-2)
        else:
            self.optimizer = None
        self.scheduler = None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def learn(self, train_data):
        self.model.train()
        total_loss = 0.0

        for images, captions in train_data:
            images   = images.to(self.device)
            captions = captions.to(self.device)

            loss = self.forward(images, captions)

            if self.optimizer is not None:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
                self.optimizer.step()

            total_loss += loss.item()

        return 0.0, total_loss / max(len(train_data), 1)