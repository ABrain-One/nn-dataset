"""
Blip2Fast - Cached BLIP-2/Q-Former + OPT-2.7B Image Captioning Model

Goal:
  Use cached BLIP-2/Q-Former features (B, 32, 768) and a frozen OPT-2.7B decoder.

Important:
  cached_blip2fast_processor usually provides captions in GPT-2 token ID space.
  This model correctly converts:
    training:   GPT-2 IDs -> text -> OPT IDs
    inference: OPT IDs   -> text -> GPT-2 IDs

Compatible with:
  - cached_blip2fast_processor

Not compatible with:
  - raw image tensors
"""

import os
import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer, OPTForCausalLM, GPT2Tokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ── Config ────────────────────────────────────────────────────────────────────
OPT_MODEL_ID = "facebook/opt-2.7b"

NUM_VISUAL_TOKENS = 32
QFORMER_HIDDEN = 768

GPT2_VOCAB_SIZE = 50257
TRAIN_MAX_LEN = 50
MAX_CAPTION_LEN = 16
PROMPT_TEXT = "a photo of "
BEAM_WIDTH = 3

_default_proj_path = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "blip2fast",
    "pretrained_projection_opt27b.pth",
)
PROJECTION_CACHE_PATH = os.environ.get("BLIP2_PROJECTION_PATH", _default_proj_path)
# ──────────────────────────────────────────────────────────────────────────────


def supported_hyperparameters():
    return {"lr", "batch"}


def _seed_everything(seed=42):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ==============================================================================
# Cache-only encoder
# ==============================================================================
class FrozenBlip2Encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.hidden_size = QFORMER_HIDDEN
        print("[Blip2Fast] Cache-only encoder: BLIP-2 is NOT loaded at runtime.")

    def forward(self, pixel_values):
        if (
            torch.is_tensor(pixel_values)
            and pixel_values.dim() == 3
            and pixel_values.shape[1] == NUM_VISUAL_TOKENS
            and pixel_values.shape[-1] == self.hidden_size
        ):
            return pixel_values.to(self.device).float()

        raise RuntimeError(
            "[Blip2Fast] Expected cached BLIP-2/Q-Former features with shape (B, 32, 768). "
            "Use transform='cached_blip2fast_processor'."
        )


# ==============================================================================
# OPT-2.7B decoder
# ==============================================================================
class OPTCaptionDecoder(nn.Module):
    def __init__(self, device, prm=None):
        super().__init__()

        self.device = device
        self.prm = prm if isinstance(prm, dict) else {}

        print(f"[Blip2Fast] Loading frozen OPT decoder: {OPT_MODEL_ID}")

        self.opt_tokenizer = AutoTokenizer.from_pretrained(OPT_MODEL_ID, use_fast=False)
        if self.opt_tokenizer.pad_token is None:
            self.opt_tokenizer.pad_token = self.opt_tokenizer.eos_token

        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token

        self.opt = OPTForCausalLM.from_pretrained(
            OPT_MODEL_ID,
            torch_dtype=torch.float16,
            device_map={"": device},
        )

        for p in self.opt.parameters():
            p.requires_grad = False

        self.opt.eval()

        # Safer than config.hidden_size because OPT input embedding dim can be model-specific.
        self.opt_embed_dim = int(self.opt.get_input_embeddings().embedding_dim)

        self.visual_projection = nn.Linear(QFORMER_HIDDEN, self.opt_embed_dim).to(device)

        self._init_or_load_projection()

    # --------------------------------------------------------------------------
    # Projection loading
    # --------------------------------------------------------------------------
    def _init_or_load_projection(self):
        nn.init.normal_(self.visual_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.visual_projection.bias)

        if os.path.exists(PROJECTION_CACHE_PATH):
            print(f"[Blip2Fast] Loading pretrained BLIP-2 language_projection from: {PROJECTION_CACHE_PATH}")
            state = torch.load(PROJECTION_CACHE_PATH, map_location=self.device, weights_only=True)

            try:
                self.visual_projection.load_state_dict(state, strict=True)
                print("[Blip2Fast] Pretrained projection loaded successfully.")
                return
            except Exception as e:
                print(f"[Blip2Fast WARN] Projection state mismatch, using random init. Error: {e}")
                return

        auto_extract = os.environ.get("BLIP2_EXTRACT_PROJECTION", "0") == "1"
        if not auto_extract:
            print(
                "[Blip2Fast WARN] Pretrained projection file not found. "
                "Using random projection init.\n"
                f"Expected path: {PROJECTION_CACHE_PATH}\n"
                "To extract once, run with: BLIP2_EXTRACT_PROJECTION=1"
            )
            return

        self._extract_projection_once()

    def _extract_projection_once(self):
        """
        Optional one-time extraction of Salesforce BLIP-2 language_projection.

        Warning:
          This temporarily loads Salesforce/blip2-opt-2.7b and may need large RAM/VRAM.
          Use only if your machine can handle it.
        """
        print("[Blip2Fast] Extracting Salesforce/blip2-opt-2.7b language_projection once...")

        os.makedirs(os.path.dirname(PROJECTION_CACHE_PATH), exist_ok=True)

        try:
            from transformers import Blip2ForConditionalGeneration
            import gc

            temp_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="cpu",
            )

            if not hasattr(temp_model, "language_projection"):
                raise RuntimeError("Salesforce BLIP-2 model has no language_projection attribute.")

            state = temp_model.language_projection.state_dict()
            torch.save(state, PROJECTION_CACHE_PATH)

            del temp_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[Blip2Fast] Projection saved to: {PROJECTION_CACHE_PATH}")

            loaded = torch.load(PROJECTION_CACHE_PATH, map_location=self.device, weights_only=True)
            self.visual_projection.load_state_dict(loaded, strict=True)
            print("[Blip2Fast] Extracted projection loaded successfully.")

        except Exception as e:
            print(f"[Blip2Fast WARN] Could not extract pretrained projection. Using random init. Error: {e}")

    # --------------------------------------------------------------------------
    # Token utilities
    # --------------------------------------------------------------------------
    def _safe_ids(self, ids):
        out = []
        for x in ids:
            try:
                v = int(x)
            except Exception:
                continue
            if v >= 0:
                out.append(v)
        return out

    def gpt2_ids_to_texts(self, gpt2_ids):
        """
        Convert cached GPT-2 token IDs into plain text captions.
        """
        texts = []

        for seq in gpt2_ids.detach().cpu().tolist():
            ids = self._safe_ids(seq)
            text = self.gpt2_tokenizer.decode(ids, skip_special_tokens=True)
            text = str(text).replace("\n", " ").strip()

            if not text:
                text = "image"

            texts.append(text)

        return texts

    def texts_to_opt_ids(self, texts):
        """
        Convert text captions into OPT token IDs and labels.

        We add prompt conditioning:
          "a photo of " + caption + eos

        Prompt tokens are attended to but ignored in the loss.
        Only actual caption/eos tokens are trained.
        """
        eos = self.opt_tokenizer.eos_token or "</s>"
        bos = self.opt_tokenizer.bos_token or "</s>"
        prompt = bos + PROMPT_TEXT

        full_texts = []
        for text in texts:
            text = str(text).replace("\n", " ").strip()
            if not text:
                text = "image"
            full_texts.append(prompt + text + eos)

        enc = self.opt_tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=TRAIN_MAX_LEN,
            add_special_tokens=False,
        )

        input_ids = enc.input_ids.to(self.device)
        attention_mask = enc.attention_mask.to(self.device)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Mask prompt tokens from loss.
        prompt_ids = self.opt_tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]

        prompt_len = int(prompt_ids.numel())
        if prompt_len > 0:
            labels[:, :prompt_len] = -100

        return input_ids, labels, attention_mask

    def opt_ids_to_gpt2_ids(self, opt_generated_ids):
        """
        Convert OPT generated token IDs back into GPT-2 token IDs for existing metrics.
        """
        texts = self.opt_tokenizer.batch_decode(
            opt_generated_ids.detach().cpu(),
            skip_special_tokens=True,
        )

        cleaned = []
        for text in texts:
            cleaned.append(self._clean_generated_caption(text))

        gpt2 = self.gpt2_tokenizer(
            cleaned,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=TRAIN_MAX_LEN,
        )

        return gpt2.input_ids.to(self.device)

    def _clean_generated_caption(self, text):
        text = str(text).replace("\n", " ").strip()

        if text.lower().startswith("</s>"):
            text = text[len("</s>"):].strip()

        if text.lower().startswith(PROMPT_TEXT.strip()):
            text = text[len(PROMPT_TEXT.strip()):].strip()

        text = " ".join(text.split())

        # Prevent chained captions by splitting at common starting phrases
        lower_text = text.lower()
        for phrase in [" a photo of", " a close up of", " a close-up of", " a picture of", " an image of", " a view of", " a bedroom that", " a room that", " a kitchen that"]:
            if phrase in lower_text:
                idx = lower_text.find(phrase)
                text = text[:idx].strip()
                lower_text = text.lower()

        # Prevent multi-caption paragraph output.
        words = text.split()
        if len(words) > 18:
            text = " ".join(words[:18])

        if not text:
            text = "image"

        return text

    # --------------------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------------------
    def forward(self, visual_features, opt_input_ids=None, opt_labels=None, opt_attention_mask=None):
        """
        Training:
          visual_features: (B, 32, 768)
          opt_input_ids: OPT token IDs
          opt_labels: OPT labels with pads/prompt masked as -100
          opt_attention_mask: OPT attention mask

        Inference:
          visual_features only
        """
        B = visual_features.shape[0]

        visual_embeds = self.visual_projection(visual_features.float())
        visual_embeds = visual_embeds.to(self.opt.dtype)

        if opt_input_ids is not None:
            text_embeds = self.opt.get_input_embeddings()(opt_input_ids)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            inputs_embeds = inputs_embeds.to(self.opt.dtype)

            ignore_visual = torch.full(
                (B, visual_embeds.shape[1]),
                -100,
                dtype=torch.long,
                device=self.device,
            )
            full_labels = torch.cat([ignore_visual, opt_labels], dim=1)

            visual_mask = torch.ones(
                B,
                visual_embeds.shape[1],
                dtype=torch.long,
                device=self.device,
            )

            if opt_attention_mask is None:
                opt_attention_mask = (opt_input_ids != self.opt_tokenizer.pad_token_id).long()

            attention_mask = torch.cat([visual_mask, opt_attention_mask], dim=1)

            out = self.opt(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=full_labels,
                use_cache=False,
            )
            return out.loss

        return self.generate(visual_embeds)

    # --------------------------------------------------------------------------
    # Generation
    # --------------------------------------------------------------------------
    def _prompt_embeds_and_mask(self, batch_size):
        bos = self.opt_tokenizer.bos_token or "</s>"
        prompt_ids = self.opt_tokenizer(
            bos + PROMPT_TEXT,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self.device)

        prompt_ids = prompt_ids.expand(batch_size, -1)
        prompt_embeds = self.opt.get_input_embeddings()(prompt_ids).to(self.opt.dtype)
        prompt_mask = torch.ones(
            batch_size,
            prompt_ids.shape[1],
            dtype=torch.long,
            device=self.device,
        )

        return prompt_ids, prompt_embeds, prompt_mask

    def generate(self, visual_embeds):
        B = visual_embeds.shape[0]

        prompt_ids, prompt_embeds, prompt_mask = self._prompt_embeds_and_mask(B)

        inputs_embeds = torch.cat([visual_embeds, prompt_embeds], dim=1)
        visual_mask = torch.ones(
            B,
            visual_embeds.shape[1],
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.cat([visual_mask, prompt_mask], dim=1)

        try:
            generated = self.opt.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=MAX_CAPTION_LEN,
                num_beams=BEAM_WIDTH,
                no_repeat_ngram_size=2,
                repetition_penalty=1.25,
                length_penalty=0.35,
                early_stopping=True,
                eos_token_id=self.opt_tokenizer.eos_token_id,
                pad_token_id=self.opt_tokenizer.pad_token_id,
                bos_token_id=self.opt_tokenizer.bos_token_id,
            )
            return generated

        except Exception as e:
            print(f"[Blip2Fast WARN] Beam generation failed: {e}. Falling back to greedy.")
            return self._greedy_decode(inputs_embeds)

    def _greedy_decode(self, initial_embeds):
        B = initial_embeds.shape[0]

        past_key_values = None
        generated = []
        curr_embeds = initial_embeds

        eos_id = self.opt_tokenizer.eos_token_id
        unfinished = torch.ones(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            for _ in range(MAX_CAPTION_LEN):
                if past_key_values is None:
                    out = self.opt(inputs_embeds=curr_embeds, use_cache=True)
                else:
                    out = self.opt(
                        input_ids=generated[-1],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
                next_token = next_token.masked_fill(~unfinished.unsqueeze(1), eos_id)

                generated.append(next_token)
                past_key_values = out.past_key_values

                unfinished = unfinished & (next_token.squeeze(1) != eos_id)
                if not unfinished.any():
                    break

        if not generated:
            return torch.empty(B, 0, dtype=torch.long, device=self.device)

        return torch.cat(generated, dim=1)


# ==============================================================================
# Optional GPT-2 vocab wrappers for old metric compatibility
# ==============================================================================
class GPT2Idx2WordWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get(self, idx, default=""):
        try:
            return self.tokenizer.decode([int(idx)], skip_special_tokens=True).strip()
        except Exception:
            return default

    def __len__(self):
        return self.tokenizer.vocab_size


class GPT2Word2IdxWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get(self, word, default=2):
        try:
            ids = self.tokenizer.encode(str(word), add_special_tokens=False)
            return ids[0] if ids else default
        except Exception:
            return default

    def __len__(self):
        return self.tokenizer.vocab_size


# ==============================================================================
# Main Net
# ==============================================================================
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()

        self.device = device
        self.prm = prm if isinstance(prm, dict) else {}

        seed = int(self.prm.get("seed", 42))
        _seed_everything(seed)

        self.vocab_size = int(out_shape[0]) if out_shape else GPT2_VOCAB_SIZE

        print(f"[Blip2Fast] Hyperparameters: {self.prm}")
        print(f"[Blip2Fast] out_shape vocab_size: {self.vocab_size}")

        self.encoder = FrozenBlip2Encoder(device)
        self.decoder = OPTCaptionDecoder(device=device, prm=self.prm)

        self.criterion = lambda outputs, labels: torch.tensor(
            0.0,
            device=self.device,
            requires_grad=True,
        )

        self.idx2word = None
        self.word2idx = None
        self.optimizer = None

        self._print_param_stats()

    def _print_param_stats(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = 100.0 * trainable / max(total, 1)
        print(f"[Blip2Fast] Total params: {total:,} | Trainable: {trainable:,} ({pct:.6f}%)")

    def _ensure_vocab(self):
        try:
            from ab.nn.loader.coco_.Caption import GLOBAL_CAPTION_VOCAB

            if not isinstance(GLOBAL_CAPTION_VOCAB.get("idx2word"), GPT2Idx2WordWrapper):
                self.idx2word = GPT2Idx2WordWrapper(self.decoder.gpt2_tokenizer)
                self.word2idx = GPT2Word2IdxWrapper(self.decoder.gpt2_tokenizer)
                GLOBAL_CAPTION_VOCAB["idx2word"] = self.idx2word
                GLOBAL_CAPTION_VOCAB["word2idx"] = self.word2idx
                print("[Blip2Fast] Installed GPT-2 vocab wrappers for metrics.")
            else:
                self.idx2word = GLOBAL_CAPTION_VOCAB["idx2word"]
                self.word2idx = GLOBAL_CAPTION_VOCAB["word2idx"]
        except Exception as e:
            print(f"[Blip2Fast WARN] Could not install vocab wrapper: {e}")

    def _select_one_caption(self, captions):
        if captions.dim() == 3:
            cap_idx = torch.randint(0, captions.shape[1], (1,), device=captions.device).item()
            captions = captions[:, cap_idx, :]
        return captions

    def forward(self, pixel_values, captions=None):
        visual_features = self.encoder(pixel_values)

        if captions is not None:
            captions = self._select_one_caption(captions)
            captions = captions.long().to(self.device)

            texts = self.decoder.gpt2_ids_to_texts(captions)
            opt_input_ids, opt_labels, opt_attention_mask = self.decoder.texts_to_opt_ids(texts)

            return self.decoder(
                visual_features,
                opt_input_ids=opt_input_ids,
                opt_labels=opt_labels,
                opt_attention_mask=opt_attention_mask,
            )

        was_training = self.decoder.training
        self.decoder.eval()

        with torch.no_grad():
            opt_generated_ids = self.decoder(visual_features)

        if was_training:
            self.decoder.train()

        return self.decoder.opt_ids_to_gpt2_ids(opt_generated_ids)

    def train_setup(self, prm):
        if isinstance(prm, dict):
            self.prm.update(prm)

        lr = float(self.prm.get("lr", 1e-4))

        # Only train projection. OPT stays frozen.
        trainable_params = [p for p in self.decoder.visual_projection.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        print(f"[Blip2Fast] Optimizer: AdamW projection-only lr={lr:.2e}")

    def learn(self, train_data):
        self.encoder.eval()
        self.decoder.train()
        self.decoder.opt.eval()
        self._ensure_vocab()

        if self.optimizer is None:
            self.train_setup(self.prm)

        total_loss = 0.0
        n = 0

        try:
            for images, captions in train_data:
                if isinstance(images, list):
                    images = torch.stack(images)
                if isinstance(captions, list):
                    captions = torch.stack(captions)

                images = images.to(self.device)
                captions = captions.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                loss = self.forward(images, captions)

                if not torch.is_tensor(loss):
                    loss = torch.tensor(float(loss), device=self.device, requires_grad=True)

                if not torch.isfinite(loss):
                    print(f"[Blip2Fast WARN] Non-finite loss: {loss}")
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.decoder.visual_projection.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss += float(loss.detach().item())
                n += 1

        except Exception as e:
            from ab.nn.util.Exception import LearnTimeException

            if isinstance(e, LearnTimeException):
                print(f"[Blip2Fast] Epoch time limit reached after {n} batches.")
            else:
                raise

        avg_loss = total_loss / max(n, 1)
        print(f"[Blip2Fast] Epoch complete: {n} batches | avg_loss={avg_loss:.4f}")

        return 0.0, avg_loss
