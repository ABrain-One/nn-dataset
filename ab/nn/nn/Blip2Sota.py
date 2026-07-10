"""
Blip2Sota - LLM-Optimized Image Captioning Model (Fixed Version)
Base: Blip2Fast / Blip2Improved
Improvements: 
  - Advanced Bottleneck Projection (768 -> 1536 -> 768)
  - LayerNorm and GELU for improved stability and non-linearity
  - Dynamic token translation between COCO IDs and GPT-2 vocabulary IDs
  - Compatible with standard blip2_processor
"""

import torch
import torch.nn as nn
from transformers import (
    Blip2Processor,
    Blip2Model,
    GPT2LMHeadModel,
    GPT2Config,
    GPT2Tokenizer,
)
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def supported_hyperparameters():
    return {'lr', 'batch'}


class FrozenBlip2Encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.hidden_size = 768  # Known hidden size for blip2-opt-2.7b Q-Former
        self.blip2 = None       # Lazy load to save 15GB VRAM
        self.eval()

    def forward(self, pixel_values): 
        # Skip if input features are already cached
        if pixel_values.dim() == 3 and pixel_values.shape[-1] == self.hidden_size:
            return pixel_values.float()

        if self.blip2 is None:
            # Lazy load ONLY when raw image (4D) is provided
            print("✅ Blip2Sota: Lazy Loading 15GB BLIP-2 Encoder on-demand...")
            self.blip2 = Blip2Model.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map={"": self.device}
            )
            self.blip2.eval()
            for param in self.blip2.parameters():
                param.requires_grad = False

        self.blip2.eval()
        with torch.no_grad():
            outputs = self.blip2.get_qformer_features(pixel_values=pixel_values)
        return outputs.last_hidden_state.float() # Safe cast for projection layer


class CaptionDecoder(nn.Module):
    def __init__(self, q_former_hidden: int, device):
        super().__init__()
        self.device = device
        gpt2_id = "gpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        config = GPT2Config.from_pretrained(gpt2_id)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_id, config=config)
        self.gpt2 = self.gpt2.to(device)
        self.gpt2_hidden = config.n_embd 

        # [TRAINABLE/LLM_IMPROVED] Bottleneck expansion for better feature mapping
        self.visual_projection = nn.Sequential(
            nn.Linear(q_former_hidden, self.gpt2_hidden * 2),
            nn.LayerNorm(self.gpt2_hidden * 2),
            nn.GELU(),
            nn.Linear(self.gpt2_hidden * 2, self.gpt2_hidden),
            nn.LayerNorm(self.gpt2_hidden),
            nn.GELU(),
        ).to(device)
        
        self.num_visual_tokens = 32

        # Checkpoint auto-load removed for strict reproducibility.

    def forward(self, visual_features, caption_ids=None, labels=None, text_mask=None):
        B = visual_features.shape[0]
        visual_embeds = self.visual_projection(visual_features)
        if caption_ids is not None:
            text_embeds = self.gpt2.transformer.wte(caption_ids)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            
            if labels is None:
                ignore_labels = torch.full((B, self.num_visual_tokens), -100, dtype=torch.long, device=self.device)
                labels = torch.cat([ignore_labels, caption_ids], dim=1)
            else:
                ignore_labels = torch.full((B, self.num_visual_tokens), -100, dtype=torch.long, device=self.device)
                labels = torch.cat([ignore_labels, labels], dim=1)
                
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)
            if text_mask is None:
                pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                text_mask = (caption_ids != pad_token_id).long()
            attention_mask[:, self.num_visual_tokens:] = text_mask
            
            outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            return outputs.loss
        else:
            out_ids = self.gpt2.generate(
                inputs_embeds=visual_embeds,
                max_new_tokens=60,
                num_beams=5,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                length_penalty=1.1,
                early_stopping=True
            )
            return out_ids


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        
        # 3. Seed support
        seed = int(prm.get("seed", 42)) if isinstance(prm, dict) else 42
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        self.device = device
        self.vocab_size = out_shape[0] if out_shape else 50257
        self.encoder = FrozenBlip2Encoder(device)
        self.decoder = CaptionDecoder(q_former_hidden=self.encoder.hidden_size, device=device)
        self.criterion = lambda outputs, labels: torch.tensor(0.0, device=self.device, requires_grad=True)
        self.idx2word = None
        self.word2idx = None
        self._print_param_stats()

    def _print_param_stats(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ Blip2Sota (GELU/LN): {total:,} params | {trainable:,} trainable")

    def _ensure_vocab(self):
        if self.vocab_size == 50257: return
        if self.idx2word is not None and len(self.idx2word) > 0: return
        from ab.nn.loader.coco_.Caption import GLOBAL_CAPTION_VOCAB
        from ab.nn.util.Const import data_dir
        
        # If GLOBAL_CAPTION_VOCAB is empty (cached mode), load it manually
        if not GLOBAL_CAPTION_VOCAB.get('idx2word'):
            vocab_path = os.path.join(data_dir, 'coco', 'vocab.pth')
            if os.path.exists(vocab_path):
                vocab_data = torch.load(vocab_path)
                GLOBAL_CAPTION_VOCAB['word2idx'] = vocab_data['word2idx']
                GLOBAL_CAPTION_VOCAB['idx2word'] = vocab_data['idx2word']
                
        self.idx2word = GLOBAL_CAPTION_VOCAB.get('idx2word', {})
        self.word2idx = GLOBAL_CAPTION_VOCAB.get('word2idx', {})

    def _captions_to_gpt2_ids(self, captions):
        self._ensure_vocab()
        texts = []
        for seq in captions:
            words = []
            for idx in seq.cpu().tolist():
                pad_id = self.word2idx.get('<PAD>', 0)
                sos_id = self.word2idx.get('<SOS>', 1)
                eos_id = self.word2idx.get('<EOS>', 3)
                if idx == pad_id or idx == sos_id:
                    continue
                if idx == eos_id:
                    break
                word = self.idx2word.get(idx, '')
                if word and word not in ('<UNK>', '<PAD>', '<SOS>', '<EOS>'):
                    words.append(word)
            texts.append(' '.join(words).strip() or 'image')

        enc = self.decoder.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=60,
            add_special_tokens=True,
        )
        input_ids = enc.input_ids.to(self.device)
        labels = input_ids.clone()
        labels[enc.attention_mask.to(self.device) == 0] = -100
        return input_ids, labels

    def _gpt2_ids_to_coco_ids(self, generated_ids):
        self._ensure_vocab()
        texts = self.decoder.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        
        coco_ids = []
        max_len = 40
        for text in texts:
            try:
                from nltk.tokenize import word_tokenize
                words = word_tokenize(text.lower())
            except Exception:
                words = text.lower().split()
            ids = [self.word2idx.get('<SOS>', 1)]
            for word in words:
                idx = self.word2idx.get(word, None)
                if idx is None:
                    idx = self.word2idx.get(word.lower(), None)
                if idx is None:
                    idx = self.word2idx.get(word.capitalize(), None)
                if idx is None:
                    idx = self.word2idx.get('<UNK>', 2)
                ids.append(idx)
            ids.append(self.word2idx.get('<EOS>', 3))
            
            if len(ids) < max_len:
                ids += [self.word2idx.get('<PAD>', 0)] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
            coco_ids.append(ids)
            
        return torch.tensor(coco_ids, dtype=torch.long, device=self.device)

    def forward(self, pixel_values, captions=None):
        self.encoder.eval()
        visual_features = self.encoder(pixel_values)
        if captions is not None:
            if captions.dim() == 3: captions = captions[:, 0, :]
            
            if self.vocab_size == 50257:
                # Fast Cached Loader (GPT-2 IDs) — randomly pick one caption per image
                if captions.dim() == 3 and captions.shape[1] > 1:
                    cap_idx = torch.randint(0, captions.shape[1], (1,)).item()
                    captions = captions[:, cap_idx, :]
                elif captions.dim() == 3:
                    captions = captions[:, 0, :]
                labels, text_mask = self._make_gpt2_labels_and_mask(captions)
                return self.decoder(visual_features, captions, labels, text_mask)
            else:
                # Standard Loader (COCO IDs) -> Convert to GPT-2 IDs
                self._ensure_vocab()
                input_ids, labels = self._captions_to_gpt2_ids(captions)
                return self.decoder(visual_features, input_ids, labels)
        else:
            was_training = self.decoder.training
            self.decoder.eval()
            with torch.no_grad():
                generated_gpt2_ids = self.decoder(visual_features, None)
            if was_training:
                self.decoder.train()
                
            if self.vocab_size == 50257:
                return generated_gpt2_ids
            else:
                return self._gpt2_ids_to_coco_ids(generated_gpt2_ids)

    def _make_gpt2_labels_and_mask(self, input_ids):
        eos_id = self.decoder.tokenizer.eos_token_id
        labels = input_ids.clone()
        text_mask = torch.ones_like(input_ids, dtype=torch.long)
        
        for i in range(input_ids.size(0)):
            eos_positions = (input_ids[i] == eos_id).nonzero(as_tuple=False).flatten()
            if len(eos_positions) >= 1:  # Fix: >= 1 (not > 1)
                first_eos = int(eos_positions[0])
                labels[i, first_eos + 1:] = -100
                text_mask[i, first_eos + 1:] = 0
                
        return labels, text_mask

    def train_setup(self, prm):
        visual_params = list(self.decoder.visual_projection.parameters())
        gpt2_top_blocks = []
        
        # Differential Learning Rate & Freezing
        for name, param in self.decoder.gpt2.named_parameters():
            if "h.8." in name or "h.9." in name or "h.10." in name or "h.11." in name or "ln_f" in name:
                gpt2_top_blocks.append(param)
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        param_groups = [
            {'params': visual_params, 'lr': prm.get('lr', 1e-4)},
            {'params': gpt2_top_blocks, 'lr': 1e-5}
        ]
        
        self.optimizer = torch.optim.AdamW(param_groups)
        
        # Setup Cosine Annealing Scheduler with 5% Warmup
        try:
            from transformers import get_cosine_schedule_with_warmup
            epochs = prm.get('epoch_max', 10)
            num_training_steps = epochs * 3697 # Approx total batches
            num_warmup_steps = int(0.05 * num_training_steps)
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=num_warmup_steps, 
                num_training_steps=num_training_steps
            )
        except ImportError:
            self.scheduler = None

    def learn(self, train_data):
        # [FROZEN] Keep encoder in eval mode always
        self.encoder.eval()
        # [TRAINABLE] Only decoder learns
        self.decoder.train()
        self._ensure_vocab()
        total_loss, num_batches = 0.0, 0
        try:
            for images, captions in train_data:
                images, captions = images.to(self.device), captions.to(self.device)
                if captions.dim() == 3: captions = captions[:, 0, :]
                self.optimizer.zero_grad()
                loss = self.forward(images, captions)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
                self.optimizer.step()
                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    self.scheduler.step()
                total_loss += loss.item()
                num_batches += 1
        except Exception as e:
            from ab.nn.util.Exception import LearnTimeException
            if isinstance(e, LearnTimeException):
                print(f"[INFO] Epoch time limit reached after {num_batches} batches.")
            else:
                raise
        return 0.0, total_loss / max(num_batches, 1)
