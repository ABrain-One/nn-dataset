import torch
import torch.nn as nn
from transformers import Blip2Processor, Blip2Model, GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def supported_hyperparameters():
    return {'lr', 'batch'}

class FrozenBlip2Encoder(nn.Module):

    def __init__(self, device):
        super().__init__()
        model_id = 'Salesforce/blip2-opt-2.7b'
        print(f'[Blip2FastSota] Loading frozen vision encoder from: {model_id}')
        self.blip2 = Blip2Model.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={'': device})
        for param in self.blip2.parameters():
            param.requires_grad = False
        self.blip2.eval()
        self.eval()
        self.hidden_size = self.blip2.config.qformer_config.hidden_size

    def forward(self, pixel_values):
        self.blip2.eval()
        with torch.no_grad():
            outputs = self.blip2.get_qformer_features(pixel_values=pixel_values)
        return outputs.last_hidden_state

class CaptionDecoder(nn.Module):

    def __init__(self, q_former_hidden: int, device):
        super().__init__()
        self.device = device
        gpt2_id = 'gpt2'
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        config = GPT2Config.from_pretrained(gpt2_id)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_id, config=config)
        self.gpt2 = self.gpt2.to(device)
        self.gpt2_hidden = config.n_embd
        self.visual_projection = nn.Sequential(nn.Linear(q_former_hidden, self.gpt2_hidden), nn.LayerNorm(self.gpt2_hidden), nn.GELU()).to(device)
        self.num_visual_tokens = 32

    def forward(self, visual_features, caption_ids=None):
        B = visual_features.shape[0]
        visual_embeds = self.visual_projection(visual_features)
        if caption_ids is not None:
            text_embeds = self.gpt2.transformer.wte(caption_ids)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            ignore_labels = torch.full((B, self.num_visual_tokens), -100, dtype=torch.long, device=self.device)
            labels = torch.cat([ignore_labels, caption_ids], dim=1)
            outputs = self.gpt2(inputs_embeds=inputs_embeds, labels=labels)
            return outputs.loss
        else:
            outputs_embeds = visual_embeds
            generated = []
            past_key_values = None
            for _ in range(40):
                out = self.gpt2(inputs_embeds=outputs_embeds, past_key_values=past_key_values, use_cache=True)
                next_token_logits = out.logits[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                generated.append(next_token)
                past_key_values = out.past_key_values
                outputs_embeds = self.gpt2.transformer.wte(next_token)
                if (next_token == self.tokenizer.eos_token_id).all():
                    break
            return torch.cat(generated, dim=1)

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape[0])
        self.prm = prm
        self.encoder = FrozenBlip2Encoder(device)
        self.decoder = CaptionDecoder(q_former_hidden=self.encoder.hidden_size, device=device)
        self.criterion = lambda outputs, labels: outputs # Generative loss is already computed in forward()
        self.idx2word = None
        self.word2idx = None
        self.optimizer = None
        print('✅ Blip2FastSota loaded: Frozen encoder + Trainable GPT2-small decoder')
        self._print_param_stats()

    def _print_param_stats(self):
        total = sum((p.numel() for p in self.parameters()))
        trainable = sum((p.numel() for p in self.parameters() if p.requires_grad))
        print(f'   Total params: {total:,} | Trainable: {trainable:,} ({100 * trainable / total:.1f}%)')

    def _ensure_vocab(self):
        if self.idx2word is not None:
            return
        from ab.nn.loader.coco_.Caption import GLOBAL_CAPTION_VOCAB
        self.idx2word = GLOBAL_CAPTION_VOCAB.get('idx2word', {})
        self.word2idx = GLOBAL_CAPTION_VOCAB.get('word2idx', {})

    def forward(self, pixel_values, captions=None):
        self.encoder.eval()
        visual_features = self.encoder(pixel_values)
        if captions is not None:
            self._ensure_vocab()
            if captions.dim() == 3:
                captions = captions[:, 0, :]
            return self.decoder(visual_features, captions)
        else:
            return self.decoder(visual_features, None)

    def train_setup(self, prm):
        self.prm = prm
        self.batch_size = prm.get('batch', 32)
        trainable_params = [p for p in self.decoder.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=prm.get('lr', 0.0001))

    def learn(self, train_data):
        self.encoder.eval()
        self.decoder.train()
        self._ensure_vocab()
        total_loss = 0.0
        num_batches = 0
        for (images, captions) in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)
            if captions.dim() == 3:
                captions = captions[:, 0, :]
            self.optimizer.zero_grad()
            loss = self.forward(images, captions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return (0.0, total_loss / max(num_batches, 1))
        self.max_captions = 40
        self.gpt2_max_positions = 40
        self.gpt2_dim = self.gpt2.config.n_embd