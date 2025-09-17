import torch
import torch.nn as nn
import torchvision.models as models
import math
from ab.nn.loader.coco_.Caption import GLOBAL_CAPTION_VOCAB

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

# CNN Encoder (ResNet50 Backbone)
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 768)

    def forward(self, images):
        features = self.cnn(images)  # [B, 2048, H, W]
        pooled = self.pool(features).flatten(1)  # [B, 2048]
        return self.fc(pooled).unsqueeze(1)  # [B, 1, 768]

# Transformer Decoder with batch_first=True
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=6, nhead=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=2048, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        # tgt: [batch, seq] -> embed: [batch, seq, d_model]
        embedded = self.embedding(tgt)
        embedded = self.pos_encoding(embedded)
        seq_len = tgt.size(1)
        tgt_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).to(tgt.device)
        # memory: [batch, mem_seq, d_model] (mem_seq=1 from encoder)
        out = self.transformer_decoder(embedded, memory, tgt_mask=tgt_mask)
        return self.fc_out(out)  # [batch, seq, vocab_size]

# Main Net: Image Captioning Model
class Net(nn.Module):
    """
    CNN + Transformer-based Image Captioning Network
    """
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.encoder = CNNEncoder()
        self.decoder = TransformerDecoder(out_shape[0])
        self.vocab_size = out_shape[0]
        self.word2idx = GLOBAL_CAPTION_VOCAB.get('word2idx', None)
        self.idx2word = GLOBAL_CAPTION_VOCAB.get('idx2word', None)
        self.model_name = "ResNetTransformer"

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=self.word2idx['<PAD>']).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=3, factor=0.5)
        train_loader = prm.get('train_loader', None)
        dataset = getattr(train_loader, 'dataset', None) if train_loader is not None else None
        if dataset is not None and hasattr(dataset, 'word2idx'):
            self.word2idx = dataset.word2idx
            self.idx2word = dataset.idx2word

    def learn(self, train_data):
        if self.word2idx is None or self.idx2word is None:
            if hasattr(train_data, 'dataset'):
                self.word2idx = getattr(train_data.dataset, 'word2idx', None)
                self.idx2word = getattr(train_data.dataset, 'idx2word', None)
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)[:, 0, :]  # [B, T]
            tgt_input = captions[:, :-1]  # [B, T-1]
            tgt_output = captions[:, 1:]  # [B, T-1]
            self.optimizer.zero_grad()
            memory = self.encoder(images)  # [B, 1, 768]
            output = self.decoder(tgt_input, memory)  # [B, T-1, vocab_size]
            loss = self.criteria[0](output.view(-1, output.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def batch_beam_search(self, images, beam_width=4, max_len=20, length_penalty=0.7):
        """
        images: [B, C, H, W]
        Returns:
            preds: [B, max_seq_len, vocab_size] (one-hot style, for metric)
        """
        batch_size = images.size(0)
        results = []
        memory = self.encoder(images)  # [B, 1, 768]
        for i in range(batch_size):
            mem = memory[i:i+1]  # [1, 1, 768]
            best_caption = self.beam_search_generate_single(mem, beam_width, max_len, length_penalty)
            results.append(best_caption)
        # Pad and convert to one-hot for metric compatibility
        max_seq_len = max(len(r) for r in results)
        preds = torch.zeros(batch_size, max_seq_len, self.vocab_size).to(self.device)
        for i, seq in enumerate(results):
            for t, idx in enumerate(seq):
                preds[i, t, idx] = 1.0
        return preds

    def beam_search_generate_single(self, memory, beam_width=4, max_len=20, length_penalty=0.7):
        """
        Single image beam search, returns a list of token indices (without <SOS>).
        memory: [1, 1, 768]
        """
        word2idx = self.word2idx
        idx2word = self.idx2word
        device = self.device

        sequences = [(0.0, [word2idx['<SOS>']])]
        for _ in range(max_len):
            all_candidates = []
            for score, seq in sequences:
                if seq[-1] == word2idx['<EOS>']:
                    all_candidates.append((score, seq))
                    continue
                tgt = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq_len]
                out = self.decoder(tgt, memory)  # [1, seq_len, vocab_size]
                log_probs = out[0, -1].log_softmax(dim=0)  # Last token probs
                topk = torch.topk(log_probs, beam_width)
                for i in range(beam_width):
                    next_word = topk.indices[i].item()
                    next_score = score + topk.values[i].item()
                    all_candidates.append((next_score, seq + [next_word]))
            sequences = sorted(all_candidates, key=lambda tup: tup[0] / (len(tup[1]) ** length_penalty), reverse=True)[:beam_width]
            if all(seq[-1] == word2idx['<EOS>'] for _, seq in sequences):
                break
        best_seq = sequences[0][1]
        # Remove <SOS>, keep until <EOS> (but not including)
        if word2idx['<EOS>'] in best_seq:
            end_idx = best_seq.index(word2idx['<EOS>'])
            return best_seq[1:end_idx+1]
        else:
            return best_seq[1:]

    def forward(self, images, captions=None, hidden_state=None):
        if self.word2idx is None or self.idx2word is None:
            raise ValueError("word2idx and idx2word must be set before evaluation.")
        memory = self.encoder(images)  # [B, 1, 768]
        if captions is not None:
            tgt_input = captions[:, :-1]  # [B, T-1]
            return self.decoder(tgt_input, memory)  # [B, T-1, vocab_size]

    # Use beam search decoding for evaluation
        return self.batch_beam_search(images, beam_width=4, max_len=20, length_penalty=0.7)

    def beam_search_generate(self, image, word2idx, idx2word, beam_width=4, max_len=20, length_penalty=0.7):
        self.eval()
        self.word2idx = word2idx
        self.idx2word = idx2word
        device = self.device
        with torch.no_grad():
            image = image.unsqueeze(0).to(device)  # [1, C, H, W]
            memory = self.encoder(image).transpose(0, 1)  # [1, 1, d_model]
            
            # (score, sequence)
            sequences = [(0.0, [word2idx['<SOS>']])]
            
            for _ in range(max_len):
                all_candidates = []
                for score, seq in sequences:
                    if seq[-1] == word2idx['<EOS>']:
                        all_candidates.append((score, seq))
                        continue
                    tgt = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq_len]
                    out = self.decoder(tgt, memory)  # [1, seq_len, vocab_size]
                    log_probs = out[0, -1].log_softmax(dim=0)  # Last token probs
                    topk = torch.topk(log_probs, beam_width)
                    for i in range(beam_width):
                        next_word = topk.indices[i].item()
                        next_score = score + topk.values[i].item()
                        all_candidates.append((next_score, seq + [next_word]))
                # Keep top k beams, apply length penalty for diversity
                sequences = sorted(all_candidates, key=lambda tup: tup[0] / (len(tup[1]) ** length_penalty), reverse=True)[:beam_width]
                # Optionally, stop early if all end with EOS
                if all(seq[-1] == word2idx['<EOS>'] for _, seq in sequences):
                    break
            
            # Choose the best sequence (highest score)
            best_seq = sequences[0][1]
            # Convert token indices to words, skip <SOS> and <EOS>
            caption = [idx2word[idx] for idx in best_seq[1:-1] if idx in idx2word]
            return ' '.join(caption)

    