import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

def supported_hyperparameters():
    # Only expose what your pipeline can set (all [0, 1])
    return {'lr', 'momentum', 'dropout', 'tie_weights'}

class MultiScaleConvNeXtEncoder(nn.Module):
    def __init__(self, in_channels=3, out_dim=512, pretrained=True):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)
        self.stem = backbone.features[0]
        self.stages = nn.ModuleList(backbone.features[1:])
        self.proj2 = nn.Conv2d(384, out_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(768, out_dim, kernel_size=1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.stem(x)
        feat_2 = None
        feat_3 = None
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            #print(f"[DEBUG] After stage {idx}: {x.shape}")
            if x.shape[1] == 384 and feat_2 is None:
                feat_2 = x
            if x.shape[1] == 768 and feat_3 is None:
                feat_3 = x
        assert feat_2 is not None, f"[ERROR] feat_2 (384 channels) not found! Last feature shape: {x.shape}"
        assert feat_3 is not None, f"[ERROR] feat_3 (768 channels) not found! Last feature shape: {x.shape}"
        #print(f"[DEBUG] Selected feat_2 shape: {feat_2.shape}")
        #print(f"[DEBUG] Selected feat_3 shape: {feat_3.shape}")
        tokens2 = self.proj2(feat_2).flatten(2).transpose(1, 2)
        tokens3 = self.proj3(feat_3).flatten(2).transpose(1, 2)
        #print(f"[DEBUG] tokens2 shape: {tokens2.shape}")
        #print(f"[DEBUG] tokens3 shape: {tokens3.shape}")
        all_tokens = torch.cat([tokens2, tokens3], dim=1)
        #print(f"[DEBUG] Concatenated tokens shape: {all_tokens.shape}")
        all_tokens = self.norm(all_tokens)
        #print(f"[DEBUG] Final output after LayerNorm: {all_tokens.shape}")
        return all_tokens

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs)

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=4, nhead=8, dim_feedforward=2048, dropout=0.1, tie_weights=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        if tie_weights:
            self.fc_out.weight = self.embedding.weight
        self.d_model = d_model

    def forward(self, captions, encoder_outputs, tgt_mask=None, memory_mask=None):
        x = self.embedding(captions) * (self.d_model ** 0.5)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(
                x,
                encoder_outputs,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask
            )
        logits = self.fc_out(x)
        return logits

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape

        vocab_size = out_shape[0]
        d_model = 512
        num_layers = 4
        nhead = 8
        dropout = prm.get('dropout', 0.1)
        tie_weights = bool(int(prm.get('tie_weights', 1)))

        self.encoder = MultiScaleConvNeXtEncoder(
            in_channels=in_shape[1],
            out_dim=d_model,
            pretrained=True
        )
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
            tie_weights=tie_weights
        )

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        lr = float(prm.get('lr', 0.001)) * 0.01 + 1e-4
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def learn(self, train_data):
        self.train()
        print_every = 500
        running_loss = 0.0
        batch_count = 0

        # We'll print a few predictions from the first batch of the epoch:
        sample_printed = False

        # If class variables exist, use for decoding captions:
        idx2word = getattr(self, 'idx2word', None)

        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)
            captions = captions[:, 0, :]
            input_ids = captions[:, :-1]
            target = captions[:, 1:]
            encoder_outputs = self.encoder(images)
            logits = self.decoder(input_ids, encoder_outputs)
            logits = logits.reshape(-1, logits.shape[-1])
            target = target.reshape(-1)
            loss = self.criteria[0](logits, target)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

            running_loss += loss.item()
            batch_count += 1

            if batch_count % print_every == 0:
                avg_loss = running_loss / print_every
                print(f"[TRAIN] Batch {batch_count} - Avg Loss (last {print_every}): {avg_loss:.4f}")
                running_loss = 0.0

            # Print sample captions for the first batch only
            if not sample_printed and idx2word is not None:
                self.eval()
                with torch.no_grad():
                    preds = self.forward(images)  # [B, max_len]
                for i in range(min(3, preds.size(0))):
                    tokens = preds[i].cpu().tolist()
                    caption = []
                    for idx in tokens:
                        word = idx2word.get(idx, '<UNK>')
                        if word == '<EOS>':
                            break
                        if word not in ['<PAD>', '<SOS>']:
                            caption.append(word)
                    print(f"Sample prediction {i+1}: {' '.join(caption)}")
                self.train()
                sample_printed = True

        # Print last partial batch loss
        if batch_count % print_every != 0 and batch_count > 0:
            avg_loss = running_loss / (batch_count % print_every)
            print(f"[TRAIN] Batch {batch_count} - Avg Loss (last {batch_count % print_every}): {avg_loss:.4f}")


    def forward(self, images, captions=None, hidden_state=None, max_len=20):
        self.eval()
        with torch.no_grad():
            encoder_outputs = self.encoder(images)
            B = images.size(0)
            device = images.device
            word2idx = getattr(self, 'word2idx', None)
            idx2word = getattr(self, 'idx2word', None)
            if captions is not None:
                input_ids = captions[:, :-1]
                logits = self.decoder(input_ids, encoder_outputs)
                return logits
            start_token = word2idx['<SOS>'] if word2idx else 1
            end_token = word2idx['<EOS>'] if word2idx else 2
            inputs = torch.full((B, 1), start_token, dtype=torch.long, device=device)
            outputs = []
            for _ in range(max_len):
                logits = self.decoder(inputs, encoder_outputs)
                next_token = logits[:, -1, :].argmax(-1, keepdim=True)
                outputs.append(next_token)
                inputs = torch.cat([inputs, next_token], dim=1)
            outputs = torch.cat(outputs, dim=1)
            return outputs
