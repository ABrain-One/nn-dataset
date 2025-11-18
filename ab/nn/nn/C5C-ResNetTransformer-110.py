import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

def supported_hyperparameters():
    return {'lr','momentum'}


class CNN_Encoder(nn.Module):
    def __init__(self, input_channels=3, output_features=768):
        super(CNN_Encoder, self).__init__()
        self.layer1 = self._block(input_channels, 64)
        self.layer2 = self._block(64, 128)
        self.layer3 = self._block(128, 256)
        self.layer4 = self._block(256, 512)
        self.layer5 = self._block(512, output_features)
        self.relu = nn.ReLU()
        self.apply(self._init_weights)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def _init_weights(self, m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class GRUDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=768):
        super(GRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if type(m) == nn.Embedding:
            nn.init.normal_(m.weight, std=0.05)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif type(m) == nn.GRU:
            for name, param in m.named_parameters():
                if 'weight' in name or 'bias' in name:
                    nn.init.uniform_(param.data, -0.1, 0.1)

    def forward(self, inputs: torch.Tensor, hidden_state: Optional[torch.Tensor] = None, memory: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # inputs: [B, T] where T is the sequence length
        # hidden_state: [B, hidden_size] (initial hidden state)
        # memory: [B, 1, hidden_size] (encoder memory)
        
        embedded = self.embedding(inputs)  # [B, T, hidden_size]
        gru_output, gru_hidden = self.gru(embedded, hidden_state)  # [B, T, hidden_size]
        logits = self.fc_out(gru_output)  # [B, T, vocab_size]
        return logits, gru_hidden

class Net(nn.Module):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = CNN_Encoder(input_channels=3, output_features=768)
        self.decoder = GRUDecoder(vocab_size=hparams.vocab_size)
        self.hidden_dim = 768

    def train_setup(self, hparams):
        pass

    def learn(self, images, captions=None, hidden_state=None):
        # Convert images to required format
        images = images.to(self.device)
        
        # Convert captions to tensor if provided
        if captions is not None:
            captions = captions.to(self.device)
            captions = captions.long()
        
        # Get encoder memory
        memory = self.encoder(images)  # [B, 1, 768]
        
        # If captions are provided, use teacher forcing
        if captions is not None:
            # Get input sequence (all but last token)
            inputs = captions[:, :-1]  # [B, T-1]
            logits, hidden_state = self.decoder(inputs, hidden_state, memory)
            return logits, hidden_state
        
        # Otherwise, generate captions
        else:
            # Initialize hidden_state if not provided
            if hidden_state is None:
                hidden_state = torch.zeros((images.size(0), self.hidden_dim), device=self.device)
            
            # Start with SOS token and create input and target sequences
            sos_token = torch.ones((images.size(0), 1), dtype=torch.long, device=self.device) * hparams.sos_idx
            inputs = sos_token
            
            # Beam search parameters
            beam_size = hparams.beam_size
            max_length = hparams.max_length
            
            # Initialize beam search structure
            scores = torch.ones((images.size(0), beam_size), device=self.device) * -float('Inf')
            scores[:, 0] = 0  # Start with zero score for the first beam
            
            # Create a list to store complete hypotheses
            hypotheses = [[] for _ in range(images.size(0))]
            hypotheses[0].append((0, 0))  # (score, length)
            
            # For each time step, expand the beam
            for step in range(max_length):
                # Embed the input tokens
                embedded = self.decoder.embedding(inputs)  # [B, 1, hidden_size]
                
                # Expand memory to match beam size
                memory_expanded = memory.expand(images.size(0), beam_size, -1)
                
                # Condition the embedded input with memory
                combined = torch.cat([embedded, memory_expanded], dim=-1)
                
                # Run the GRU
                gru_output, gru_hidden = self.gru(combined, hidden_state)
                
                # Project to vocabulary
                logits = self.fc_out(gru_output)  # [B, 1, vocab_size]
                
                # Select top k predictions for each image in the beam
                probs = torch.softmax(logits, dim=-1)
                topk_probs, topk_indices = probs.topk(beam_size, dim=-1)
                
                # Expand scores to match beam size
                scores_expanded = scores.unsqueeze(1).expand(images.size(0), topk_indices.size(1), beam_size)
                
                # Combine scores and new probabilities
                new_scores = scores_expanded + topk_probs
                
                # Update scores and indices
                scores = new_scores
                indices = topk_indices
                
                # Update hypotheses
                next_hypotheses = []
                for i in range(images.size(0)):
                    # For each image, extend the hypotheses with the new tokens
                    for score, idx in zip(new_scores[i], indices[i]):
                        if [idx.item()] not in hypotheses[i]:
                            hypotheses[i].append([idx.item()])
                            next_hypotheses.append((score.item(), len(hypotheses[i])))
                
                # Update scores and indices for the next step
                scores = torch.tensor([s[0] for s in next_hypotheses], device=self.device)
                indices = torch.tensor([i[1] for i in next_hypotheses], device=self.device)
                
                # Update inputs for the next step
                inputs = indices
            
            # After beam search, select the best hypothesis for each image
            best_idx = torch.argmax(scores, dim=1)
            best_tokens = torch.stack([torch.tensor(hyp[best_idx[i]], device=self.device) for i, hyp in enumerate(hypotheses)])
            
            # Convert to one-hot format
            logits = torch.zeros((images.size(0), max_length, hparams.vocab_size), device=self.device)
            for i, tokens in enumerate(best_tokens):
                for t, token in enumerate(tokens):
                    logits[i, t, token] = 1
            
            return logits, gru_hidden

    def forward(self, images, captions=None, hidden_state=None):
        # images: [B, C, H, W]
        # captions: [B, T] or [B, T, V] (optional)
        # hidden_state: [B, H] (optional)

        # Move images to device
        images = images.to(self.device)
        
        # Get encoder memory
        memory = self.encoder(images)  # [B, 1, 768]
        
        # If captions are provided, use teacher forcing
        if captions is not None:
            # Convert captions to tensor if not already
            if isinstance(captions, torch.Tensor):
                captions = captions.to(self.device)
                captions = captions.long()
            else:
                captions = torch.tensor(captions, dtype=torch.long, device=self.device)
            
            # Get input sequence (all but last token)
            inputs = captions[:, :-1]  # [B, T-1]
            
            # Run decoder
            logits, hidden_state = self.decoder(inputs, hidden_state, memory)
            
            # Assert shape
            assert logits.shape == inputs.shape
            assert logits.shape[-1] == self.decoder.vocab_size
            
            return logits, hidden_state
        
        # Otherwise, generate captions
        else:
            # Initialize hidden_state if not provided
            if hidden_state is None:
                hidden_state = torch.zeros((images.size(0), self.hidden_dim), device=self.device)
            
            # Start with SOS token and create input and target sequences
            sos_token = torch.ones((images.size(0), 1), dtype=torch.long, device=self.device) * 1  # SOS token index 1
            inputs = sos_token
            
            # For each time step, expand the beam
            max_length = 20  # Fixed maximum length for generation
            for step in range(max_length):
                # Embed the input tokens
                embedded = self.decoder.embedding(inputs)  # [B, 1, hidden_size]
                
                # Expand memory to match batch size
                memory_expanded = memory.expand(images.size(0), -1)
                
                # Condition the embedded input with memory
                combined = torch.cat([embedded, memory_expanded], dim=-1)
                
                # Run the GRU
                gru_output, gru_hidden = self.gru(combined, hidden_state)
                
                # Project to vocabulary
                logits = self.fc_out(gru_output)  # [B, 1, vocab_size]
                
                # Get probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Select next token (greedy)
                next_token = torch.argmax(probs, dim=-1)
                
                # Append to inputs
                inputs = torch.cat([inputs, next_token], dim=1)
            
            # Assert shape
            assert inputs.shape[1] == max_length
            assert inputs.shape[-1] == self.decoder.vocab_size
            
            return inputs, gru_hidden