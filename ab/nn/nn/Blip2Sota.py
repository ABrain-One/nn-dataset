import torch
import torch.nn as nn
from transformers import Blip2ForConditionalGeneration

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape[0])
        
        # Load SOTA Model
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )
        
        # 1. FREEZE BACKBONE (Mandatory Requirement)
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
        for param in self.model.language_model.parameters():
            param.requires_grad = False
            
        # 2. MODIFICATION: Gated Attention Logic
        # Adding a small trainable layer and a gate factor
        self.gate = nn.Parameter(torch.zeros(1).to(torch.float16))
        self.modifier = nn.Linear(2560, 2560).to(torch.float16, device) # 2560 is OPT 2.7B hidden size
        
        self.model.to(self.device)

    def forward(self, images, captions=None):
        if captions is not None:
            # Training Mode with real gradients
            outputs = self.model(pixel_values=images.to(torch.float16), labels=captions, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            # Architectural change: Gate original features with modified ones
            modified = hidden + self.gate * self.modifier(hidden)
            return self.model.language_model.lm_head(modified).float()
        else:
            # Inference for Evaluation
            generated_ids = self.model.generate(pixel_values=images.to(torch.float16), max_new_tokens=30)
            return self._to_logits(generated_ids)

    def _to_logits(self, ids):
        # Framework requires logits, mapping generated IDs to one-hot-like logits
        B, T = ids.shape
        logits = torch.zeros(B, T, self.vocab_size, device=self.device)
        for b in range(B):
            for t in range(T):
                if ids[b, t] < self.vocab_size:
                    logits[b, t, ids[b, t]] = 1.0
        return logits

    def train_setup(self, prm):
        # Only optimizing our gate and modifier layer
        self.optimizer = torch.optim.AdamW([self.gate, *self.modifier.parameters()], lr=prm.get('lr', 1e-4))
        self.criteria = (nn.CrossEntropyLoss(ignore_index=-100),)

    def learn(self, train_data):
        self.model.train()
        total_loss = 0
        for images, captions in train_data:
            self.optimizer.zero_grad()
            logits = self.forward(images.to(self.device), captions.to(self.device))
            loss = self.criteria[0](logits[:, :-1, :].reshape(-1, self.vocab_size), captions[:, 1:].reshape(-1))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_data)