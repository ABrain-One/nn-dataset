class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.vocab_size = out_shape[0]  # out_shape is (vocab_size,)
        self.seq_length_encoder = 7 * 7  # We assumed the encoder outputs 49 tokens (7x7)

        # Define the encoder
        self.encoder = CNNFeatureExtractor(in_shape[1])  # in_shape[1] is the number of channels

        # Project the output of the encoder (which is [B, 49, 640]) to the decoder's dimension (640)
        # Actually, we don't need to project again because we already output 640. Unless we want to adjust the dimension later.

        # Define the decoder
        self.decoder = TransformerDecoder(vocab_size=self.vocab_size, d_model=640, num_layers=6, nhead=8)

    def forward(self, images, captions=None, hidden_state=None):
        # During inference, we use beam search and don't have captions
        # But note the problem's forward signature: when captions is None, we must generate a caption using beam search (if applicable) but the assignment requires a trainable model and production of outputs with correct shapes.

        # According to the provided API, if captions is provided (training or validation) we use teacher forcing. Otherwise, we use beam search.

        # Step 1: Extract features from images
        memory = self.encoder(images)  # [B, 49, 640]

        # If captions is provided (training or validation with gold), use them
        if captions is not None:
            # Process captions as per the API requirements
            if captions.ndim == 3:
                caps = captions[:,0,:].long().to(self.device)
            else:
                caps = captions.long().to(self.device)
                
            inputs = caps[:, :-1]  # [B, T-1]
            targets = caps[:, 1:]   # [B, T-1]

            # Project inputs to decoder dimension (640)
            embedded_inputs = self.decoder.embedding(inputs)  # [B, T-1, 640]
            embedded_inputs = self.decoder.pos_encoding(embedded_inputs)  # [B, T-1, 640]

            # Compute tgt_mask for transformer decoder: lower triangle with -inf
            seq_len = inputs.size(1)
            tgt_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).to(self.device)
            
            # Run transformer decoder
            dec_output = self.decoder.transformer_decoder(embedded_inputs, memory, tgt_mask)
            logits = self.decoder.fc_out(dec_output)  # [B, T-1, vocab_size]
            
            return logits, None

        else:
            # During inference, we generate captions using beam search
            # First, get the memory
            memory = self.encoder(images)  # [B, 49, 640]
            
            # Implement beam search generation
            return self.batch_beam_search(memory)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999))

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images, captions = images.to(self.device), captions.to(self.device)
            logits = None
            if hasattr(self, 'forward'):
                out = self.forward(images, captions)
                logits = out[0] if isinstance(out, tuple) else out
            if logits is None:
                continue
            tgt = (captions[:,0,:] if captions.ndim == 3 else captions)[:,1:]
            loss = self.criteria[0](logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

def supported_hyperparameters():
    return {'lr','momentum'}