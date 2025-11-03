class EncoderCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=640):
        super().__init__()
        self.out_channels = out_channels
        
        # Four downsampling layers
        self.stage0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.stage3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        self.final_proj = nn.Linear(512 * 14 * 14, 640)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        # Now x has shape [B, 512, 14, 14]
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.out_channels)
        # But note: we specified out_channels=640, but currently we have 512.
        # Wait, no: in the final_proj, we are projecting to 640.
        # Correction: I intended to have the channel dimension be 512 initially, but then we want to project to 640.

        # Actually, let's change: the last stage outputs 512 channels, then we flatten and project to 640.
        # But wait, the sequence would then be [B, 196, 640].

        # However, we can also consider that the hidden state of the decoder might be connected differently.

        # But the problem requires [B, S, H] with H>=640.

        # Let's recompute: after stage3, we have 512 channels and 14x14 spatial, so we need to project the 512 to 640.

        # But why not have the last stage output 640 channels?

        # Change the last convolution to output 640 channels and then do not need the final_proj if we set out_channels=640.

        # Alternatively, let's redesign the stages to output 640 channels at the last stage.

        # Let's set the initial channel to 3, then after stage0: 64, stage1: 128, stage2: 256, stage3: 512.
        # And then we project 512 to 640.

        # But then the memory dimension is 640, which is okay.

        # So:

        # After stage3: [B, 512, 14, 14] -> we flatten the last two dimensions and then multiply by 640.

        # However, that would require a matrix multiplication of 512*196 (input) and 640 output.

        # Let's do:

        #   x = x.view(x.size(0), -1)   # [B, 512*196]
        #   x = self.final_proj(x)       # [B, 512*196, 640]

        # But that reshapes to [B, 512*196, 640] which is not the desired sequence format.

        # We want [B, S, H] = [B, 14*14, 640] ?

        # The original idea was to have a fixed hidden dimension for the decoder. The decoder must attend over S tokens.

        # In the encoder, if we have a 14x14 grid, that means 196 tokens, each with 640 dimensions.

        # How do we project the 512-channel per position to 640?

        # We can do:

        #   x = self.final_proj(x.view(x.size(0), -1, 512))
        #   then permute to [B, 196, 640]

        # Let's define:

        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        # Now x has shape [B, 512, 14, 14]
        # Calculate the number of spatial locations: 14*14=196
        s = 14 * 14
        x = x.view(x.size(0), s, -1)  # [B, s, 512]
        # Project the channel dimension from 512 to 640
        x = self.final_proj(x.permute(0, 2, 1))  # [B, 512, s]
        x = x.permute(0, 2, 1)  # [B, s, 640]

        return x

def supported_hyperparameters():
    return {'lr','momentum'}
