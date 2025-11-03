class Encoder(nn.Module):
        def __init__(self, in_shape, hidden_dim=768):
            super().__init__()
            c_in = in_shape[1]  # First element of in_shape is the number of channels
            self.num_classes = hidden_dim  # But wait, we are not for classification, so we remove the linear head.

            self.features = nn.Sequential()
            # First layer definition
            self.features.add_module('initial', self._block(c_in, 64, kernel_size=5, padding=2, strides=2))
            # Then for each stage
            in_channels = 64
            for i, stage_channels in enumerate(self.channels):
                stage = self._stage(in_channels, stage_channels)
                self.features.add_module(f'stage_{i}', stage)
                in_channels = stage_channels[-1]   # After last layer of the stage, the output channels

            self.features.add_module("final_pool", nn.AdaptiveAvgPool2d((1,1)))

        def _block(self, in_channels, out_channels, kernel_size, padding, strides=2):
            # like BagNetUnit but simpler: it has a convolution and then a batchnorm and activation
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=strides, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def _stage(self, in_channels, stage_channels):
            # Apply three blocks consecutively
            layers = []
            for out_ch in stage_channels:
                layers.append(self._block(in_channels, out_ch, kernel_size=3, padding=1, strides=1))
                in_channels = out_ch
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.features(x)
            x = x.transpose(1,2)   # The BagNetUnit example did a transpose to have [B, H, 1, 1] ? No, let's look.

            # Actually, the BagNetUnit example had:
            #   self.features = nn.Sequential( ... )
            #   self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(1))
            #   self.output = nn.Linear(in_channels, num_classes)

            # We removed the output linear, but we kept the final_pool.

            # After final_pool, we have [B, in_channels, 1, 1] -> then we can reshape to [B, 1, in_channels] ?

            # But our hidden_dim is 768, and we want [B, 1, hidden_dim].

            # Let's change the final block to be a projection.

            # We didn't have a projection layer, so we need to add one.

            # Revised plan: Remove the final pool and then add a linear projection.

            # We want to output a single vector per image. So we remove the final_pool and add a flatten and a linear projection.

            # Let's change the encoder to not have the final pool and then project.

            # But then the forward without final pool:

            #   x = self.features(x)   # [B, in_channels, H_final, W_final]
            #   x = x.flatten(start_dim=1)   # [B, in_channels * (H_final * W_final)]

            # This might be large and not matching our hidden_dim.

def supported_hyperparameters():
    return {'lr','momentum'}
