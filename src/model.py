import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=4, base_features=64):
        """Configurable U-Net encoder-decoder architecture.

        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB).
            out_channels: Number of output channels (1 for binary segmentation).
            depth: Number of encoder levels (2-6). Controls model capacity.
            base_features: Number of features in the first layer. Doubles each level.
        """
        super(UNet, self).__init__()
        self.depth = depth

        def conv_block(in_ch, out_ch, dropout=0.0):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)

        # Encoder
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        ch = in_channels
        for i in range(depth):
            features = base_features * (2 ** i)
            # Dropout in bottleneck (last encoder) and second-deepest level
            if i == depth - 1:
                drop = 0.2
            elif i == depth - 2:
                drop = 0.1
            else:
                drop = 0.0
            self.encoders.append(conv_block(ch, features, dropout=drop))
            ch = features

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            features = base_features * (2 ** i)
            self.upconvs.append(nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2))
            # Dropout in first decoder level (closest to bottleneck)
            drop = 0.1 if i == depth - 2 else 0.0
            self.decoders.append(conv_block(features * 2, features, dropout=drop))

        self.final = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc_features = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            enc_features.append(x)
            if i < self.depth - 1:
                x = self.pool(x)

        # Decoder path
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = enc_features[self.depth - 2 - i]
            # Safe crop for odd dimensions
            if x.shape != skip.shape:
                x = TF.center_crop(x, [skip.shape[2], skip.shape[3]])
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.final(x)
