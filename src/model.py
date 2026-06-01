import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


def make_norm(norm, num_channels):
    if norm == 'batch':
        return nn.BatchNorm2d(num_channels)
    elif norm == 'instance':
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == 'group':
        num_groups = min(32, num_channels)
        while num_channels % num_groups != 0:
            num_groups //= 2
        return nn.GroupNorm(num_groups, num_channels)
    else:
        raise ValueError(f"Unknown norm: {norm!r}. Choose 'batch', 'group', or 'instance'.")


class ResidualConvBlock(nn.Module):
    """Two conv layers with a residual (identity) shortcut."""

    def __init__(self, in_ch, out_ch, dropout=0.0, norm='batch'):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            make_norm(norm, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            make_norm(norm, out_ch),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.relu(self.block(x) + self.shortcut(x)))


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=4, base_features=64, norm='batch'):
        """Configurable U-Net encoder-decoder architecture.

        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB).
            out_channels: Number of output channels (1 for binary, N for multi-class).
            depth: Number of encoder levels (2-6). Controls model capacity.
            base_features: Number of features in the first layer. Doubles each level.
            norm: Normalization layer — 'batch', 'group', or 'instance'.
                  'group' is recommended for small batches or cross-domain data.
        """
        super(UNet, self).__init__()
        self.depth = depth
        self.out_channels = out_channels

        def conv_block(in_ch, out_ch, dropout=0.0):
            return ResidualConvBlock(in_ch, out_ch, dropout, norm=norm)

        # Encoder
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        ch = in_channels
        for i in range(depth):
            features = base_features * (2 ** i)
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
            if x.shape != skip.shape:
                x = TF.center_crop(x, [skip.shape[2], skip.shape[3]])
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.final(x)
