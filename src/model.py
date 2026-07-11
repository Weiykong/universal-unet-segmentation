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

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // reduction), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class AttentionGate(nn.Module):
    """Attention Gate for skip connections in U-Net."""

    def __init__(self, F_g, F_l, F_int, norm='batch'):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            make_norm(norm, F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            make_norm(norm, F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha


class ResidualConvBlock(nn.Module):
    """Two conv layers with a residual (identity) shortcut."""

    def __init__(self, in_ch, out_ch, dropout=0.0, norm='batch', se_block=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            make_norm(norm, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            make_norm(norm, out_ch),
        )
        self.se = SqueezeExcitation(out_ch) if se_block else nn.Identity()
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        return self.drop(self.relu(out + self.shortcut(x)))


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=4, base_features=64, norm='batch', attention=False, se_block=False):
        """Configurable U-Net encoder-decoder architecture.

        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB).
            out_channels: Number of output channels (1 for binary, N for multi-class).
            depth: Number of encoder levels (2-6). Controls model capacity.
            base_features: Number of features in the first layer. Doubles each level.
            norm: Normalization layer — 'batch', 'group', or 'instance'.
                  'group' is recommended for small batches or cross-domain data.
            attention: Enable Attention Gates on skip connections.
            se_block: Enable Squeeze-and-Excitation blocks in residual units.
        """
        super(UNet, self).__init__()
        self.depth = depth
        self.out_channels = out_channels
        self.attention = attention
        self.se_block = se_block

        def conv_block(in_ch, out_ch, dropout=0.0, se_block=False):
            return ResidualConvBlock(in_ch, out_ch, dropout, norm=norm, se_block=se_block)

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
            self.encoders.append(conv_block(ch, features, dropout=drop, se_block=se_block))
            ch = features

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attention_gates = nn.ModuleList() if attention else None
        for i in range(depth - 2, -1, -1):
            features = base_features * (2 ** i)
            self.upconvs.append(nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2))
            if attention:
                self.attention_gates.append(
                    AttentionGate(F_g=features, F_l=features, F_int=max(1, features // 2), norm=norm)
                )
            drop = 0.1 if i == depth - 2 else 0.0
            self.decoders.append(conv_block(features * 2, features, dropout=drop, se_block=se_block))

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
            if self.attention:
                skip = self.attention_gates[i](x, skip)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.final(x)


class LegacyUNet(nn.Module):
    """Pre-residual UNet for loading checkpoints saved before the residual refactor.
    Matches the original plain nn.Sequential conv_block architecture exactly."""

    def __init__(self, in_channels=1, out_channels=1, depth=4, base_features=64):
        super().__init__()
        self.depth = depth
        self.out_channels = out_channels

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

        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        ch = in_channels
        for i in range(depth):
            features = base_features * (2 ** i)
            drop = 0.2 if i == depth - 1 else (0.1 if i == depth - 2 else 0.0)
            self.encoders.append(conv_block(ch, features, dropout=drop))
            ch = features

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            features = base_features * (2 ** i)
            self.upconvs.append(nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2))
            drop = 0.1 if i == depth - 2 else 0.0
            self.decoders.append(conv_block(features * 2, features, dropout=drop))

        self.final = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        enc_features = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            enc_features.append(x)
            if i < self.depth - 1:
                x = self.pool(x)

        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = enc_features[self.depth - 2 - i]
            if x.shape != skip.shape:
                x = TF.center_crop(x, [skip.shape[2], skip.shape[3]])
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.final(x)
