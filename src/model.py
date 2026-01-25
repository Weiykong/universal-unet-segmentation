import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Helper: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),   # <--- THE MISSING INGREDIENT
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),   # <--- KEEPS WEIGHTS ALIVE
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(1, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up2(e3)
        
        # Safe Crop for Odd Dimensions
        if d2.shape != e2.shape:
            d2 = TF.center_crop(d2, [e2.shape[2], e2.shape[3]])
            
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        
        # Safe Crop for Odd Dimensions
        if d1.shape != e1.shape:
            d1 = TF.center_crop(d1, [e1.shape[2], e1.shape[3]])
            
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)