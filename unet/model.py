import torch

import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_ch=32, layer_num=4):
        super().__init__()
        self.layer_num = layer_num

        # Encoder
        self.downs = nn.ModuleList()
        chs = [in_channels] + [base_ch * (2 ** i) for i in range(layer_num)]
        for i in range(layer_num):
            self.downs.append(DoubleConv(chs[i], chs[i+1]))

        self.pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(layer_num)])

        # Bottleneck
        self.bottleneck = DoubleConv(chs[-1], chs[-1]*2)

        # Decoder
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(layer_num-1, -1, -1):
            self.ups.append(nn.ConvTranspose2d(chs[i+1]*2 if i==layer_num-1 else chs[i+2], chs[i+1], 2, stride=2))
            self.up_convs.append(DoubleConv(chs[i+1]*2, chs[i+1]))

        # Final output
        self.final_conv = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x):
        enc_feats = []
        input_size = x.size()[2:]  # (H, W)
        for i in range(self.layer_num):
            x = self.downs[i](x)
            enc_feats.append(x)
            x = self.pools[i](x)

        x = self.bottleneck(x)

        for i in range(self.layer_num-1, -1, -1):
            x = self.ups[self.layer_num-1-i](x)
            enc_feat = enc_feats[i]
            # Crop encoder features to match upsampled size
            if x.size()[2:] != enc_feat.size()[2:]:
                _, _, h, w = x.size()
                enc_feat = enc_feat[:, :, :h, :w]
            x = torch.cat([x, enc_feat], dim=1)
            x = self.up_convs[self.layer_num-1-i](x)

        x = self.final_conv(x)
        # Resize output to match input spatial size if needed
        if x.size()[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=1, base_ch=32, layer_num=4)
    x = torch.randn(1, 1, 256, 256)  # Example input
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be (1, 1, 256, 256) if input is (1, 1, 256, 256)
    print(model)  # Print model architecture
