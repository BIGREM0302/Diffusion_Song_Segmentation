import torch
from torch import nn

class UnsupAutoregEncoder(nn.Module):
    def __init__(self, img_h=204, img_w=128, input_channels=2, seq_len=17, hidden_dim=256):
        super().__init__()
        mid_channels = 20
        output_channels = 2

        self.img_h = img_h
        self.img_w = img_w
        self.output_channels = output_channels
        self.seq_len = seq_len

        # 卷積序列
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, mid_channels, 3, padding=1),
            nn.SiLU(),
            nn.LayerNorm([mid_channels, img_h, img_w]),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.SiLU(),
            nn.LayerNorm([mid_channels, img_h, img_w]),
            nn.Conv2d(mid_channels, output_channels, 3, padding=1),
        )

        # 使用 dummy tensor 自動計算 linear in_features
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, img_h, img_w)
            dummy = self.layers(dummy)  # (1, output_channels, H, W)
            dummy = dummy.reshape(1, output_channels, seq_len, -1).permute(0, 2, 1, 3).reshape(1, seq_len, -1)
            in_features = dummy.size(-1)

        self.squeeze = nn.Linear(in_features, hidden_dim)
        self.pos_enc = nn.Embedding(seq_len, hidden_dim)

    def forward(self, x):
        bs = x.size(0)
        x = self.layers(x)  # (bs, output_channels, H, W)
        # reshape 成 (batch, seq_len, feature_dim)
        x = x.reshape(bs, self.output_channels, self.seq_len, -1).permute(0, 2, 1, 3).reshape(bs, self.seq_len, -1)
        x = self.squeeze(x)

        pos = torch.arange(0, self.seq_len, device=x.device).long()
        pos_emb = self.pos_enc(pos).unsqueeze(0)
        x += pos_emb
        return x


class CtpAutoregEncoder(nn.Module):

    def __init__(self, img_h=108, img_w=128):
        super().__init__()
        input_channels = 10
        mid_channels = 20
        output_channels = 2
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, mid_channels, 3, padding=1),
            nn.SiLU(),
            nn.LayerNorm([mid_channels, img_h, img_w]),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.SiLU(),
            nn.LayerNorm([mid_channels, img_h, img_w]),
            nn.Conv2d(mid_channels, output_channels, 3, padding=1),
        )
        self.img_h = img_h
        self.img_w = img_w
        self.squeeze = nn.Linear(1024, 128)
        self.output_channels = output_channels
        self.pos_enc = nn.Embedding(27, 128)

    def forward(self, x):
        bs = x.size(0)
        x = self.layers(x)
        x = x.reshape(bs, self.output_channels, 27, 4 * 128).permute(0, 2, 1, 3).reshape(bs, 27, -1)
        x = self.squeeze(x)

        pos = torch.arange(0, 27).to(x.device).long()
        pos_emb = self.pos_enc(pos).unsqueeze(0)
        x += pos_emb
        return x


class LshAutoregEncoder(nn.Module):

    def __init__(self, img_h=136, img_w=128):
        super().__init__()
        input_channels = 12
        mid_channels = 20
        output_channels = 2
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, mid_channels, 3, padding=1),
            nn.SiLU(),
            nn.LayerNorm([mid_channels, img_h, img_w]),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.SiLU(),
            nn.LayerNorm([mid_channels, img_h, img_w]),
            nn.Conv2d(mid_channels, output_channels, 3, padding=1),
        )
        self.img_h = img_h
        self.img_w = img_w
        self.squeeze = nn.Linear(2048, 256)
        self.output_channels = output_channels
        self.pos_enc = nn.Embedding(17, 256)

    def forward(self, x):
        bs = x.size(0)
        x = self.layers(x)
        x = x.reshape(bs, self.output_channels, 17, 8 * 128).permute(0, 2, 1, 3).reshape(bs, 17, -1)
        x = self.squeeze(x)

        pos = torch.arange(0, 17).to(x.device).long()
        pos_emb = self.pos_enc(pos).unsqueeze(0)
        x += pos_emb
        return x


class AccAutoregEncoder(nn.Module):

    def __init__(self, img_h=136, img_w=128):
        super().__init__()
        input_channels = 14
        mid_channels = 20
        output_channels = 2
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, mid_channels, 3, padding=1),
            nn.SiLU(),
            nn.LayerNorm([mid_channels, img_h, img_w]),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.SiLU(),
            nn.LayerNorm([mid_channels, img_h, img_w]),
            nn.Conv2d(mid_channels, output_channels, 3, padding=1),
        )
        self.img_h = img_h
        self.img_w = img_w
        self.squeeze = nn.Linear(2048, 256)
        self.output_channels = output_channels
        self.pos_enc = nn.Embedding(17, 256)

    def forward(self, x):
        bs = x.size(0)
        x = self.layers(x)[:, :, 0: 200]
        x = x.reshape(bs, self.output_channels, 17, 8 * 128).permute(0, 2, 1, 3).reshape(bs, 17, -1)
        x = self.squeeze(x)

        pos = torch.arange(0, 17).to(x.device).long()
        pos_emb = self.pos_enc(pos).unsqueeze(0)
        x += pos_emb
        return x
