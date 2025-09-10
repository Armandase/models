import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingNet(nn.Module):
    """Petit réseau pour produire les poids de gating."""
    def __init__(self, in_channels, out_dim):
        super().__init__()
        # on réduit à un vecteur global via GAP
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, out_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        h = self.pool(x).view(x.size(0), -1)
        return F.softmax(self.fc(h), dim=1)  # (B, out_dim)


class HierarchicalMoEConv(nn.Module):
    """
    Convolution remplacée par un Hiérarchical MoE:
      - Top gating: M branches
      - Chaque branche: K conv experts + low gating
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 M=2, K=2):
        super().__init__()
        self.M = M
        self.K = K

        # Gating de premier niveau
        self.top_gating = GatingNet(in_channels, M)

        # Gating de second niveau
        self.low_gatings = nn.ModuleList([GatingNet(in_channels, K) for _ in range(M)])

        # Experts conv
        self.experts = nn.ModuleList()
        for i in range(M):
            branch_experts = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding)
                for _ in range(K)
            ])
            self.experts.append(branch_experts)

    def forward(self, x):
        B, _, H, W = x.shape
        top_w = self.top_gating(x)   # (B, M)

        outputs = torch.zeros(B, self.experts[0][0].out_channels, H // self.experts[0][0].stride[0],
                              W // self.experts[0][0].stride[0], device=x.device)

        for i in range(self.M):
            low_w = self.low_gatings[i](x)  # (B, K)
            for j in range(self.K):
                y = self.experts[i][j](x)  # (B, C, H’, W’)
                weight = (top_w[:, i] * low_w[:, j]).view(B, 1, 1, 1)
                outputs += weight * y

        return outputs

class ClassicMediumCNN_HME(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, img_size=32,
                 M=2, K=2):
        super().__init__()
        self.img_size = img_size

        self.features = nn.Sequential(
            HierarchicalMoEConv(input_channels, 64, kernel_size=3, padding=1, M=M, K=K),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            HierarchicalMoEConv(64, 128, kernel_size=3, padding=1, M=M, K=K),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            HierarchicalMoEConv(128, 256, kernel_size=3, padding=1, M=M, K=K),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        final_size = self.img_size // 8
        self.classifier = nn.Sequential(
            nn.Linear(256 * final_size * final_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x  # logits
