import torch
import torch.nn as nn
import timm  # Using timm to load a supported model
from torchvision.models.feature_extraction import create_feature_extractor


# 🧠 Updated DKDC layer (no depthwise error)
class DKDC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DKDC, self).__init__()
        self.k3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.k5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x3 = self.k3(x)
        x5 = self.k5(x)
        out = torch.cat([x3, x5], dim=1)
        return self.relu(self.bn(out))


# 🌀 Capsule-like Feature Layer (simulated)
class CapsuleLikeBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(CapsuleLikeBlock, self).__init__()
        self.capsule = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.LayerNorm(out_features)
        )

    def forward(self, x):
        return self.capsule(x)


# 🏥 Final Lung Classification Model
class LungClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(LungClassifier, self).__init__()

        # Load GhostNetV2_100 model from timm
        backbone = timm.create_model('ghostnetv2_100', pretrained=True, features_only=True)
        
        # Use final output feature channel size
        feature_info = backbone.feature_info
        final_feat_channels = feature_info[-1]['num_chs']  # should be 960 for ghostnetv2_100

        self.backbone = backbone

        # Step 2: DKDC block
        self.dkdc = DKDC(in_channels=final_feat_channels, out_channels=64)  # 960 -> DKDC -> 64 x 2 = 128

        # Step 3: Capsule-like transformation
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.capsule_like = CapsuleLikeBlock(128, 64)  # DKDC output channels = 64 * 2 = 128

        # Final classification layer
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        # Extract features from the backbone
        features = self.backbone(x)
        x = features[-1]  # Last stage output, [B, 960, H, W]

        # Pass through DKDC layer
        x = self.dkdc(x)

        # Pooling
        x = self.pool(x).squeeze(-1).squeeze(-1)  # [B, C, 1, 1] -> [B, C]

        # Capsule-like transformation
        x = self.capsule_like(x)

        # Final classification
        return self.classifier(x)
