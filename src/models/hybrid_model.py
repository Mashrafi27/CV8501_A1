import torch, torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
import rtdl

def build_mri_backbone(pretrained=True):
    weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
    m = r3d_18(weights=weights)
    d_mri = m.fc.in_features
    m.fc = nn.Identity()  # use embeddings
    return m, d_mri

def build_ehr_encoder(n_num_features, d_token=192, n_layers=2, n_heads=4, d_ff=384, dropout=0.2):
    enc = rtdl.FTTransformer.make_default(
        n_num_features=n_num_features, cat_cardinalities=[],
        d_token=d_token, n_heads=n_heads, n_layers=n_layers,
        attention_dropout=dropout, ffn_dropout=dropout,
        prenormalization=True
    )
    return enc, d_token

class FusionHead(nn.Module):
    def __init__(self, in_dim, hidden=512, dropout=0.2, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, z): return self.net(z)

class HybridModel(nn.Module):
    def __init__(self, n_ehr_features, pretrained_mri=True, n_classes=3, dropout=0.2):
        super().__init__()
        self.mri_backbone, d_mri = build_mri_backbone(pretrained_mri)
        self.ehr_enc, d_ehr = build_ehr_encoder(n_ehr_features, dropout=dropout)
        self.head = FusionHead(d_mri + d_ehr, hidden=512, dropout=dropout, n_classes=n_classes)

    def forward(self, x_mri, x_ehr):
        z_mri = self.mri_backbone(x_mri)               # (N,512)
        z_ehr = self.ehr_enc(None, x_ehr)              # (N,192)
        z = torch.cat([z_mri, z_ehr], dim=1)
        return self.head(z)
