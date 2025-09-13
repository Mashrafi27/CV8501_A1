import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

# MRI encoder: torchvision 3D ResNet-18 (penultimate features)
def build_mri_backbone(pretrained=True):
    weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
    m = r3d_18(weights=weights)
    d = m.fc.in_features
    m.fc = nn.Identity()
    return m, d

# EHR encoder: try FT-Transformer, fallback to MLP (still NN)
class EHREncoderMLP(nn.Module):
    def __init__(self, in_dim, hidden=512, out_dim=192, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim), nn.ReLU(True),
        )
    def forward(self, x): return self.net(x)

def try_build_fttransformer(n_features, dropout=0.2):
    try:
        import rtdl
    except Exception:
        return None, None
    # make_default path
    try:
        enc = rtdl.FTTransformer.make_default(
            n_num_features=n_features, cat_cardinalities=[],
            d_token=192, n_heads=4, n_layers=2,
            attention_dropout=dropout, ffn_dropout=dropout, prenormalization=True
        )
        return enc, 192
    except Exception:
        pass
    # ctor paths
    for kwargs in [
        dict(n_num_features=n_features, cat_cardinalities=[],
             d_token=192, n_blocks=2, attention_n_heads=4,
             ffn_d_hidden=384, attention_dropout=dropout,
             ffn_dropout=dropout, residual_dropout=dropout, prenormalization=True),
        dict(n_num_features=n_features, cat_cardinalities=[], d_token=192, n_blocks=2),
    ]:
        try:
            enc = rtdl.FTTransformer(**kwargs)
            return enc, 192
        except Exception:
            continue
    return None, None

def build_ehr_encoder(n_num_features, dropout=0.2):
    enc, dim = try_build_fttransformer(n_num_features, dropout)
    if enc is not None:
        return enc, dim, "fttransformer"
    return EHREncoderMLP(n_num_features, hidden=512, out_dim=192, dropout=dropout), 192, "mlp"

class LateFusionModel(nn.Module):
    """
    Decision-level late fusion (end-to-end):
      - MRI branch: r3d_18 -> logits_mri
      - EHR branch: FT-Transformer/MLP -> logits_ehr
      - Fuse probs: p_fused = α p_mri + (1-α) p_ehr   (α is fixed, not learned)
      - Backprop through both encoders via fused loss
    """
    def __init__(self, n_ehr_features, n_classes=3, dropout=0.2, alpha=0.5, pretrained_mri=True):
        super().__init__()
        self.alpha = float(alpha)
        self.mri_backbone, d_mri = build_mri_backbone(pretrained=pretrained_mri)
        self.ehr_enc, d_ehr, self.ehr_kind = build_ehr_encoder(n_ehr_features, dropout=dropout)
        self.mri_head = nn.Linear(d_mri, n_classes)
        self.ehr_head = nn.Linear(d_ehr, n_classes)

    def forward_branch(self, x_mri=None, x_ehr=None, mode="fusion"):
        if mode == "mri":
            z = self.mri_backbone(x_mri)
            return self.mri_head(z)
        if mode == "ehr":
            z = self.ehr_enc(None, x_ehr) if self.ehr_kind == "fttransformer" else self.ehr_enc(x_ehr)
            return self.ehr_head(z)
        # fusion: return both logits
        z_mri = self.mri_backbone(x_mri)
        z_ehr = self.ehr_enc(None, x_ehr) if self.ehr_kind == "fttransformer" else self.ehr_enc(x_ehr)
        return self.mri_head(z_mri), self.ehr_head(z_ehr)

    def fuse_logits(self, logits_mri, logits_ehr):
        pm = torch.softmax(logits_mri, dim=1)
        pe = torch.softmax(logits_ehr, dim=1)
        pf = self.alpha * pm + (1.0 - self.alpha) * pe
        return torch.log(pf.clamp_min(1e-8))  # log-probs for NLLLoss
