import torch, torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

# ---------------- MRI backbone (built-in) ----------------
def build_mri_backbone(pretrained=True):
    weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
    m = r3d_18(weights=weights)
    d_mri = m.fc.in_features
    m.fc = nn.Identity()  # expose embeddings
    return m, d_mri

# ---------------- EHR encoder (robust: FT-Transformer -> fallback MLP) ----------------
class EHREncoderMLP(nn.Module):
    def __init__(self, in_dim, hidden=512, out_dim=192, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):  # x: (N, D_ehr)
        return self.net(x)

def try_build_fttransformer(n_features, dropout=0.2):
    """
    Try several RTDL FTTransformer signatures.
    Returns (module, out_dim) or (None, None) if not possible.
    """
    try:
        import rtdl  # package is deprecated; versions vary
    except Exception:
        return None, None

    # Try v0.0.13-style: make_default with d_token
    try:
        enc = rtdl.FTTransformer.make_default(
            n_num_features=n_features,
            cat_cardinalities=[],
            d_token=192,
            n_heads=4,
            n_layers=2,
            attention_dropout=dropout,
            ffn_dropout=dropout,
            prenormalization=True
        )
        return enc, 192
    except TypeError:
        pass
    except Exception:
        pass

    # Try constructor-style (other releases):
    for kwargs in [
        dict(  # common alt signature
            n_num_features=n_features, cat_cardinalities=[],
            d_token=192, n_blocks=2, attention_n_heads=4,
            ffn_d_hidden=384, attention_dropout=dropout,
            ffn_dropout=dropout, residual_dropout=dropout,
            prenormalization=True
        ),
        dict(  # ultra-minimal fallback (older forks)
            n_num_features=n_features, cat_cardinalities=[], d_token=192, n_blocks=2
        )
    ]:
        try:
            enc = rtdl.FTTransformer(**kwargs)
            return enc, 192
        except TypeError:
            continue
        except Exception:
            continue

    return None, None

def build_ehr_encoder(n_num_features, dropout=0.2):
    enc, d = try_build_fttransformer(n_num_features, dropout=dropout)
    if enc is not None:
        return enc, d, "fttransformer"
    # fallback: simple, solid MLP (still a neural network)
    return EHREncoderMLP(n_num_features, hidden=512, out_dim=192, dropout=dropout), 192, "mlp"

# ---------------- Fusion head ----------------
class FusionHead(nn.Module):
    def __init__(self, in_dim, hidden=512, dropout=0.2, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, z): return self.net(z)

# ---------------- Hybrid model ----------------
class HybridModel(nn.Module):
    """
    MRI: r3d_18 penultimate embeddings (expects 3 x T x H x W).
    EHR: FT-Transformer if available & compatible; otherwise MLP.
    Fusion: concat -> MLP head -> logits.
    """
    def __init__(self, n_ehr_features, pretrained_mri=True, n_classes=3, dropout=0.2):
        super().__init__()
        self.mri_backbone, d_mri = build_mri_backbone(pretrained=pretrained_mri)
        self.ehr_enc, d_ehr, self.ehr_kind = build_ehr_encoder(n_ehr_features, dropout=dropout)
        self.head = FusionHead(d_mri + d_ehr, hidden=512, dropout=dropout, n_classes=n_classes)

    def forward(self, x_mri, x_ehr):
        z_mri = self.mri_backbone(x_mri)   # (N, 512)
        # FTTransformer API: forward(x_cat=None, x_num=...)
        if hasattr(self.ehr_enc, "forward") and self.ehr_kind == "fttransformer":
            z_ehr = self.ehr_enc(None, x_ehr)
        else:
            z_ehr = self.ehr_enc(x_ehr)
        z = torch.cat([z_mri, z_ehr], dim=1)
        return self.head(z)
