import numpy as np, torch
from torch.utils.data import Dataset

class MMHybridDataset(Dataset):
    """
    Hybrid multimodal dataset: loads MRI .npy + EHR features per subject.
    """
    def __init__(self, df, ehr_cols, target_shape=(64,112,112)):
        self.df = df.reset_index(drop=True)
        self.ehr_cols = ehr_cols
        self.tgt = target_shape

    def __len__(self): return len(self.df)

    def _resize_3d(self, vol):
        v = torch.from_numpy(vol[None, None, ...])   # (1,1,Z,Y,X)
        v = torch.nn.functional.interpolate(v, size=self.tgt, mode="trilinear", align_corners=False)
        return v[0]  # (1,T,H,W)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        vol = np.load(r["std_path"]).astype(np.float32)   # (Z,Y,X)
        v3d = self._resize_3d(vol)                        # (1,T,H,W)
        ehr = torch.from_numpy(r[self.ehr_cols].astype(np.float32).values)
        y = int(r["DIAGNOSIS"]) - 1   # {0,1,2}
        return v3d, ehr, y
