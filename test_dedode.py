"""Quick DeDoDe matching debug."""
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import torch, cv2, numpy as np, warnings
warnings.filterwarnings("ignore")
from kornia.feature import DeDoDe

device = torch.device("mps")
d = DeDoDe.from_pretrained(
    detector_weights="L-upright", descriptor_weights="B-upright"
).to(device).eval()

def to_rgb(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0).to(device)

m_img = cv2.imread("dataset/models/model_2.png")
s_img = cv2.imread("dataset/scenes/scene_27.jpg")

with torch.no_grad():
    kp_m, sc_m, desc_m = d(to_rgb(m_img), n=2048)
    kp_s, sc_s, desc_s = d(to_rgb(s_img), n=2048)

print(f"Model: {kp_m.shape[1]} kps, desc {desc_m.shape}")
print(f"Scene: {kp_s.shape[1]} kps, desc {desc_s.shape}")

dm = desc_m[0].float()
ds = desc_s[0].float()
sim = torch.mm(dm, ds.T)
print(f"Sim range: {sim.min():.3f} to {sim.max():.3f}")

val_f, idx_f = sim.topk(2, dim=1)
ratio = val_f[:, 1] / (val_f[:, 0] + 1e-8)
print(f"Ratio range: {ratio.min():.3f} to {ratio.max():.3f}, mean={ratio.mean():.3f}")
print(f"Ratio < 0.78: {(ratio<0.78).sum()}, < 0.90: {(ratio<0.90).sum()}, < 0.95: {(ratio<0.95).sum()}")

# Also count mutual matches at various thresholds
_, idx_b = sim.topk(1, dim=0)
idx_b = idx_b.squeeze(0)
m_idx = torch.arange(len(dm), device=device)
mutual = idx_b[idx_f[:, 0]] == m_idx

for r in [0.78, 0.85, 0.90, 0.95, 0.98]:
    ok = (ratio < r) & mutual
    print(f"  ratio<{r} + mutual: {ok.sum()} matches")
