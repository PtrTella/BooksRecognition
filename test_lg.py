"""Quick test to inspect LightGlue output format."""
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import torch, cv2, numpy as np
from kornia.feature import DISK, LightGlue

device = torch.device('mps')
disk = DISK.from_pretrained('depth').to(device)
lg = LightGlue(features='disk').to(device).eval()

img_m = cv2.imread('dataset/models/model_17.png')
img_s = cv2.imread('dataset/scenes/scene_2.jpg')

def to_rgb(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).float().permute(2,0,1).unsqueeze(0) / 255.0

tm = to_rgb(img_m).to(device)
ts = to_rgb(img_s).to(device)

with torch.no_grad():
    fm = disk(tm, n=2048, pad_if_not_divisible=True)
    fs = disk(ts, n=2048, pad_if_not_divisible=True)

print(f'Model kp: {fm[0].keypoints.shape}')
print(f'Scene kp: {fs[0].keypoints.shape}')

with torch.no_grad():
    inp = {
        'image0': {
            'keypoints': fm[0].keypoints.unsqueeze(0),
            'descriptors': fm[0].descriptors.unsqueeze(0),
            'image_size': torch.tensor([[img_m.shape[1], img_m.shape[0]]]).to(device),
        },
        'image1': {
            'keypoints': fs[0].keypoints.unsqueeze(0),
            'descriptors': fs[0].descriptors.unsqueeze(0),
            'image_size': torch.tensor([[img_s.shape[1], img_s.shape[0]]]).to(device),
        },
    }
    result = lg(inp)

print(f'Result keys: {list(result.keys())}')
m = result['matches'][0].cpu().numpy()
print(f'Matches shape: {m.shape}')
print(f'First 5 matches:\n{m[:5]}')

# Check format
if m.ndim == 1:
    print("FORMAT: 1D - matches[i] = j means kp0[i] -> kp1[j], -1=unmatched")
    valid = m >= 0
    print(f'Valid matches: {valid.sum()}')
elif m.ndim == 2:
    print(f"FORMAT: 2D with {m.shape[1]} columns")
    valid = (m[:, 0] >= 0) & (m[:, 1] >= 0)
    print(f'Valid matches: {valid.sum()}')

if 'scores' in result:
    scores = result['scores'][0].cpu().numpy()
    print(f'Scores shape: {scores.shape}')
    if m.ndim == 1:
        vs = scores[m >= 0]
    else:
        vs = scores[valid]
    if len(vs) > 0:
        print(f'Score range: {vs.min():.3f} - {vs.max():.3f}, median: {np.median(vs):.3f}')

# Also test a non-matching pair  
img_bad = cv2.imread('dataset/scenes/scene_11.jpg')
tb = to_rgb(img_bad).to(device)
with torch.no_grad():
    fb = disk(tb, n=2048, pad_if_not_divisible=True)
    inp2 = {
        'image0': {
            'keypoints': fm[0].keypoints.unsqueeze(0),
            'descriptors': fm[0].descriptors.unsqueeze(0),
            'image_size': torch.tensor([[img_m.shape[1], img_m.shape[0]]]).to(device),
        },
        'image1': {
            'keypoints': fb[0].keypoints.unsqueeze(0),
            'descriptors': fb[0].descriptors.unsqueeze(0),
            'image_size': torch.tensor([[img_bad.shape[1], img_bad.shape[0]]]).to(device),
        },
    }
    result2 = lg(inp2)

m2 = result2['matches'][0].cpu().numpy()
s2 = result2['scores'][0].cpu().numpy()
if m2.ndim == 1:
    v2 = (m2 >= 0).sum()
else:
    v2 = ((m2[:, 0] >= 0) & (m2[:, 1] >= 0)).sum()
print(f'\nNon-matching pair (model_17 vs scene_11): {v2} matches')
print(f'False match scores: min={s2.min():.3f}, max={s2.max():.3f}, median={np.median(s2):.3f}')
print(f'Scores > 0.5: {(s2 > 0.5).sum()}, > 0.7: {(s2 > 0.7).sum()}, > 0.9: {(s2 > 0.9).sum()}')

# Check true pair score distribution
s1 = result['scores'][0].cpu().numpy()
print(f'\nTrue match scores: min={s1.min():.3f}, max={s1.max():.3f}, median={np.median(s1):.3f}')
print(f'Scores > 0.5: {(s1 > 0.5).sum()}, > 0.7: {(s1 > 0.7).sum()}, > 0.9: {(s1 > 0.9).sum()}')
