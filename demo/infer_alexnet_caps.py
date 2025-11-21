import json
import importlib.util
import sys
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T



ROOT     = Path(__file__).resolve().parents[1]
print("LEMUR root", ROOT    )

# Make sure the project root is on sys.path so `import ab...` works
if str(ROOT ) not in sys.path:
    sys.path.insert(0, str(ROOT ))

# Model file (your custom AlexNet captioner)
MODEL_PATH = ROOT    / "ab" / "nn" / "nn" / "C10C-ALEXNETLSTM-mohsin.py"

# Checkpoint and vocab
CKPT_PATH = ROOT     / "_weights" / "C10C-ALEXNETLSTM_tinycaps_trained.pth"
VOCAB_PATH = ROOT    / "data" / "coco" / "vocab_10k.json"

# Existing demo images (you said: nn-dataset/data/demo_images/*.jpg)
IMG_DIR = ROOT   / "data" / "demo_images"

# Output directory
OUT_DIR  = ROOT  / "_runs" / "alexnet_caps_demo"
OUT_JSON = OUT_DIR / "captions.json"
OUT_CSV  = OUT_DIR / "captions.csv"



for path, name in [
    (MODEL_PATH, "Model file"),
    (CKPT_PATH, "Checkpoint"),
    (VOCAB_PATH, "Vocab file"),
    (IMG_DIR, "Image directory"),
]:
    print(f"[CHECK] {name}: {path}")
    if not path.exists():
        raise FileNotFoundError(f"{name} not found at: {path}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Load vocab
# ------------------------------------------------------------

print("\n[STEP] Loading vocabulary...")
with VOCAB_PATH.open() as f:
    vocab = json.load(f)

w2i = vocab["word2idx"]
idx2word = {int(k): v for k, v in vocab["idx2word"].items()}

PAD = w2i["<PAD>"]
BOS = w2i["<BOS>"]
EOS = w2i["<EOS>"]

print(f"[OK] Vocab size: {len(w2i)}")

# ------------------------------------------------------------
# Load model from file path (works even with '-' in filename)
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[STEP] Device:", device)

print("[STEP] Loading model from:", MODEL_PATH)
spec = importlib.util.spec_from_file_location("alexnet_caption_net", MODEL_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not create spec for {MODEL_PATH}")

m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

Net = m.Net

vocab_size = len(w2i)
print(f"[STEP] Building Net(vocab_size={vocab_size})")
net = Net(
    in_shape=(2, 3, 224, 224),
    out_shape=(vocab_size,),
    prm={"lr": 1e-3, "momentum": 0.9, "dropout": 0.2, "max_len": 20},
    device=device,
).to(device)

print("[STEP] Loading checkpoint:", CKPT_PATH)
state = torch.load(CKPT_PATH, map_location=device)
state = state.get("state_dict", state)
missing, unexpected = net.load_state_dict(state, strict=False)
print(f"[OK] Loaded checkpoint; missing: {len(missing)} unexpected: {len(unexpected)}")
net.eval()

# ------------------------------------------------------------
# Caption all images in data/demo_images
# ------------------------------------------------------------

tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])

images = sorted(IMG_DIR.glob("*.jpg"))
print("\n[STEP] Images found:", [p.name for p in images])
if not images:
    raise RuntimeError(f"No .jpg images found in {IMG_DIR}")

results = []

for img_path in images:
    print(f"\n[STEP] Captioning: {img_path.name}")
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = net(x, captions=None)     # (1, T, V)
        ids = logits.argmax(-1).squeeze(0).tolist()

    print("  raw token ids:", ids)

    words = []
    for i in ids:
        if i == EOS:
            break
        if i in (PAD, BOS):
            continue
        words.append(idx2word.get(int(i), "<UNK>"))

    caption = " ".join(words) or "(empty)"
    print(f"[CAP] {img_path.name} -> {caption}")
    results.append({"image": img_path.name, "caption": caption})


with OUT_JSON.open("w") as f:
    json.dump(results, f, indent=2)
print("\n[OK] Saved JSON ->", OUT_JSON)

with OUT_CSV.open("w") as f:
    f.write("image,caption\n")
    for r in results:
        cap = r["caption"].replace('"', '""')
        f.write(f'{r["image"]},"{cap}"\n')
print("[OK] Saved CSV  ->", OUT_CSV)