import json
import importlib
from pathlib import Path
import sys

import torch
from PIL import Image
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parents[1]
print(">>> ROOT:", ROOT)


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODEL_MODULE = "ab.nn.nn.C10C-ALEXNETLSTM-mohsin"

CKPT_PATH = ROOT / "_weights" / "C10C-ALEXNETLSTM_tinycaps_trained.pth"

VOCAB_PATH = ROOT / "data" / "coco" / "vocab_10k.json"

IMG_DIR = ROOT / "data" / "demo_images"



def load_vocab(path: Path):
    print(">>> loading vocab from:", path)
    with path.open() as f:
        vocab = json.load(f)

    w2i = vocab["word2idx"]
    idx2word = {int(k): v for k, v in vocab["idx2word"].items()}
    pad = w2i["<PAD>"]
    bos = w2i["<BOS>"]
    eos = w2i["<EOS>"]
    print(">>> vocab loaded; size:", len(w2i))
    return w2i, idx2word, pad, bos, eos


def load_model(module_name: str, ckpt: Path, vocab_size: int, device):
    print(">>> importing model module:", module_name)
    m = importlib.import_module(module_name)
    Net = m.Net

    print(">>> building Net(vocab_size =", vocab_size, ")")
    net = Net(
        in_shape=(2, 3, 224, 224),
        out_shape=(vocab_size,),
        prm={"lr": 1e-3, "momentum": 0.9, "dropout": 0.2, "max_len": 20},
        device=device,
    ).to(device)

    print(">>> loading checkpoint from:", ckpt)
    state = torch.load(ckpt, map_location=device)
    state = state.get("state_dict", state)
    missing, unexpected = net.load_state_dict(state, strict=False)
    print(">>> loaded ckpt; missing:", len(missing), "unexpected:", len(unexpected))
    net.eval()
    return net


def caption_image(img_path: Path, net, idx2word, pad_id, bos_id, eos_id, device):
    print(">>> captioning image:", img_path)
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = net(x, captions=None)
        ids = logits.argmax(-1).squeeze(0).tolist()

    print(">>> raw token ids:", ids)

    words = []
    for i in ids:
        if i == eos_id:
            break
        if i in (pad_id, bos_id):
            continue
        words.append(idx2word.get(int(i), "<UNK>"))

    caption = " ".join(words) or "(empty)"
    return caption


def main():
    print(">>> starting img-caption-single.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> device:", device)


    for path, name in [
        (CKPT_PATH, "Checkpoint"),
        (VOCAB_PATH, "Vocab"),
        (IMG_DIR, "Image directory"),
    ]:
        print(f">>> checking {name} at: {path}")
        if not path.exists():
            raise FileNotFoundError(f"{name} not found at: {path}")

    # list images
    img_paths = sorted(IMG_DIR.glob("*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"No .jpg images found in {IMG_DIR}")

    # optionally limit to first 3 (so you can say to professor: “this script captions first 3 images in demo_images”)
    img_paths = img_paths[:3]
    print(">>> images to caption:", [p.name for p in img_paths])

    # load vocab + model
    w2i, idx2word, pad, bos, eos = load_vocab(VOCAB_PATH)
    net = load_model(MODEL_MODULE, CKPT_PATH, len(w2i), device)

    print("\n—— RESULTS ——")
    for p in img_paths:
        cap = caption_image(p, net, idx2word, pad, bos, eos, device)
        print(f"{p.name} -> {cap}")
    print("———————")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", repr(e))
