import argparse, json, re, importlib.util, torch
from pathlib import Path
from collections import Counter
from PIL import Image
import torchvision.transforms as T

def build_vocab(ann_path: Path, max_words: int = 1000):
    # special tokens
    special = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    if ann_path and ann_path.exists():
        data = json.load(open(ann_path))
        tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
        ctr = Counter()
        for a in data.get("annotations", []):
            ctr.update(tok(a.get("caption", "")))
        words = [w for w, _ in ctr.most_common(max(1, max_words - len(special)))]
        vocab = special + words
        note = f"built from {ann_path} ({len(vocab)} words)"
    else:
        # fallback “dummy” vocab—still decodes consistently, but words may be placeholders
        vocab = special + [f"w{i}" for i in range(max_words - len(special))]
        note = "fallback vocab (no COCO captions found)"
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    return vocab, w2i, i2w, note

def load_model(model_file: Path, ckpt_path: Path, vocab_size: int, device, max_len: int):
    # import model file directly (ab/nn/nn/C10C-ALEXNETLSTM.py)
    assert model_file.exists(), f"Missing model file: {model_file}"
    spec = importlib.util.spec_from_file_location("C10C_ALEXNETLSTM", str(model_file))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

    net = m.Net((2,3,224,224), (vocab_size,), {"lr":3e-4,"momentum":0.9,"dropout":0.2,"max_len":max_len}, device).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict):
        sd = sd.get("state_dict", sd)

    try:
        net.load_state_dict(sd, strict=False)
    except RuntimeError:
        # if vocab shapes don’t match, drop decoder weights and load the rest
        for k in list(sd.keys()):
            if k.startswith("decoder.") or "embed" in k or k.startswith("fc."):
                sd.pop(k)
        net.load_state_dict(sd, strict=False)
    net.eval()
    return net

def decode(ids, idx2word, pad_id, bos_id, eos_id):
    words = []
    for i in ids:
        if i == eos_id: break
        if i not in (pad_id, bos_id):
            words.append(idx2word.get(int(i), f"<{i}>"))
    return " ".join(words) or "(empty)"

def main():
    p = argparse.ArgumentParser(description="Single-image caption inference (AlexNet encoder + Transformer decoder)")
    p.add_argument("--image", required=True, help="Absolute path to an image (JPG/PNG). Tip: drag & drop from Finder.")
    p.add_argument("--ckpt", default=None, help="Path to .pth checkpoint (defaults to newest in _weights/)")
    p.add_argument("--ann", default="data/coco/annotations/captions_train2017.json", help="COCO captions file (optional)")
    p.add_argument("--max_len", type=int, default=20)
    args = p.parse_args()

    ROOT = Path.cwd()
    img_path = Path(args.image).expanduser()
    assert img_path.exists(), f"Image not found: {img_path}"

    # checkpoint
    if args.ckpt:
        ckpt = Path(args.ckpt)
        assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
    else:
        ckpts = sorted((ROOT/"_weights").glob("*.pth"), key=lambda p: p.stat().st_mtime)
        assert ckpts, "No checkpoints in _weights/. Train or copy one there."
        ckpt = ckpts[-1]

    # vocab
    ann_path = (ROOT/args.ann) if not Path(args.ann).is_absolute() else Path(args.ann)
    vocab, w2i, i2w, note = build_vocab(ann_path)
    PAD, BOS, EOS = w2i["<PAD>"], w2i["<BOS>"], w2i["<EOS>"]
    V = len(vocab)

    print(f"Using ckpt: {ckpt.name}")
    print(f"Vocab: {V} words ({note})")

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_file = ROOT/"ab/nn/nn/C10C-ALEXNETLSTM.py"
    net = load_model(model_file, ckpt, V, device, args.max_len)

    # preprocess
    tf = T.Compose([T.Resize((224,224)), T.ToTensor()])

    # run
    with torch.no_grad():
        x = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        logits = net(x, captions=None)              # (B, L, V)
        ids = logits.argmax(-1).squeeze(0).tolist()
        caption = decode(ids, i2w, PAD, BOS, EOS)

    # print + save
    print("\nCaption ->", caption, "\n")
    out_dir = ROOT/"_runs/single_caption"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir/f"{img_path.stem}.txt"
    out_json = out_dir/f"{img_path.stem}.json"
    out_txt.write_text(caption)
    json.dump({"image": img_path.name, "caption": caption}, open(out_json, "w"), indent=2)
    print("Saved:", out_txt)
    print("Saved:", out_json)

if __name__ == "__main__":
    main()
