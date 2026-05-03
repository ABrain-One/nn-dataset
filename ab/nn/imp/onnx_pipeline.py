#!/usr/bin/env python3
import sys, json, shutil, importlib.util, re
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
import onnxruntime as ort

from huggingface_hub import (
    hf_hub_download,
    list_repo_files,
    upload_folder,
    create_repo,
)

# --- CONFIG ---
ONNX_TARGET_REPO = "NN-Dataset/onnx"
SOURCE_REPO = "NN-Dataset/checkpoints-epoch-50"


# ------------------------
# ONNX ACCURACY
# ------------------------
def eval_onnx_acc(onnx_path, target_h, data_root):
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name

    tfm = T.Compose([
        T.ToTensor(),
        T.Resize((target_h, target_h)),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010))
    ])

    dataset = torchvision.datasets.CIFAR10(
        root=str(data_root), train=False, download=True, transform=tfm
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=100)

    correct, total = 0, 0

    for x, y in loader:
        for i in range(len(x)):
            inp = x[i:i+1].cpu().numpy()

            outputs = session.run(None, {input_name: inp})[0]
            pred = outputs.argmax(axis=1)[0]

            if pred == y[i].item():
                correct += 1

            total += 1

            if total >= 1000:
                break
        if total >= 1000:
            break

    return correct / total


# ------------------------
# MAIN PIPELINE
# ------------------------
def run_onnx_pipeline(args, dataset_root):

    # --- PATH SETUP ---
    sys.path.insert(0, str(dataset_root))

    work_dir = dataset_root / "_work"
    out_dir = work_dir / "onnx"
    data_root = work_dir / "data"
    temp_dir = work_dir / "temp"

    for p in [out_dir, data_root, temp_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # --- HF SETUP ---
    if args.push_hf:
        create_repo(ONNX_TARGET_REPO, repo_type="model", exist_ok=True)

    arch_dir = dataset_root / "ab" / "nn" / "nn"
    transforms_dir = dataset_root / "ab" / "nn" / "transform"
    local_models_json = dataset_root / "all_models.json"

    # download metadata
    try:
        f = hf_hub_download(SOURCE_REPO, "all_models.json", local_dir=str(out_dir))
        shutil.copy(f, local_models_json)
    except:
        pass

    model_db = json.load(open(local_models_json))
    hf_files = list_repo_files(SOURCE_REPO)

    py_files = sorted([
        p for p in arch_dir.rglob("*.py")
        if f"{p.stem}.pth" in hf_files
    ])

    # --- TARGET DIR ---
    target_dir = out_dir / "fp32" / "img-classification_cifar-10_acc"
    target_dir.mkdir(parents=True, exist_ok=True)

    # --- LOAD JSON ---
    json_path = target_dir / "all_models.json"
    results = json.load(open(json_path)) if json_path.exists() else {}

    skipped_path = target_dir / "skipped_models.json"
    skipped = json.load(open(skipped_path)) if skipped_path.exists() else {}

    # ------------------------
    # PROCESS LOOP
    # ------------------------
    for py_path in py_files:
        name = py_path.stem

        if name in results:
            print(f"SKIP {name} (already processed)")
            continue

        try:
            print(f"\nProcessing {name}")

            if name not in model_db:
                print("  skip: no metadata")
                continue

            prm = model_db[name].get("prm", {})
            transform_val = prm.get("transform", None)

            # --- RESOLUTION ---
            target_h = 32
            if transform_val:
                tf_file = transforms_dir / f"{transform_val}.py"
                if tf_file.exists():
                    m = re.search(r"(Resize|size).*?(\d+)", tf_file.read_text())
                    if m:
                        target_h = int(m.group(2))

            onnx_path = target_dir / f"{name}.onnx"

            # --- REUSE EXISTING ONNX ---
            if onnx_path.exists():
                print(f"  USING EXISTING ONNX {name}")

                acc = eval_onnx_acc(onnx_path, target_h, data_root)
                print(f"  accuracy: {acc:.4f}")

                results[name] = {
                    "accuracy": acc,
                    "transform": transform_val
                }
                continue

            # --- DOWNLOAD CHECKPOINT ---
            try:
                pth = hf_hub_download(SOURCE_REPO, f"{name}.pth", cache_dir=str(temp_dir))
            except Exception as e:
                print(f"  SKIP {name} (download failed): {e}")
                skipped[name] = f"download_failed: {str(e)}"
                continue

            # --- LOAD MODEL ---
            spec = importlib.util.spec_from_file_location("mod", py_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            model = mod.Net(
                in_shape=(1, 3, 32, 32),
                out_shape=(10,),
                prm=prm,
                device="cpu"
            )

            try:
                ckpt = torch.load(pth, map_location="cpu")
                model.load_state_dict(
                    ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt,
                    strict=False
                )
            except Exception as e:
                print(f"  SKIP {name} (load failed): {e}")
                skipped[name] = f"load_failed: {str(e)}"
                continue

            model.eval()

            dummy = torch.randn(1, 3, target_h, target_h)

            # --- EXPORT ONNX ---
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                input_names=["input"],
                output_names=["output"],
                opset_version=18
            )

            # --- EVAL ---
            acc = eval_onnx_acc(onnx_path, target_h, data_root)
            print(f"  accuracy: {acc:.4f}")

            results[name] = {
                "accuracy": acc,
                "transform": transform_val
            }

        except Exception as e:
            print(f"  SKIP {name} (runtime failed): {e}")
            skipped[name] = f"runtime_failed: {str(e)}"
            continue

    # ------------------------
    # SAVE JSON
    # ------------------------
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(skipped_path, "w") as f:
        json.dump(skipped, f, indent=2)

    print("Saved JSON files")

    # ------------------------
    # UPLOAD
    # ------------------------
    if args.push_hf:
        upload_folder(
            folder_path=str(out_dir),
            repo_id=ONNX_TARGET_REPO,
            repo_type="model",
        )