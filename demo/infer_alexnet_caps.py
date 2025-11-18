import sys, os, importlib.util, inspect
import torch, torch.nn as nn

# --- dynamic import of the model file ---
repo_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(repo_root)  # ensure project root on path

model_mod_path = os.path.join(repo_root, "ab", "nn", "nn", "C10C_ALEXNETLSTM.py")
if not os.path.exists(model_mod_path):
    # fallback if file still has hyphen:
    model_mod_path = os.path.join(repo_root, "ab", "nn", "nn", "C10C-ALEXNETLSTM.py")
if not os.path.exists(model_mod_path):
    raise FileNotFoundError(f"Model file not found at {model_mod_path}")

spec = importlib.util.spec_from_file_location("model_module", model_mod_path)
model_module = importlib.util.module_from_spec(spec)
sys.modules["model_module"] = model_module
spec.loader.exec_module(model_module)

# --- try common class names first ---
_candidate_names = [
    "AlexNetTransformerCaptioner",
    "CaptionModel",
    "Model",
    "AlexNetLSTM",
    "AlexNetTransformer",
    "C10C_ALEXNETLSTM",
    "C10CAlexNet",
]
Model = None
for name in _candidate_names:
    Model = getattr(model_module, name, None)
    if isinstance(Model, type):
        break

# --- if still not found, auto-detect an nn.Module subclass ---
if Model is None:
    candidates = []
    for name, obj in inspect.getmembers(model_module, inspect.isclass):
        if issubclass(obj, nn.Module) and obj.__module__ == "model_module":
            # prefer classes that expose .generate or look like captioners
            score = 0
            if hasattr(obj, "generate"): score += 5
            if "caption" in name.lower(): score += 3
            if "alex" in name.lower():    score += 2
            # rough param count heuristic
            try:
                param_count = sum(p.numel() for p in obj().__dict__.get('parameters', []) )
            except Exception:
                param_count = 0
            candidates.append((score, name, obj, param_count))
    if candidates:
        # pick highest score; if tie, any
        candidates.sort(key=lambda x: (x[0], x[3]), reverse=True)
        Model = candidates[0][2]

if Model is None:
    raise ImportError(
        "Could not locate the model class in C10C_ALEXNETLSTM.py. "
        "Please open the file and tell me the exact class name (e.g., CaptionModel)."
    )
