import json
from pathlib import Path
from ab.nn.util.Util import *

def generate_checksum(obj):
    """Generate a SHA256 checksum from a JSON-serializable object, excluding 'uid'."""
    obj_copy = {k: v for k, v in obj.items() if k != 'uid'}
    return uuid4(obj_copy)

# Path to the 'stat' folder (same directory as the script)
base_dir = Path(__file__).resolve().parent / "stat"
for p in base_dir.iterdir():
    if not p.is_dir():
        continue
    for epoch_file in p.iterdir():
        try:
            epoch = int(epoch_file.stem)
        except Exception:
            continue
        with open(epoch_file, 'r') as f:
            trials = json.load(f)
        trials2 = []
        for prm in trials:
            checksum = generate_checksum(prm)
            prm['uid'] = checksum  # ✅ Replace uid with checksum
            trials2.append(prm)
        with open(epoch_file, "w") as f:
            json.dump(trials2, f, indent=4)
        print(f"Processed: {epoch_file}")
