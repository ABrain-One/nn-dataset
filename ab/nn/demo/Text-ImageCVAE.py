
import torch
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
from huggingface_hub import hf_hub_download

# --- 1. Import Both Model Architectures ---
from ab.nn.nn.ConditionalVAE3 import Net as NetV3
from ab.nn.nn.ConditionalVAE4 import Net as NetV4

# --- 2. Configuration and Weight Downloading for BOTH models ---
REPO_ID = "NN-Dataset/ConditionalVAE4-checkpoints"
MODELS_TO_LOAD = {
    "ConditionalVAE3": {
        "class": NetV3,
        "filename": "ConditionalVAE3/best_model.pth"
    },
    "ConditionalVAE4": {
        "class": NetV4,
        "filename": "ConditionalVAE4/best_model.pth"
    }
}

models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_name, config in MODELS_TO_LOAD.items():
    print(f"--- Loading Model: {model_name} ---")
    checkpoint_dir = Path(f"checkpoints/{model_name}")
    # This is the ideal path, but we'll verify it after download
    weights_path = checkpoint_dir / "best_model.pth"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(weights_path):
        print(f"Weights for {model_name} not found locally. Downloading...")
        try:
            # --- THE FIX: Capture the returned path from the download function ---
            # This works with older library versions by omitting 'local_file'
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=config["filename"],
                local_dir=checkpoint_dir,
                local_dir_use_symlinks=False
            )
            # Use the actual path where the file was saved
            weights_path = downloaded_path
            print("Download complete.")
        except Exception as e:
            raise RuntimeError(f"Failed to download weights for {model_name}. Error: {e}")

    # Instantiate the correct model class
    model_instance = config["class"](in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device=device).to(device)

    # Load weights from the verified path
    model_instance.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model_instance.eval()

    models[model_name] = model_instance
    print(f"--- Model '{model_name}' Ready ---")

# --- 3. FastAPI Web Server ---
OUTPUT_DIR = Path("demo/generated_images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Prompt(BaseModel):
    text: str
    model_choice: str


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse('/home/robin/VL/nn-dataset/ab/nn/demo/Text-ImageCVAE.html')


@app.post("/generate")
async def generate_image_api(prompt: Prompt):
    print(f"Received prompt: '{prompt.text}' for model: '{prompt.model_choice}'")

    model_to_use = models.get(prompt.model_choice)
    if not model_to_use:
        return {"error": f"Model '{prompt.model_choice}' not found."}

    generated_image = model_to_use.generate([prompt.text])[0]

    safe_filename = "".join(c for c in prompt.text if c.isalnum() or c in (' ', '_')).rstrip()
    image_filename = f"{prompt.model_choice}_{safe_filename.replace(' ', '_')[:30]}_{os.urandom(4).hex()}.png"
    image_path = os.path.join(OUTPUT_DIR, image_filename)
    generated_image.save(image_path)
    print(f"Image saved to {image_path}")

    return {"image_path": f"demo/generated_images/{image_filename}"}


@app.get("/demo/generated_images/{image_name}")
async def get_generated_image(image_name: str):
    return FileResponse(os.path.join(OUTPUT_DIR, image_name))


# --- 4. Server Launch ---
if __name__ == "__main__":
    uvicorn.run(app, host="10.85.13.56", port=8000)