import os
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from ab.nn.nn.UnetD import Net as UnetD_Net
from ab.nn.util.Const import demo_dir

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = demo_dir / 'generated_images'

os.makedirs(IMAGE_DIR, exist_ok=True)

# --- Pointing to your Unet-D model repository ---
model_name = 'Unet-D'
REPO_ID = "NN-Dataset/Unet-D-checkpoints"

# This assumes the script is run from a 'demo' directory inside the project
checkpoint_dir = os.path.join(demo_dir, 'checkpoints', model_name)

# --- Point to the 'best_model.pth' file your script saves ---
ch_file = 'best_model.pth'
checkpoint_file = os.path.join(checkpoint_dir, ch_file)

if not os.path.exists(checkpoint_file):
    print(f"Weights for {model_name} not found locally. Downloading from {REPO_ID}...")
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Download the checkpoint file
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=ch_file,
            local_dir=checkpoint_dir,
            local_dir_use_symlinks=False
        )
        print("Download complete.")
    except Exception as e:
        raise RuntimeError(f"Failed to download weights for {model_name}. Error: {e}")

# --- Load Model ---
model = None
if UnetD_Net and os.path.exists(checkpoint_file):
    try:
        # --- Initialize the Unet-D 'Net' class ---
        # NOTE: 'model_channels': 96 is taken from your Unet-D.py's __main__ block.
        # If your model was trained with 128, change 96 to 128.
        prm = {'model_channels': 96}

        model = UnetD_Net(
            in_shape=None,  # Not needed for inference
            out_shape=None,  # Not needed for inference
            prm=prm,
            device=DEVICE
        )

        # --- Load weights manually into the compiled model's original module ---
        # This is the correct way, as seen in your Unet_D.py __main__ block
        model.unet._orig_mod.load_state_dict(torch.load(checkpoint_file, map_location=torch.device(DEVICE)))
        model.to(DEVICE)
        model.eval()  # Set model to evaluation mode
        print(f"Unet-D model loaded successfully from {checkpoint_file} on {DEVICE}")
    except Exception as e:
        print(f"--- ERROR LOADING MODEL: {e} ---")
        print("This may be due to a 'model_channels' mismatch (try 96 or 128).")
        model = None
elif not os.path.exists(checkpoint_file):
    print(f"Checkpoint file not found at {checkpoint_file}. Model not loaded.")
else:
    print("Model class could not be initialized. The application will not work.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    prompt: str


@app.get("/")
async def serve_index():
    """Serves the main index.html file."""
    # --- THIS LINE IS UPDATED ---
    return FileResponse(demo_dir / 'Text-ImageUnet.html')


@app.post("/generate_image")
async def generate_image_endpoint(request: ImageRequest):
    if model is None:
        return JSONResponse(content={"error": "Model or Tokenizer is not loaded."}, status_code=500)
    try:
        # --- Use the model's 'generate' method for diffusion ---
        with torch.no_grad():
            # num_inference_steps=50 is a good balance for a demo.
            # Your script's __main__ uses 250, which might be slow.
            generated_images_list = model.generate(
                text_prompts=[request.prompt],
                num_inference_steps=50
            )

        if not generated_images_list:
            raise Exception("Model failed to generate an image.")

        # The generate method returns a list of PIL Images
        generated_image_pil = generated_images_list[0]

        image_name = f"{uuid.uuid4()}.png"
        image_path = os.path.join(IMAGE_DIR, image_name)

        # --- Save the PIL image directly ---
        generated_image_pil.save(image_path)

        print(f"Image saved to {image_path}")
        return JSONResponse({"image_path": image_name})

    except Exception as e:
        print(f"Error during image generation: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/{image_path:path}")
async def get_image(image_path: str):
    """Serves the generated images."""
    if ".." in image_path or image_path == "Text-ImageUnet.html":
        return JSONResponse(content={"error": "Invalid file path"}, status_code=400)

    full_path = os.path.join(IMAGE_DIR, image_path)
    if os.path.exists(full_path) and os.path.isfile(full_path):
        return FileResponse(full_path)
    return JSONResponse(content={"error": "File not found"}, status_code=404)


# --- Server Launch ---
if __name__ == "__main__":
    # Listen on 0.0.0.0 to be accessible remotely
    uvicorn.run(app, host="0.0.0.0", port=8002)