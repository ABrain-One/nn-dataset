# File: ab/nn/demo/Text-ImageCVAE.py
# Description: A dedicated demo script for the ConditionalVAE4 model.

import torch
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- 1. Load Your Custom ConditionalVAE4 Model ---
print("--- Loading Your Custom ConditionalVAE4 Model ---")
from ab.nn.nn.ConditionalVAE4 import Net

WEIGHTS_PATH = "checkpoints/ConditionalVAE4/best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your CVAE-GAN architecture and weights
model = Net(in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device=device).to(device)

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"ConditionalVAE4 weights not found at {WEIGHTS_PATH}. Please ensure the checkpoint file exists.")

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

print(f"--- Model 'ConditionalVAE4' Ready ---")

# --- 2. FastAPI Web Server ---
OUTPUT_DIR = "ab/nn/demo/generated_images"

app = FastAPI()

# Add CORS middleware to allow browser connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Prompt(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse('ab/nn/demo/index.html')


@app.post("/generate")
async def generate_image_api(prompt: Prompt):
    print(f"Received prompt: {prompt.text}")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Use the loaded model to generate an image
    generated_image = model.generate([prompt.text])[0]

    # Save the generated image
    safe_filename = "".join(c for c in prompt.text if c.isalnum() or c in (' ', '_')).rstrip()
    image_filename = f"{safe_filename.replace(' ', '_')[:30]}.png"
    image_path = os.path.join(OUTPUT_DIR, image_filename)
    generated_image.save(image_path)
    print(f"Image saved to {image_path}")

    return {"image_path": image_path.replace("ab/nn/", "")}


@app.get("/demo/generated_images/{image_name}")
async def get_generated_image(image_name: str):
    return FileResponse(os.path.join(OUTPUT_DIR, image_name))


# --- 3. Server Launch ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)