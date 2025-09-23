import torch
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

# --- Make sure this import path is correct for your project ---
from ab.nn.nn.ConditionalVAE2 import Net, export_torch_weights  # Use your final model

# --- Configuration ---
# This MUST point to the final weights file you saved after training
WEIGHTS_PATH = "checkpoints/ConditionalVAE3/best_model.pth"
OUTPUT_DIR = "demo/generated_images"

# --- Load the Model ---
print("--- Loading Final Trained Model ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Dummy shape for initialization; the real weights will be loaded
dummy_in_shape = (1, 3, 256, 256)
model = Net(in_shape=dummy_in_shape, out_shape=None, prm={}, device=device).to(device)

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"Weights file not found at {WEIGHTS_PATH}. Please train the model and save the weights first.")

# Load the final, trained weights into the model
model.load_weights(WEIGHTS_PATH)
model.eval()  # Set the model to evaluation mode
print("--- Model Ready ---")

# --- FastAPI Web Server ---
app = FastAPI()


class Prompt(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Serve the main HTML page
    return FileResponse('demo/index.html')


@app.post("/generate")
async def generate_image(prompt: Prompt):
    print(f"Received prompt: {prompt.text}")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Use the model to generate a new image from the user's prompt
    generated_images = model.generate([prompt.text])

    # Save the generated image
    image_filename = f"{prompt.text.replace(' ', '_')[:30]}.png"
    image_path = os.path.join(OUTPUT_DIR, image_filename)
    generated_images[0].save(image_path)
    print(f"Image saved to {image_path}")

    return {"image_path": image_path}


@app.get("/generated_images/{image_name}")
async def get_generated_image(image_name: str):
    # Serve the generated image file
    return FileResponse(os.path.join(OUTPUT_DIR, image_name))
