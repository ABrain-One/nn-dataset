import sys
import os
import torch
import uuid
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from PIL import Image

# This block tells Python to look for modules in the 'nn-dataset' directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# This import will now work correctly
from ab.nn.nn.ConditionalVAE4 import Net

# --- Configuration ---
IMAGE_DIR = "generated_images"
PROMPT_FILE = "prompts.txt"

# Create directory
os.makedirs(IMAGE_DIR, exist_ok=True)

# --- Load the Model ---
print("--- Loading Trained Model ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_in_shape = (1, 3, 256, 256)

# The model's __init__ method will automatically find and load the best checkpoint.
model = Net(in_shape=dummy_in_shape, out_shape=None, prm={}, device=device).to(device)

model.eval()
print("--- Model Ready ---")

# --- FastAPI Setup ---
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

@app.post("/generate_image")
async def generate_image(request: ImageRequest):
    try:
        generated_images = model.generate([request.prompt])
        pil_image = generated_images[0]

        unique_id = str(uuid.uuid4())
        image_filename = f"{unique_id}.png"
        image_path = os.path.join(IMAGE_DIR, image_filename)

        pil_image.save(image_path)

        with open(PROMPT_FILE, "a") as f:
            f.write(f"{unique_id}: {request.prompt}\n")

        return JSONResponse({"image_url": image_filename})

    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/generated_images/{image_path}")
async def get_image(image_path: str):
    image_file = os.path.join(IMAGE_DIR, image_path)
    if os.path.exists(image_file):
        return FileResponse(image_file)
    return JSONResponse(content={"error": "File not found"}, status_code=404)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Corrected path to point to the file inside the DEMO folder
    return FileResponse('DEMO/index.html')

if __name__ == "__main__":
    import uvicorn
    # Updated port to 8003
    uvicorn.run(app, host="0.0.0.0", port=8003)