import os
import sys
import uuid
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.cors import CORSMiddleware
from torchvision.utils import save_image
from transformers import CLIPTokenizer
from huggingface_hub import hf_hub_download

# Add the project root to the Python path to allow importing from 'ab'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# --- Model & Tokenizer Configuration ---
try:
    from ab.nn.nn.ConditionalGAN import Net as ConditionalGAN
except ImportError as e:
    print(f"Error importing ConditionalGAN: {e}")
    print("Please ensure the file 'ConditionalGAN.py' exists in 'nn-dataset/ab/nn/nn/'")
    ConditionalGAN = None

# --- Hugging Face repository details ---
HF_REPO_ID = "NN-Dataset/ConditionalGAN-checkpoints"
HF_FILENAME = "generator.pth"

def download_checkpoint_from_hf(repo_id, filename):
    """
    Downloads a model checkpoint from the Hugging Face Hub and returns its local path.
    """
    try:
        print(f"Downloading {filename} from {repo_id}...")
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
        print("Download complete.")
        return checkpoint_path
    except Exception as e:
        print(f"--- ERROR DOWNLOADING CHECKPOINT: {e} ---")
        return None

GENERATOR_CHECKPOINT_PATH = download_checkpoint_from_hf(HF_REPO_ID, HF_FILENAME)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOISE_DIM = 100
MAX_LENGTH = 16

# --- Create Absolute Paths ---
# Get the directory where app.py is located to resolve path issues
script_dir = os.path.dirname(__file__)
IMAGE_DIR = os.path.join(script_dir, "generated_images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# --- Initialize Tokenizer ---
try:
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
except Exception as e:
    print(f"Could not load CLIPTokenizer: {e}")
    tokenizer = None

# --- Load Model ---
model = None
if ConditionalGAN and tokenizer and GENERATOR_CHECKPOINT_PATH:
    try:
        shape_a_placeholder, shape_b_placeholder, prm_placeholder = (0,), (0,), {}
        model = ConditionalGAN(
            shape_a=shape_a_placeholder,
            shape_b=shape_b_placeholder,
            prm=prm_placeholder,
            device=DEVICE
        )
        model.generator.load_state_dict(torch.load(GENERATOR_CHECKPOINT_PATH, map_location=torch.device(DEVICE)))
        model.to(DEVICE)
        model.eval()
        print(f"Generator model loaded successfully from {GENERATOR_CHECKPOINT_PATH} on {DEVICE}")
    except Exception as e:
        print(f"--- ERROR LOADING MODEL: {e} ---")
        model = None
else:
    if not GENERATOR_CHECKPOINT_PATH:
        print("Model loading skipped because checkpoint could not be downloaded.")
    else:
        print("Model or Tokenizer could not be initialized. The application will not work.")

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

# --- Define path to index.html to serve the frontend ---
index_html_path = os.path.join(script_dir, "index.html")

@app.get("/")
async def serve_index():
    """Serves the main index.html file."""
    if not os.path.exists(index_html_path):
        return JSONResponse(content={"error": f"index.html not found at {index_html_path}"}, status_code=404)
    return FileResponse(index_html_path)

def text_to_tokens(prompt: str):
    tokenized_output = tokenizer(
        [prompt],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    return tokenized_output['input_ids'].to(DEVICE)

@app.post("/generate_image")
async def generate_image_endpoint(request: ImageRequest):
    if model is None or tokenizer is None:
        return JSONResponse(content={"error": "Model or Tokenizer is not loaded."}, status_code=500)
    try:
        text_tokens = text_to_tokens(request.prompt)
        noise = torch.randn(1, NOISE_DIM, device=DEVICE)
        with torch.no_grad():
            generated_image = model.generator(noise, text_tokens)

        generated_image = (generated_image * 0.5 + 0.5).clamp(0, 1)
        image_name = f"{uuid.uuid4()}.png"
        image_path = os.path.join(IMAGE_DIR, image_name)
        save_image(generated_image, image_path)
        print(f"Image saved to {image_path}")
        return JSONResponse({"image_path": image_name})

    except Exception as e:
        print(f"Error during image generation: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/{image_path:path}")
async def get_image(image_path: str):
    if ".." in image_path or image_path == "index.html":
        return JSONResponse(content={"error": "Invalid file path"}, status_code=400)

    full_path = os.path.join(IMAGE_DIR, image_path)
    if os.path.exists(full_path) and os.path.isfile(full_path):
        return FileResponse(full_path)
    return JSONResponse(content={"error": "File not found"}, status_code=404)

# --- Server Launch Block ---
if __name__ == "__main__":
    # This block allows running the app directly with 'python -m ab.nn.demo.app'
    # Note: The IP address is hardcoded. Change to "0.0.0.0" to allow access from other devices on your network.
    uvicorn.run(app, host="10.85.13.56", port=8006)