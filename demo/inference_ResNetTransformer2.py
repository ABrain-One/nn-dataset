import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import sys
import argparse

# Add root to path to import modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ab.nn.nn.ResNetTransformer import Net

# ... (skipping unchanged lines)

def load_vocab(vocab_path):
    print(f"Loading vocabulary from {vocab_path}...")
    data = torch.load(vocab_path)
    return data['word2idx'], data['idx2word']

def load_model(model_path, word2idx, idx2word, device):
    print(f"Loading model from {model_path}...")
    
    # Model parameters (must match training config)
    in_shape = (1, 3, 224, 224)
    out_shape = (len(word2idx),)
    prm = {'lr': 0.0001} # Dummy param for init
    
    model = Net(in_shape, out_shape, prm, device)
    
    # Load weights
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        # Handle potential 'module.' prefix from DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("Model weights loaded successfully.")
    else:
        print(f"WARNING: Checkpoint not found at {model_path}. Using random weights.")
    
    model.to(device)
    model.eval()
    model.word2idx = word2idx
    model.idx2word = idx2word
    return model

def generate_caption(model, image_path, device):
    # Standard ImageNet normalization
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    img_tensor = transform(img).to(device)
    
    print(f"\nGenerating caption for: {image_path}")
    with torch.no_grad():
        caption = model.beam_search_generate(img_tensor, model.word2idx, model.idx2word)
    
    return caption

def main():
    parser = argparse.ArgumentParser(description="Image Captioning Inference")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model_path', type=str, default="out/checkpoints/ab.nn.nn.ResNetTransformer/best_model.pth", help='Path to model checkpoint')
    parser.add_argument('--vocab_path', type=str, default="data/coco/vocab.pth", help='Path to vocab file')
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    vocab_path = ROOT / args.vocab_path
    model_path = ROOT / args.model_path
    image_path = Path(args.image)

    if not vocab_path.exists():
        print(f"Error: Vocab file not found at {vocab_path}")
        return

    # 1. Load Vocab
    word2idx, idx2word = load_vocab(vocab_path)

    # 2. Load Model
    model = load_model(model_path, word2idx, idx2word, device)

    # 3. Inference
    caption = generate_caption(model, image_path, device)
    
    print("-" * 50)
    print(f"OUTPUT CAPTION: {caption}")
    print("-" * 50)

if __name__ == "__main__":
    main()
