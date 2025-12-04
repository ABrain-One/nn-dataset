import argparse
import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import sys
import os

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ab.nn.nn.ResNetTransformer import Net

def load_vocab(path):
    print(f">>> Loading vocab from {path}")
    if not path.exists():
        raise FileNotFoundError(f"Vocab not found at {path}")
    data = torch.load(path)
    return data['word2idx'], data['idx2word']

def load_model(ckpt_path, vocab_size, device):
    print(f">>> Loading model from {ckpt_path}")
    # Dummy shapes/prms as they are mostly for training setup or unused by inference parts
    in_shape = (1, 3, 224, 224)
    out_shape = (vocab_size,)
    prm = {'lr': 0.0001} 
    
    model = Net(in_shape, out_shape, prm, device)
    
    if ckpt_path and ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location=device)
        # Handle if state_dict is inside a key like 'state_dict' or 'model'
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Remove 'module.' prefix if saved from DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        print(">>> Model weights loaded.")
    else:
        print(">>> WARNING: Checkpoint not found or not provided. Using random weights.")
    
    model.to(device)
    model.eval()
    return model

def caption_image(model, img_path, word2idx, idx2word, device):
    print(f">>> Captioning {img_path}")
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).to(device)
    
    # Use the model's beam search
    # Note: ResNetTransformer.py has a method batch_beam_search but also beam_search_generate
    # Let's use batch_beam_search for consistency with the class
    
    # We need to set word2idx/idx2word on the model instance if not already
    model.word2idx = word2idx
    model.idx2word = idx2word
    
    with torch.no_grad():
        # Add batch dim
        img_batch = img_tensor.unsqueeze(0)
        # batch_beam_search returns one-hot tensor [B, Seq, Vocab]
        # But we want the actual words.
        # Let's use the helper method `beam_search_generate_single` inside the model if possible,
        # or `beam_search_generate` which returns a string.
        
        # The class has `beam_search_generate` which takes image (3D) and returns string.
        # The class has `beam_search_generate` which takes image (3D) and returns string.
        caption = model.beam_search_generate(img_tensor, word2idx, idx2word, repetition_penalty=1.2)
        
    return caption

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True, help='Path to image')
    parser.add_argument('--ckpt', type=str, help='Path to model checkpoint (.pth)')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")
    
    # Load Vocab
    vocab_path = ROOT / "data/coco/vocab_small.pth"
    word2idx, idx2word = load_vocab(vocab_path)
    
    # Find checkpoint if not provided
    ckpt_path = None
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        # Try to find the latest checkpoint in out/checkpoints
        ckpt_root = ROOT / "out/checkpoints"
        if ckpt_root.exists():
            # Find most recently modified directory
            dirs = [d for d in ckpt_root.iterdir() if d.is_dir()]
            if dirs:
                latest_dir = max(dirs, key=os.path.getmtime)
                possible_ckpt = latest_dir / "best_model.pth"
                if possible_ckpt.exists():
                    ckpt_path = possible_ckpt
                    print(f">>> Found latest checkpoint: {ckpt_path}")
    
    if not ckpt_path:
        print(">>> ERROR: No checkpoint found. Please provide --ckpt or ensure training saved one.")
        # We continue just to show it runs, but results will be garbage
    
    model = load_model(ckpt_path, len(word2idx), device)
    
    img_path = Path(args.img)
    if not img_path.exists():
        print(f">>> Image not found: {img_path}")
        return
        
    caption = caption_image(model, img_path, word2idx, idx2word, device)
    print(f"\n>>> Generated Caption: {caption}\n")

if __name__ == '__main__':
    main()
