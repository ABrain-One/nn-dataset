import argparse
import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import sys
import os
import csv
import json
from tqdm import tqdm

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
    in_shape = (1, 3, 224, 224)
    out_shape = (vocab_size,)
    prm = {'lr': 0.0001} 
    
    model = Net(in_shape, out_shape, prm, device)
    
    if ckpt_path and ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        print(">>> Model weights loaded.")
    else:
        print(">>> WARNING: Checkpoint not found. Using random weights.")
    
    model.to(device)
    model.eval()
    return model

def generate_captions_csv(model, img_dir, output_csv, word2idx, idx2word, device, limit=None):
    print(f">>> Generating captions for images in {img_dir}")
    
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    model.word2idx = word2idx
    model.idx2word = idx2word
    
    img_paths = sorted(list(Path(img_dir).glob("*.jpg")))
    if limit:
        img_paths = img_paths[:limit]
        
    results = []
    
    print(f">>> Found {len(img_paths)} images. Processing...")
    
    with torch.no_grad():
        for img_path in tqdm(img_paths):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).to(device)
                
                caption = model.beam_search_generate(img_tensor, word2idx, idx2word)
                results.append({'image': img_path.name, 'caption': caption})
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                results.append({'image': img_path.name, 'caption': "ERROR"})

    print(f">>> Saving results to {output_csv}")
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['image', 'caption']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print(">>> Done!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='data/demo_images', help='Directory of images')
    parser.add_argument('--out', type=str, default='captions.csv', help='Output CSV file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")
    
    # Load Vocab
    vocab_path = ROOT / "data/coco/vocab_small.pth"
    word2idx, idx2word = load_vocab(vocab_path)
    
    # Find checkpoint
    ckpt_root = ROOT / "out/checkpoints"
    ckpt_path = None
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
        print(">>> ERROR: No checkpoint found.")
        return
    
    model = load_model(ckpt_path, len(word2idx), device)
    
    generate_captions_csv(model, args.img_dir, args.out, word2idx, idx2word, device, args.limit)

if __name__ == '__main__':
    main()
