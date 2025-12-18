"""
Test script to download and explore the IMDB-Wiki dataset
"""
import torch
import torchvision.transforms as transforms
from ab.nn.loader.imdb_wiki import loader, __norm_mean, __norm_dev
import matplotlib.pyplot as plt


def simple_transform(norm):
    """
    Simple transform function compatible with the loader pattern.
    :param norm: Tuple of (mean, std) normalization values
    :return: Composed transform
    """
    mean, std = norm
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def main():
    print("=" * 60)
    print("IMDB-Wiki Dataset Loader Test")
    print("=" * 60)
    
    print("\n[1/4] Loading dataset (this may take a while on first run)...")
    out_shape, min_acc, train_ds, test_ds = loader(simple_transform, task="age-regression")
    
    print(f"\n[2/4] Dataset loaded successfully!")
    print(f"  - Output shape: {out_shape}")
    print(f"  - Minimum accuracy threshold: {min_acc}")
    print(f"  - Train samples: {len(train_ds):,}")
    print(f"  - Test samples: {len(test_ds):,}")
    
    # Sample a few data points
    print(f"\n[3/4] Sampling data from training set...")
    for i in range(min(5, len(train_ds))):
        x, y = train_ds[i]
        print(f"  Sample {i}: Image shape={x.shape}, Age={y.item():.1f} years")
    
    # Show age distribution
    print(f"\n[4/4] Analyzing age distribution (sampling 1000 examples)...")
    sample_size = min(1000, len(train_ds))
    ages = []
    for i in range(sample_size):
        _, y = train_ds[i]
        ages.append(y.item())
    
    ages = torch.tensor(ages)
    print(f"  - Mean age: {ages.mean():.2f} years")
    print(f"  - Min age: {ages.min():.2f} years")
    print(f"  - Max age: {ages.max():.2f} years")
    print(f"  - Std dev: {ages.std():.2f} years")
    
    print("\n" + "=" * 60)
    print("Dataset cache location:")
    print("  C:\\Users\\arunk\\.cache\\huggingface\\datasets")
    print("=" * 60)
    
    # Visualize a few samples
    print("\n[Optional] Displaying 4 sample images...")
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for i in range(4):
        x, y = train_ds[i]
        img = x.permute(1, 2, 0).numpy()  # CHW -> HWC
        axes[i].imshow(img)
        axes[i].set_title(f"Age: {y.item():.1f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('imdb_wiki_samples.png')
    print("  Saved visualization to: imdb_wiki_samples.png")
    
    print("\n✓ Test completed successfully!")

if __name__ == "__main__":
    main()
