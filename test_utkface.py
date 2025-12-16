from ab.nn.loader.utkface import loader
from ab.nn.transform.norm import transform

print("Loading UTKFace dataset...")
out, acc, train, val, test = loader(transform, 'age-regression')

print(f"\nDataset loaded successfully!")
print(f"Output shape: {out}")
print(f"Min accuracy: {acc}")
print(f"Train size: {len(train)}")
print(f"Validation size: {len(val)}")
print(f"Test size: {len(test)}")

print(f"\nTesting sample...")
x, y = train[0]
print(f"Image shape: {x.shape}")
print(f"Age: {y.item()}")
print("\n✓ UTKFace loader working!")
