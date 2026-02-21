"""Quick smoke-test for the UTKFace loader."""
from ab.nn.loader.utkface import loader
from ab.nn.transform.norm import transform

print("Loading UTKFace dataset...")
out, acc, train_ds, test_ds = loader(transform, 'age-regression')

print(f"\nDataset loaded successfully!")
print(f"Output shape : {out}")
print(f"Min accuracy : {acc}")
print(f"Train size   : {len(train_ds)}")
print(f"Test size    : {len(test_ds)}")

print(f"\nSample check...")
x, y = train_ds[0]
print(f"Image shape  : {x.shape}")
print(f"Age label    : {y.item():.1f}")
print("\nUTKFace loader OK")
