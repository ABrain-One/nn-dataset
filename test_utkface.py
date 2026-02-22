# Smoke-test: verify UTKFace loader returns correct shapes and splits
from ab.nn.loader.utkface import loader
from ab.nn.transform.norm import transform

out, acc, train_ds, test_ds = loader(transform, 'age-regression')
print(f"Output: {out}, Min acc: {acc}, Train: {len(train_ds)}, Test: {len(test_ds)}")
x, y = train_ds[0]
print(f"Image: {x.shape}, Age: {y.item():.1f}")
print("UTKFace loader OK")
