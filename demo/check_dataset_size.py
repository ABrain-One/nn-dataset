"""
Quick script to check IMDB-Wiki dataset size before downloading
"""
from datasets import load_dataset_builder

dataset_name = "systemk/imdb-wiki"
config = "default"

print("Checking dataset info...")
print("=" * 60)

try:
    # Get dataset info without downloading
    ds_builder = load_dataset_builder(dataset_name, config)
    
    print(f"Dataset: {dataset_name}")
    print(f"Config: {config}")
    print("\nDataset Info:")
    print(f"  Description: {ds_builder.info.description[:200]}...")
    
    # Get download size
    if hasattr(ds_builder.info, 'download_size') and ds_builder.info.download_size:
        size_gb = ds_builder.info.download_size / (1024**3)
        size_mb = ds_builder.info.download_size / (1024**2)
        print(f"\n  Download size: {size_gb:.2f} GB ({size_mb:.1f} MB)")
    
    # Get dataset size after processing
    if hasattr(ds_builder.info, 'dataset_size') and ds_builder.info.dataset_size:
        size_gb = ds_builder.info.dataset_size / (1024**3)
        size_mb = ds_builder.info.dataset_size / (1024**2)
        print(f"  Dataset size (processed): {size_gb:.2f} GB ({size_mb:.1f} MB)")
    
    # Show splits
    if hasattr(ds_builder.info, 'splits') and ds_builder.info.splits:
        print("\n  Splits:")
        for split_name, split_info in ds_builder.info.splits.items():
            print(f"    - {split_name}: {split_info.num_examples:,} examples")
    
    print("\n" + "=" * 60)
    print("✓ Check complete - no data downloaded yet!")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nNote: Some datasets don't expose size info until download.")
    print("Estimated size for IMDB-Wiki: 5-20 GB depending on config")

