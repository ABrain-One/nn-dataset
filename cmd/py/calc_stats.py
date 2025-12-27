import os
import sys
import json
import argparse
import torch
import pandas as pd

# Add project root to path to allow importing ab modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ab.nn.api import data
from ab.nn.util import NNAnalysis, Const

# Configuration for dataset shapes
# Format: 'dataset_name': {'in_shape': (batch, channels, height, width), 'out_shape': (classes,)}
DATASET_CONFIG = {
    'cifar-10': {
        'in_shape': (1, 3, 32, 32),
        'out_shape': (10,)
    },
    'cifar-100': {
        'in_shape': (1, 3, 32, 32),
        'out_shape': (100,)
    },
    'mnist': {
        'in_shape': (1, 1, 28, 28),
        'out_shape': (10,)
    },
    'fashion-mnist': {
        'in_shape': (1, 1, 28, 28),
        'out_shape': (10,)
    },
    'svhn': {
        'in_shape': (1, 3, 32, 32),
        'out_shape': (10,)
    },
    'utkface': {
        'in_shape': (1, 3, 200, 200),
        'out_shape': (1,)
    },
    'coco': {
        'in_shape': (1, 3, 224, 224),
        'out_shape': (1000,)
    },
    'celeba-gender': {
        'in_shape': (1, 3, 128, 128),
        'out_shape': (2,)
    },
    # Add other datasets as needed
}

def main():
    parser = argparse.ArgumentParser(description="Calculate statistics for LEMUR models.")
    parser.add_argument('--task', type=str, help="Filter by task name")
    parser.add_argument('--dataset', type=str, help="Filter by dataset name")
    parser.add_argument('--nn', type=str, help="Filter by neural network name")
    parser.add_argument('--limit', type=int, default=50, help="Limit the number of models to process")
    args = parser.parse_args()

    print(f"Fetching data with filters: task={args.task}, dataset={args.dataset}, nn={args.nn}, limit={args.limit}")

    try:
        # Fetch data using the API
        df = data(task=args.task, dataset=args.dataset, nn=args.nn, max_rows=args.limit)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    if df is None or df.empty:
        print("No models found matching the criteria.")
        return

    print(f"Found {len(df)} models to process.")

    # Ensure stat directory exists
    if not hasattr(Const, 'stat_dir'):
         print("Error: Const.stat_dir is not defined.")
         return
         
    os.makedirs(Const.stat_dir, exist_ok=True)

    for index, row in df.iterrows():
        if 'prm_id' not in row:
            print(f"Skipping row {index}: 'prm_id' column missing.")
            continue
            
        prm_id = row['prm_id']
        stat_file_path = os.path.join(Const.stat_dir, f"{prm_id}.json")

        # Idempotency check
        if os.path.exists(stat_file_path):
            continue

        dataset_name = row.get('dataset')
        if not dataset_name:
            print(f"Skipping {prm_id}: Dataset name missing.")
            continue
            
        dataset_key = dataset_name.lower()
        if dataset_key not in DATASET_CONFIG:
            print(f"Skipping {prm_id}: Dataset '{dataset_name}' configuration not found.")
            continue

        config = DATASET_CONFIG[dataset_key]
        in_shape = config['in_shape']
        out_shape = config['out_shape']
        
        nn_code = row.get('nn_code')
        prm = row.get('prm')

        if not nn_code:
            print(f"Skipping {prm_id}: nn_code missing.")
            continue
            
        # Parse prm if it's a string
        if isinstance(prm, str):
            try:
                prm = json.loads(prm.replace("'", '"'))
            except json.JSONDecodeError:
                print(f"Skipping {prm_id}: Could not parse prm string.")
                continue
        elif prm is None:
             print(f"Skipping {prm_id}: prm missing.")
             continue

        print(f"Processing model {prm_id}...")

        try:
            # Load model class dynamically
            local_scope = {'torch': torch, 'nn': torch.nn}
            exec(nn_code, local_scope, local_scope)
            
            if 'Net' not in local_scope:
                raise ValueError("Class 'Net' not found in nn_code.")
            
            # Instantiate model
            device = torch.device('cpu') 
            model = local_scope['Net'](in_shape=in_shape, out_shape=out_shape, prm=prm, device=device)
            
            # Analyze
            stats = NNAnalysis.analyze_model_comprehensive(model, nn_code, in_shape)
            
            # Add metadata
            stats['nn'] = row.get('nn')
            stats['prm_id'] = prm_id
            
            # Save to file
            with open(stat_file_path, 'w') as f:
                json.dump(stats, f, indent=4)
                
            print(f"Saved stats for {prm_id}.")

        except Exception as e:
            print(f"Error processing {prm_id}: {e}")
            # Save error state
            error_stats = {
                'nn': row.get('nn'),
                'prm_id': prm_id,
                'error': repr(e)
            }
            with open(stat_file_path, 'w') as f:
                json.dump(error_stats, f, indent=4)

if __name__ == "__main__":
    main()