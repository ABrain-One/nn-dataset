# Training Visualization Guide

## Overview

The `simple_accuracy_stat_viz.py` script is a tool for visualizing training metrics (accuracy) across multiple neural network models during training on the CIFAR-10 dataset.

## File Location

This script reads information from

```
nn-dataset/
├── ab/
│   └── nn/
│       └── stat/
│           └── train/              ← Training statistics are read from here
└── ...
```

and saves graphs in 

```
nn-dataset/cmd/py/simple_stat_viz/
├── nn-training-graphs/            ← Graphs are saved here (auto-created)
```

## What It Does

The script:

1. **Reads training statistics** from JSON files stored in `ab/nn/stat/train/` for each specified model
2. **Extracts accuracy data** per epoch from the JSON files
3. **Generates comparison plots** showing accuracy vs. epochs for all models
4. **Creates multiple visualizations**:
   - Full training plot (all epochs)
   - First 50 epochs plot (if training exceeds 50 epochs)
5. **Prints a summary table** with:
   - Final accuracy
   - Maximum accuracy achieved
   - Epoch at which maximum accuracy occurred
   - Total number of epochs trained
6. **Saves graphs** to the `nn-training-graphs/` directory

## Prerequisites

Ensure you have the required Python packages installed:

```bash
pip install matplotlib
```

and install all the LEMUR project dependencies from root directory of the project:

```bash
pip install -r requirements.txt
```

## How to Use

### Step 1: Configure Model Names

Edit the `MODEL_NAMES` list in the script to specify which models you want to visualize:

```python
MODEL_NAMES = [
    'AlexNet', 
    'MoEv9-AlexNet', 
    'MoEv9-AlexNetv2',
    'MoEv9-AlexNetv3',
    'MoEv9-AlexNetv4'
]
```

### Step 2: Verify Statistics Directory Structure

The script expects training statistics in this format:

```
ab/nn/stat/train/
├── img-classification_cifar-10_acc_AlexNet/
│   ├── 1.json
│   ├── 2.json
│   ├── 3.json
│   └── ...
├── img-classification_cifar-10_acc_MoEv9-AlexNet/
│   ├── 1.json
│   ├── 2.json
│   └── ...
└── ...
```

Each JSON file should contain accuracy data in one of these formats:

**Format 1 (Dictionary):**
```json
{
  "accuracy": 0.8523,
  "loss": 0.4321,
  ...
}
```

**Format 2 (List):**
```json
[
  {
    "accuracy": 0.8523,
    "metric": "acc",
    ...
  }
]
```

### Step 3: Run the Script

From the `nn-dataset` root directory:

```bash
python simple_accuracy_stat_viz.py
```

Or if using a virtual environment:

```bash
# Activate your virtual environment first
source myenv/bin/activate  # Linux/Mac
# or
myenv\Scripts\activate     # Windows

# Then run the script
python simple_accuracy_stat_viz.py
```

## Output

### Generated Files

The script creates graphs in the `nn-training-graphs/` directory:

- `accuracy_vs_epochs_all_models.png` - Complete training visualization
- `accuracy_vs_epochs_all_models_first50.png` - First 50 epochs (if applicable)

### Console Output

Example output:

```
Looking for model directory: img-classification_cifar-10_acc_AlexNet
Model: AlexNet | Highest Accuracy: 0.8734 at epoch 45
Looking for model directory: img-classification_cifar-10_acc_MoEv9-AlexNet
Model: MoEv9-AlexNet | Highest Accuracy: 0.8956 at epoch 48
...

Summary Table:
              model  final_accuracy  epochs  max_accuracy  max_accuracy_epoch
0          AlexNet          0.8734      50        0.8734                  45
1   MoEv9-AlexNet          0.8956      50        0.8956                  48
...
```

## Customization

### Change Statistics Directory Pattern

Modify the `STAT_MODEL_DIR_PATTERN` variable:

```python
STAT_MODEL_DIR_PATTERN = "img-classification_cifar-10_acc_{model}"
```

### Change Output Directory

Modify the `GRAPH_DIR` variable:

```python
GRAPH_DIR = os.path.join(os.path.dirname(__file__), "your-custom-dir")
```

### Adjust Plot Styling

Modify the matplotlib plotting parameters in the `main()` function:

```python
plt.figure(figsize=(10, 6))  # Change figure size
plt.plot(epochs, accuracies, marker='o', markersize=5, linewidth=2, label=model)
```

## Troubleshooting

### No data loaded for model

**Error:** `[DEBUG] No accuracy data loaded for model 'ModelName'. Directory missing or no epoch files.`

**Solution:** 
- Verify the model directory exists in `ab/nn/stat/train/`
- Check that JSON files are named with numbers (e.g., `1.json`, `2.json`)
- Ensure JSON files contain an `accuracy` field

### Model directory not found

**Error:** `[DEBUG] Model directory not found: ab/nn/stat/train/...`

**Solution:**
- Ensure training has been completed and statistics were saved
- Verify the model name matches the directory name exactly
- Check the `STAT_MODEL_DIR_PATTERN` matches your directory structure

### No accuracy found in JSON file

**Error:** `[DEBUG] No accuracy found in /path/to/file.json`

**Solution:**
- Open the JSON file and verify it contains an `accuracy` field
- Check the JSON structure matches one of the supported formats

## Integration with Training Pipeline

This script is designed to work with the training pipeline in `ab/nn/train.py`. After training completes, statistics are automatically saved to the appropriate directory structure, and this visualization script can be run to analyze results.

## License

This script is part of the nn-dataset project and follows the same license terms.
