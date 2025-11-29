# simple_accuracy_stat_viz - Visualize training accuracy statistics from JSON files .feat Yashkumar R L and Harsh R M

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from ab.nn.util.Const import stat_train_dir

# User: specify your model names here (must match stat file names in ab/nn/stat/train/*.json)
MODEL_NAMES = [
    'MoE-0707', 'MoEv2', 'MoEv3', 'MoEv4', 'MoEv5', 'MoEv6', 'MoEv7', 'MoEv8'
    # 'AlexNet', 'MoEv9-AlexNet', 'MoEv9-AlexNetv2','MoEv9-AlexNetv3','MoEv9-AlexNetv4'
    # 'MoE-hetero4-Alex-Dense-Air-Bag', 'AlexNet','DenseNet','AirNext','BagNet'
]

# Subdirectory pattern for each model's stats
STAT_MODEL_DIR_PATTERN = "img-classification_cifar-10_acc_{model}"

# Directory to save graphs (parallel to this script)
GRAPH_DIR = os.path.join(os.path.dirname(__file__), "nn-training-graphs")



def load_epoch_accuracy(model_name):
    """
    Loads all epoch JSON files for a given model and returns lists of epoch numbers and accuracy values.
    """
    model_dir = os.path.join(stat_train_dir, STAT_MODEL_DIR_PATTERN.format(model=model_name))
    if not os.path.isdir(model_dir):
        print(f"[DEBUG] Model directory not found: {model_dir}")
        return [], []
    epoch_files = []
    for fname in os.listdir(model_dir):
        if fname.endswith('.json') and fname[:-5].isdigit():
            epoch_files.append((int(fname[:-5]), fname))
    epoch_files.sort()
    if not epoch_files:
        print(f"[DEBUG] No epoch JSON files found in {model_dir}")
        return [], []
    epochs = []
    accuracies = []
    for epoch, fname in epoch_files:
        fpath = os.path.join(model_dir, fname)
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            acc = None
            if isinstance(data, dict):
                acc = data.get('accuracy', None)
            elif isinstance(data, list):
                # Try to find accuracy in a list of dicts
                for entry in data:
                    if isinstance(entry, dict) and 'accuracy' in entry:
                        acc = entry['accuracy']
                        break
            if acc is not None:
                epochs.append(epoch)
                accuracies.append(acc)
            else:
                print(f"[DEBUG] No accuracy found in {fpath}")
        except Exception as e:
            print(f"[DEBUG] Failed to load {fpath}: {e}")
    return epochs, accuracies



def main():
    if not os.path.exists(GRAPH_DIR):
        os.makedirs(GRAPH_DIR)

    # Gather all model data for plotting
    all_model_data = []
    max_epochs = 0
    for model in MODEL_NAMES:
        print(f"Looking for model directory: {STAT_MODEL_DIR_PATTERN.format(model=model)}")
        epochs, accuracies = load_epoch_accuracy(model)
        if not epochs or not accuracies:
            print(f"[DEBUG] No accuracy data loaded for model '{model}'. Directory missing or no epoch files.")
            continue
        max_acc = max(accuracies)
        print(f"Model: {model} | Highest Accuracy: {max_acc:.4f} at epoch {epochs[accuracies.index(max_acc)]}")
        all_model_data.append((model, epochs, accuracies))
        max_epochs = max(max_epochs, max(epochs))
    # Plot all epochs
    if all_model_data:
        plt.figure(figsize=(10, 6))
        for model, epochs, accuracies in all_model_data:
            plt.plot(epochs, accuracies, marker='o', markersize=5, linewidth=2, label=model)
        plt.title('Accuracy vs Epochs for All Models', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Neural Network')
        plt.tight_layout()
        graph_path = os.path.join(GRAPH_DIR, "accuracy_vs_epochs_all_models.png")
        plt.savefig(graph_path)
        print(f"Saved: {graph_path}")
        plt.show()
        # If any model has >50 epochs, plot only first 50 epochs as well
        if max_epochs > 50:
            plt.figure(figsize=(10, 6))
            for model, epochs, accuracies in all_model_data:
                # Only plot up to epoch 50
                filtered = [(e, a) for e, a in zip(epochs, accuracies) if e <= 50]
                if filtered:
                    f_epochs, f_accuracies = zip(*filtered)
                    plt.plot(f_epochs, f_accuracies, marker='o', markersize=5, linewidth=2, label=model)
            plt.title('Accuracy vs Epochs (First 50 Epochs)', fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Accuracy', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title='Neural Network')
            plt.tight_layout()
            graph_path_50 = os.path.join(GRAPH_DIR, "accuracy_vs_epochs_all_models_first50.png")
            plt.savefig(graph_path_50)
            print(f"Saved: {graph_path_50}")
            plt.show()
        # Print summary table
        summary_rows = []
        for model, epochs, accuracies in all_model_data:
            summary_rows.append({
                "model": model,
                "final_accuracy": accuracies[-1] if accuracies else None,
                "epochs": len(epochs),
                "max_accuracy": max(accuracies) if accuracies else None,
                "max_accuracy_epoch": epochs[accuracies.index(max(accuracies))] if accuracies else None
            })
        df = pd.DataFrame(summary_rows)
        print("\nSummary Table:")
        print(df)
    else:
        print("No valid accuracy data to plot.")

if __name__ == "__main__":
    main()
