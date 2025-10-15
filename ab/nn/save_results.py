# File: save_results.py
# Description: This simplified version loads the single 'best_model.pth'
#              checkpoint created by the 'Save on Improvement' training process.

import torch
import os
import argparse
import importlib
from ab.nn.util.Util import out_dir

# --- Make sure this import path is correct for the  project ---
from ab.nn.util.Util import export_torch_weights


def save_best_model(model_name):
    """
    Loads the single best checkpoint saved during training and exports the
    final weights and sample images.
    """
    print(f"--- Starting Post-Training Save Process for model: {model_name} ---")

    # --- Configuration ---
    OUTPUT_FOLDER = f"final_results/{model_name}"
    PROMPTS_FOR_GENERATION = [
        "a photo of a red sports car on a highway",
        "a vintage blue pickup truck parked on a city street",
        "a black convertible with the top down at sunset"
    ]

    # --- THE FIX: Directly load the single best checkpoint ---
    checkpoint_dir = out_dir / 'checkpoints' /  model_name
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

    if not os.path.exists(best_checkpoint_path):
        print(f"Error: The best model checkpoint was not found at {best_checkpoint_path}")
        print("Please ensure your training has run long enough to save at least one 'best_model.pth' file.")
        return

    print(f"Loading best checkpoint from: {best_checkpoint_path}")

    try:
        # Dynamically import the correct model class
        model_module_path = f"ab.nn.nn.{model_name}"
        model_module = importlib.import_module(model_module_path)
        Net = model_module.Net

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy_in_shape = (1, 3, 256, 256)
        model = Net(in_shape=dummy_in_shape, out_shape=None, prm={}, device=device).to(device)

        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        model.eval()
        print("Model weights loaded successfully.")

        # --- Save the Best Final Weights and Images ---
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        final_weights_path = os.path.join(OUTPUT_FOLDER, f"{model_name}_best_weights.pth")
        export_torch_weights(model, final_weights_path)

        print(f"\nGenerating {len(PROMPTS_FOR_GENERATION)} final sample images...")
        final_images = model.generate(PROMPTS_FOR_GENERATION)

        for i, img in enumerate(final_images):
            prompt_str = PROMPTS_FOR_GENERATION[i].replace(" ", "_")[:40]
            image_path = os.path.join(OUTPUT_FOLDER, f"final_image_{i + 1}_{prompt_str}.png")
            img.save(image_path)
            print(f"Saved final image to {image_path}")

        print("\n--- Post-Training Process Finished Successfully ---")

    except (AttributeError, ImportError) as e:
        print(f"\n--- SCRIPT FAILED ---")
        print(f"Error: Could not import or find the 'Net' class for model '{model_name}'.")
        print("Please ensure the model file is named correctly and contains a 'Net' class.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\n--- SCRIPT FAILED ---")
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save the best performing model weights and generate sample images.")
    parser.add_argument("--model", type=str, required=True,
                        help="The name of the model to process (e.g., ConditionalVAE2, ConditionalVAE3).")
    args = parser.parse_args()

    save_best_model(args.model)
