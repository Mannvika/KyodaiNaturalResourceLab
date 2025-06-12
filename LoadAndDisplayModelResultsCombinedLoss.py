#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import segmentation_models_pytorch as smp
import re
from pathlib import Path

print("Imports successful", flush=True)

# ===================================================================
# --- ⚙️ CONFIGURATION - EDIT THESE VARIABLES ---
# ===================================================================

# Directory containing the .pth model files from your training run.
MODELS_DIR = "training_run_results_combined_loss"

# Root directory of the precomputed data.
DATA_ROOT = "Data"

# Number of random samples from the test set to process for each model.
NUM_SAMPLES = 10

# Directory where the output folders for each model will be saved.
OUTPUT_DIR = "individual_model_results"

# ===================================================================


class PrecomputedNoise2NoiseDataset(Dataset):
    """
    Loads precomputed tensor data (noisy input, clean ground truth) based on a manifest file.
    """
    def __init__(self, manifest_file, root_dir):
        self.root_dir = root_dir
        try:
            self.manifest = pd.read_csv(manifest_file)
        except FileNotFoundError:
            print(f"Error: Manifest file not found at {manifest_file}")
            self.manifest = pd.DataFrame()

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        if idx >= len(self.manifest):
            raise IndexError("Index out of bounds")
        record = self.manifest.iloc[idx]
        noisy1_path = os.path.join(self.root_dir, record['noisy1_path'])
        clean_path = os.path.join(self.root_dir, record['clean_path'])
        time_step = record['time_step']
        config_name = record['config_name']
        try:
            noisy1_tensor = torch.load(noisy1_path)
            clean_tensor = torch.load(clean_path)
            return noisy1_tensor, clean_tensor, record['id'], time_step, config_name
        except FileNotFoundError as e:
            print(f"Error loading file for sample id {record['id']}: {e}")
            dummy_tensor = torch.zeros((1, 1500, 1500), dtype=torch.float)
            return dummy_tensor, dummy_tensor, "error_id", 0.0, "error_config"

def get_model_params_from_path(model_path):
    """
    Parses the model's architecture details from its filename,
    compatible with the new 'CombinedLoss' naming scheme.
    """
    filename = Path(model_path).name
    params = {}

    # Parse Encoder Depth (e.g., "ED_4")
    ed_match = re.search(r'ED_(\d+)', filename)
    if ed_match:
        params['encoder_depth'] = int(ed_match.group(1))
    else:
        raise ValueError(f"Could not parse encoder depth (ED_*) from {filename}")

    # Parse Decoder Channels (e.g., "DC_256_128_64")
    # This regex ensures we only capture the digits and underscores part.
    dc_match = re.search(r'DC_(\d+(?:_\d+)*)', filename)
    if dc_match:
        channels_str = dc_match.group(1)
        params['decoder_channels'] = tuple(map(int, channels_str.split('_')))
    else:
        raise ValueError(f"Could not parse decoder channels (DC_*) from {filename}")

    # Use the filename (without extension) as the model's unique ID
    params['model_name'] = Path(filename).stem
    return params

def main():
    """
    Main execution function to run inference on all models in a directory.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- 1. Load Dataset and Create Test Loader ---
    manifest_path = os.path.join(DATA_ROOT, 'manifest.csv')
    if not os.path.exists(manifest_path):
        print(f"Manifest file {manifest_path} not found. Cannot proceed.")
        return

    full_dataset = PrecomputedNoise2NoiseDataset(manifest_file=manifest_path, root_dir=DATA_ROOT)

    if len(full_dataset) == 0:
        print("Dataset is empty. Cannot create test loader.")
        return

    # Create a reproducible split to get the same test set every time
    total_samples = len(full_dataset)
    indices = list(range(total_samples))
    np.random.seed(42) # Use a fixed seed for consistency
    np.random.shuffle(indices)
    
    train_ratio, val_ratio = 0.7, 0.15
    train_split_idx = int(train_ratio * total_samples)
    val_split_idx = train_split_idx + int(val_ratio * total_samples)
    test_indices = indices[val_split_idx:] if total_samples >= 3 else []

    if not test_indices:
        print("Warning: No test samples found after splitting. Cannot run inference.")
        return

    test_subset = Subset(full_dataset, test_indices)
    # Use shuffle=False to process samples in a consistent order if needed
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)
    print(f"Loaded {len(test_subset)} test samples.")

    # --- 2. Discover and Process Each Model ---
    model_paths = list(Path(MODELS_DIR).glob("*.pth"))
    if not model_paths:
        print(f"No .pth model files found in '{MODELS_DIR}'. Exiting.")
        return

    print(f"\nFound {len(model_paths)} models to process.")

    for model_path in model_paths:
        try:
            params = get_model_params_from_path(str(model_path))
            model_name = params['model_name']
            print(f"\n{'='*20}\nProcessing Model: {model_name}\n{'='*20}")

            # Create a dedicated output directory for this model
            model_output_dir = os.path.join(OUTPUT_DIR, model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            # Initialize and load the model
            model = smp.Unet(
                encoder_name='resnet18',
                encoder_weights=None, # Weights are loaded from the .pth file
                encoder_depth=params['encoder_depth'],
                decoder_channels=params['decoder_channels'],
                in_channels=1,
                classes=1
            )
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()

            # --- 3. Run Inference on Test Samples ---
            print(f"Starting inference on {min(NUM_SAMPLES, len(test_loader))} samples...")
            with torch.no_grad():
                for i, (noisy_tensor, clean_tensor, sample_id, time_step, config_name) in enumerate(test_loader):
                    if i >= NUM_SAMPLES:
                        break

                    config_name_val, time_step_val, sample_id_val = config_name[0], time_step.item(), sample_id[0]
                    print(f"  > Processing Sample {i+1}/{NUM_SAMPLES} (ID: {sample_id_val})")

                    noisy_input = noisy_tensor.to(DEVICE)
                    denoised_output = model(noisy_input).cpu().squeeze().numpy()
                    noisy_np = noisy_tensor.cpu().squeeze().numpy()
                    clean_np = clean_tensor.cpu().squeeze().numpy()

                    # --- 4. Generate and Save Output Image ---
                    vmin = min(noisy_np.min(), clean_np.min(), denoised_output.min())
                    vmax = max(noisy_np.max(), clean_np.max(), denoised_output.max())

                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    figure_title = f"Model: {model_name}\nSample ID: {sample_id_val} | Day: {time_step_val:.2f} | Sim: {config_name_val}"
                    fig.suptitle(figure_title, fontsize=16)

                    # Plot Noisy, Denoised, and Clean images
                    axes[0].imshow(noisy_np, cmap='bwr', vmin=vmin, vmax=vmax)
                    axes[0].set_title("Noisy Input")
                    axes[0].axis('off')

                    axes[1].imshow(denoised_output, cmap='bwr', vmin=vmin, vmax=vmax)
                    axes[1].set_title("Denoised Output")
                    axes[1].axis('off')

                    mappable = axes[2].imshow(clean_np, cmap='bwr', vmin=vmin, vmax=vmax)
                    axes[2].set_title("Clean Ground Truth")
                    axes[2].axis('off')
                    
                    # Add a shared colorbar
                    fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle

                    save_path = os.path.join(model_output_dir, f"sample_{sample_id_val}.png")
                    plt.savefig(save_path)
                    plt.close(fig)

        except Exception as e:
            print(f"An error occurred while processing {model_path.name}: {e}")
            print("Skipping to the next model.")
            continue

    print("\n✅ All models processed. Inference script finished.")


if __name__ == '__main__':
    main()