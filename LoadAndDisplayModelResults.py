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

# List of paths to the model files you want to compare.
# Add or remove model paths from this list.
MODEL_PATHS = [
    "training_run_results/best_model_LR_0.0001_WD_1e-06_ED_5_DC_128_64_32_16_8_LOSS_MSE.pth",
    "training_run_results/best_model_LR_0.001_WD_1e-05_ED_3_DC_256_128_64_LOSS_L1.pth",
]
DATA_ROOT = "Data"
NUM_SAMPLES = 5
OUTPUT_DIR = "inference_results"

class PrecomputedNoise2NoiseDataset(Dataset):
    def __init__(self, manifest_file, root_dir):
        self.root_dir = root_dir
        try:
            self.manifest = pd.read_csv(manifest_file)
        except FileNotFoundError:
            print(f"Error: Manifest file not found at {manifest_file}")
            self.manifest = pd.DataFrame() # Empty dataframe

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        if idx >= len(self.manifest):
            raise IndexError("Index out of bounds")
            
        record = self.manifest.iloc[idx]
        
        noisy1_path = os.path.join(self.root_dir, record['noisy1_path'])
        clean_path = os.path.join(self.root_dir, record['clean_path'])

        try:
            noisy1_tensor = torch.load(noisy1_path)
            clean_tensor = torch.load(clean_path)
            # Return the noisy input, the clean target, and the sample ID for file naming
            return noisy1_tensor, clean_tensor, record['id']
        except FileNotFoundError as e:
            print(f"Error loading file for sample id {record['id']}: {e}")
            dummy_tensor = torch.zeros((1, 1500, 1500), dtype=torch.float)
            return dummy_tensor, dummy_tensor, "error_id"

def get_model_params_from_path(model_path):
    """
    Parses a model filename to extract architecture hyperparameters.
    Example filename: best_model_LR_0.001_WD_1e-05_ED_5_DC_128_64_32_16_8_LOSS_MSE.pth
    """
    filename = Path(model_path).name
    params = {}
    
    # Extract Encoder Depth (ED)
    ed_match = re.search(r'ED_(\d+)', filename)
    if ed_match:
        params['encoder_depth'] = int(ed_match.group(1))
    else:
        raise ValueError(f"Could not parse encoder depth (ED_*) from {filename}")

    # Extract Decoder Channels (DC)
    dc_match = re.search(r'DC_((?:\d+_?)+)', filename)
    if dc_match:
        channels_str = dc_match.group(1).strip('_')
        params['decoder_channels'] = tuple(map(int, channels_str.split('_')))
    else:
        raise ValueError(f"Could not parse decoder channels (DC_*) from {filename}")

    # Extract a readable name for plot titles
    params['combo_id'] = filename.replace('best_model_', '').replace('.pth', '')
    
    return params

def main():
    # --- Basic Setup ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Load Data ---
    manifest_path = os.path.join(DATA_ROOT, 'manifest.csv')
    if not os.path.exists(manifest_path):
        print(f"Manifest file {manifest_path} not found. Cannot proceed.")
        return
        
    full_dataset = PrecomputedNoise2NoiseDataset(manifest_file=manifest_path, root_dir=DATA_ROOT)

    # Recreate the exact same test split as in training for consistency
    if len(full_dataset) > 0:
        total_samples = len(full_dataset)
        indices = list(range(total_samples))
        # Use a fixed seed for a reproducible train/val/test split
        np.random.seed(42) 
        np.random.shuffle(indices)

        train_ratio = 0.7
        val_ratio = 0.15
        
        if total_samples < 3:
            test_indices = []
        else:
            train_split_idx = int(train_ratio * total_samples)
            val_split_idx = train_split_idx + int(val_ratio * total_samples)
            test_indices = indices[val_split_idx:]
        
        if not test_indices:
            print("Warning: No test samples found based on the split ratios. Cannot run inference.")
            return

        test_subset = Subset(full_dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=True) # Shuffle to get random samples
        print(f"Loaded {len(test_subset)} test samples.")
    else:
        print("Dataset is empty. Cannot create test loader.")
        return

    # --- Load Models ---
    models = []
    model_infos = []
    for path in MODEL_PATHS:
        if not os.path.exists(path):
            print(f"Warning: Model path not found, skipping: {path}")
            continue
        try:
            print(f"\nLoading model from: {path}")
            params = get_model_params_from_path(path)
            model_infos.append(params)
            
            model = smp.Unet(
                encoder_name='resnet18',
                encoder_weights=None, # Not needed for inference when loading a state_dict
                encoder_depth=params['encoder_depth'],
                decoder_channels=params['decoder_channels'],
                in_channels=1,
                classes=1,
            )
            
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval() # Set model to evaluation mode
            models.append(model)
            print(f"Successfully loaded model with ED={params['encoder_depth']} and DC={params['decoder_channels']}")

        except Exception as e:
            print(f"Error loading model {path}: {e}")
            
    if not models:
        print("No models were successfully loaded. Exiting.")
        return

    # --- Inference and Visualization Loop ---
    print(f"\nStarting inference on {min(NUM_SAMPLES, len(test_loader))} samples...")
    with torch.no_grad():
        for i, (noisy_tensor, clean_tensor, sample_id) in enumerate(test_loader):
            if i >= NUM_SAMPLES:
                break

            print(f"Processing sample {i+1}/{NUM_SAMPLES} (ID: {sample_id[0]})")
            
            noisy_input = noisy_tensor.to(DEVICE)
            
            denoised_outputs = []
            for model in models:
                denoised_tensor = model(noisy_input)
                denoised_np = denoised_tensor.cpu().squeeze().numpy()
                denoised_outputs.append(denoised_np)

            noisy_np = noisy_tensor.cpu().squeeze().numpy()
            clean_np = clean_tensor.cpu().squeeze().numpy()

            # --- Plotting ---
            num_models = len(models)
            fig, axes = plt.subplots(1, 2 + num_models, figsize=(6 * (2 + num_models), 6))
            
            axes[0].imshow(noisy_np, cmap='gray')
            axes[0].set_title("Noisy Input")
            axes[0].axis('off')

            for j, denoised_img in enumerate(denoised_outputs):
                ax = axes[j + 1]
                title_text = model_infos[j]['combo_id'].replace('_', ' ')
                ax.set_title(f"Denoised: {title_text}", fontsize=8)
                ax.imshow(denoised_img, cmap='gray')
                ax.axis('off')
                
            axes[-1].imshow(clean_np, cmap='gray')
            axes[-1].set_title("Clean Ground Truth")
            axes[-1].axis('off')

            plt.tight_layout()
            save_path = os.path.join(OUTPUT_DIR, f"comparison_sample_{sample_id[0]}.png")
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved comparison plot to {save_path}")

    print("\nInference and display script finished.")


if __name__ == '__main__':
    main()