#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

# Directory containing the model .pth files
MODEL_ROOT = r'wideresults/'

# Root directory for the dataset
DATA_ROOT = r"datasets/TopographicData"

# Directory to save the output images
OUTPUT_DIR = "IndividualModelResults"

# Number of random test samples to process
NUM_SAMPLES = 5

# ===================================================================

# Automatically find all model paths in the specified root directory
MODEL_PATHS = []
if os.path.isdir(MODEL_ROOT):
    for file in os.listdir(MODEL_ROOT):
        if file.startswith('best_model_') and file.endswith('.pth'):
            MODEL_PATHS.append(os.path.join(MODEL_ROOT, file))
else:
    print(f"Warning: MODEL_ROOT directory not found at '{MODEL_ROOT}'")


class PrecomputedNoise2NoiseDataset(Dataset):
    """
    Dataset class for loading precomputed tensor data based on a manifest file.
    """
    def __init__(self, manifest_file, root_dir, for_training=True):
        self.root_dir = Path(root_dir)
        self.for_training = for_training
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
        
        noisy1_path = self.root_dir / record['noisy1_path']
        clean_path = self.root_dir / record['clean_path']

        try:
            noisy1_tensor = torch.load(noisy1_path, weights_only=True)
            clean_tensor = torch.load(clean_path, weights_only=True)

            if self.for_training:
                noisy2_path = self.root_dir / record['noisy2_path']
                noisy2_tensor = torch.load(noisy2_path, weights_only=True)
                return noisy1_tensor, noisy2_tensor, clean_tensor
            else:
                # Return metadata along with tensors for testing/inference
                return noisy1_tensor, clean_tensor, record['id'], record['time_step'], record['config_name']

        except FileNotFoundError as e:
            # Return dummy data if a file is not found to prevent crashing the loader
            print(f"Error loading file for sample id {record['id']}: {e}")
            dummy_tensor = torch.zeros((1, 1500, 1500), dtype=torch.float)
            if self.for_training:
                return dummy_tensor, dummy_tensor, dummy_tensor
            else:
                return dummy_tensor, dummy_tensor, "error_id", 0.0, "error_config"


def get_model_params_from_path(model_path):
    """
    Parses model hyperparameters (encoder depth, decoder channels) from a filename.
    """
    filename = Path(model_path).name
    params = {}
    
    # Extract encoder depth
    ed_match = re.search(r'ED_(\d+)', filename)
    if ed_match: 
        params['encoder_depth'] = int(ed_match.group(1))
    else: 
        raise ValueError(f"Could not parse encoder depth (ED_*) from {filename}")
    
    # Extract decoder channels
    dc_match = re.search(r'DC_((?:\d+_?)+)', filename)
    if dc_match:
        channels_str = dc_match.group(1).strip('_')
        params['decoder_channels'] = tuple(map(int, channels_str.split('_')))
    else: 
        raise ValueError(f"Could not parse decoder channels (DC_*) from {filename}")
    
    # Create a unique ID for the model configuration
    params['combo_id'] = filename.replace('best_model_', '').replace('.pth', '')
    return params

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- DATA LOADING ---
    manifest_path = Path(r'data_splits') / "test_manifest.csv"
    if not manifest_path.exists():
        print(f"Manifest file {manifest_path} not found. Cannot proceed.")
        return
    
    test_dataset = PrecomputedNoise2NoiseDataset(manifest_file=manifest_path, root_dir=DATA_ROOT, for_training=False)

    if len(test_dataset) == 0:
        print("Test dataset is empty. Cannot create test loader.")
        return
        
    # Load the entire test set and shuffle to get random samples
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    print(f"Loaded {len(test_dataset)} test samples from {manifest_path}.")

    # --- MODEL LOADING ---
    models, model_infos = [], []
    for path in MODEL_PATHS:
        if not os.path.exists(path):
            print(f"Warning: Model path not found, skipping: {path}")
            continue
        try:
            print(f"\nLoading model from: {path}")
            params = get_model_params_from_path(path)
            
            model = smp.Unet(
                encoder_name='resnet34', 
                encoder_weights=None, 
                encoder_depth=params['encoder_depth'], 
                decoder_channels=params['decoder_channels'], 
                in_channels=1, 
                classes=1
            )
            
            model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            model.to(DEVICE)
            model.eval()
            
            models.append(model)
            model_infos.append(params)

        except Exception as e:
            print(f"Error loading model {path}: {e}")
            
    if not models:
        print("No models were successfully loaded. Exiting.")
        return

    # --- INFERENCE AND PLOTTING LOOP ---
    print(f"\nStarting inference on {min(NUM_SAMPLES, len(test_loader))} random samples...")
    processed_samples = 0
    with torch.no_grad():
        for noisy_tensor, clean_tensor, sample_id, time_step, config_name in test_loader:
            if processed_samples >= NUM_SAMPLES:
                break

            # This is the corrected line
            if config_name[0] == "error_config":
                print("Skipping sample due to a data loading error.")
                continue

            config_name_val = config_name[0]
            time_step_val = time_step.item()
            sample_id_val = sample_id.item()
            
            print(f"Processing Sample {processed_samples + 1}/{NUM_SAMPLES} (ID: {sample_id_val}, Day: {time_step_val}, Sim: {config_name_val})")
            
            noisy_input = noisy_tensor.to(DEVICE)
            noisy_np = noisy_tensor.cpu().squeeze().numpy()
            clean_np = clean_tensor.cpu().squeeze().numpy()

            # Generate all denoised outputs for the current sample
            denoised_outputs_np = [model(noisy_input).cpu().squeeze().numpy() for model in models]
            
            # *** MODIFICATION START: Loop through each model to create individual plots ***
            for i, model in enumerate(models):
                denoised_np = denoised_outputs_np[i]
                model_info = model_infos[i]

                # Create a 3-panel figure for each model
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle(f"Day: {time_step_val:.2f} | Sim: {config_name_val} | Sample ID: {sample_id_val}", fontsize=16)

                # Dynamically calculate color limits for this specific triplet for best contrast
                vmin = min(noisy_np.min(), clean_np.min(), denoised_np.min())
                vmax = max(noisy_np.max(), clean_np.max(), denoised_np.max())

                # Plot 1: Noisy Input
                axes[0].imshow(noisy_np, cmap='bwr', vmin=vmin, vmax=vmax)
                axes[0].set_title("Noisy Input")
                axes[0].axis('off')
                
                # Plot 2: Denoised Output (from the current model)
                model_title = model_info['combo_id'].replace('_', ' ')
                axes[1].set_title(f"Denoised: {model_title}", fontsize=12)
                mappable = axes[1].imshow(denoised_np, cmap='bwr', vmin=vmin, vmax=vmax)
                axes[1].axis('off')
                
                # Plot 3: Clean Ground Truth
                axes[2].imshow(clean_np, cmap='bwr', vmin=vmin, vmax=vmax)
                axes[2].set_title("Clean Ground Truth")
                axes[2].axis('off')

                # Add a single colorbar for all subplots
                fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.75, pad=0.02)
                
                plt.tight_layout(rect=[0, 0, 1, 0.94]) # Adjust rect to make space for suptitle
                
                # Define a unique filename for each individual plot
                save_path = Path(OUTPUT_DIR) / f"sample_{sample_id_val}_model_{model_info['combo_id']}.png"
                plt.savefig(save_path)
                plt.close(fig)
                print(f"  -> Saved individual plot to {save_path}")
            # *** MODIFICATION END ***
            
            processed_samples += 1

    print("\nInference and individual plotting script finished.")

if __name__ == "__main__":
    main()