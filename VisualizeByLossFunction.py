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
from collections import defaultdict

print("Imports successful", flush=True)

# ===================================================================
# --- ⚙️ CONFIGURATION - EDIT THESE VARIABLES ---
# ===================================================================



MODEL_PATHS = []

MODEL_ROOT = r'initial_width_results_1e4/'


# get all model paths from the current directory
for file in os.listdir(MODEL_ROOT):
    if file.startswith('best_model_') and file.endswith('.pth'):
        file = os.path.join(MODEL_ROOT, file)
        MODEL_PATHS.append(file)


DATA_ROOT = r"datasets/TopographicData"
NUM_SAMPLES = 40
OUTPUT_DIR = "IdenticalTestResults1e-4_ByLoss"

# ===================================================================


class PrecomputedNoise2NoiseDataset(Dataset):
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
        noisy2_path = self.root_dir / record['noisy2_path']
        clean_path = self.root_dir / record['clean_path']

        try:
            noisy1_tensor = torch.load(noisy1_path, weights_only=True)
            clean_tensor = torch.load(clean_path, weights_only=True)

            if self.for_training:
                noisy2_tensor = torch.load(noisy2_path, weights_only=True)
                return noisy1_tensor, noisy2_tensor, clean_tensor
            else:
                return noisy1_tensor, clean_tensor, record['id'], record['time_step'], record['config_name']

        except FileNotFoundError as e:
            print(f"Error loading file for sample id {record['id']}: {e}")
            dummy_tensor = torch.zeros((1, 1500, 1500), dtype=torch.float)
            if self.for_training:
                return dummy_tensor, dummy_tensor, dummy_tensor
            else:
                return dummy_tensor, dummy_tensor, "error_id", 0.0, "error_config"


def get_model_params_from_path(model_path):
    """Parses model parameters and complex loss functions from the filename."""
    filename = Path(model_path).name
    params = {}
    
    # --- MODIFICATION: Regex now captures combined losses (e.g., MSE+SSIM) ---
    loss_match = re.search(r'LOSS_([\w\+]+)', filename)
    if loss_match:
        params['loss_function'] = loss_match.group(1)
    else:
        params['loss_function'] = 'Unknown'

    ed_match = re.search(r'ED_(\d+)', filename)
    if ed_match: params['encoder_depth'] = int(ed_match.group(1))
    else: raise ValueError(f"Could not parse encoder depth (ED_*) from {filename}")
    
    dc_match = re.search(r'DC_((?:\d+_?)+)', filename)
    if dc_match:
        channels_str = dc_match.group(1).strip('_')
        params['decoder_channels'] = tuple(map(int, channels_str.split('_')))
    else: raise ValueError(f"Could not parse decoder channels (DC_*) from {filename}")
    
    params['combo_id'] = filename.replace('best_model_', '').replace('.pth', '')
    return params

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    manifest_path = Path(r'data_splits') / "test_manifest.csv"
    if not manifest_path.exists():
        print(f"Manifest file {manifest_path} not found. Cannot proceed.")
        return
    
    test_dataset = PrecomputedNoise2NoiseDataset(manifest_file=manifest_path, root_dir=DATA_ROOT, for_training=False)

    if len(test_dataset) > 0:
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        print(f"Loaded {len(test_dataset)} test samples from {manifest_path}.")
    else:
        print("Test dataset is empty. Cannot create test loader.")
        return
        
    models_by_loss = defaultdict(list)
    
    for path in MODEL_PATHS:
        if not os.path.exists(path):
            print(f"Warning: Model path not found, skipping: {path}")
            continue
        try:
            print(f"Loading model from: {path}")
            params = get_model_params_from_path(path)
            loss_function = params['loss_function']
            
            model = smp.Unet(
                encoder_name='resnet18', 
                encoder_weights=None, 
                encoder_depth=params['encoder_depth'], 
                decoder_channels=params['decoder_channels'], 
                in_channels=1, 
                classes=1
            )
            
            model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            model.to(DEVICE)
            model.eval()
            
            models_by_loss[loss_function].append({'model': model, 'params': params})
            print(f"  > Successfully loaded and assigned to group '{loss_function}'.")
            
        except Exception as e:
            print(f"Error loading model {path}: {e}")
            
    if not models_by_loss:
        print("No models were successfully loaded. Exiting.")
        return
        
    print("\n--- Model Groups ---")
    for loss, group in models_by_loss.items():
        print(f"Loss: {loss}, Models: {len(group)}")

    print(f"\nStarting inference on {min(NUM_SAMPLES, len(test_loader))} samples...")
    processed_samples = 0
    with torch.no_grad():
        for noisy_tensor, clean_tensor, sample_id, time_step, config_name in test_loader:
            if processed_samples >= NUM_SAMPLES: break

            if isinstance(sample_id[0], str) and sample_id[0] == "error_id":
                print("Skipping sample due to a data loading error.")
                continue

            config_name_val, time_step_val, sample_id_val = config_name[0], time_step.item(), sample_id.item()
            print(f"\nProcessing Sample {processed_samples + 1}/{NUM_SAMPLES} (ID: {sample_id_val}, Day: {time_step_val}, Sim: {config_name_val})")
            
            noisy_input = noisy_tensor.to(DEVICE)
            noisy_np = noisy_tensor.cpu().squeeze().numpy()
            clean_np = clean_tensor.cpu().squeeze().numpy()

            for loss_func, model_group in models_by_loss.items():
                
                # --- MODIFICATION: Create a separate sub-folder for each loss function ---
                loss_output_dir = Path(OUTPUT_DIR) / loss_func
                os.makedirs(loss_output_dir, exist_ok=True)
                print(f"  > Generating plot for loss group '{loss_func}'")
                
                denoised_outputs_np = [
                    item['model'](noisy_input).cpu().squeeze().numpy() for item in model_group
                ]
                model_infos_group = [item['params'] for item in model_group]

                all_images_for_plot = [noisy_np, clean_np] + denoised_outputs_np
                vmin = min(img.min() for img in all_images_for_plot)
                vmax = max(img.max() for img in all_images_for_plot)

                num_plots = 2 + len(model_group)
                fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6))
                
                fig.suptitle(f'Loss: {loss_func} | Day: {time_step_val:.2f} | Sim: {config_name_val}', fontsize=20)
                
                mappable = None
                
                axes[0].imshow(noisy_np, cmap='bwr', vmin=vmin, vmax=vmax)
                axes[0].set_title("Noisy Input")
                axes[0].axis('off')

                for j, (denoised_img, info) in enumerate(zip(denoised_outputs_np, model_infos_group), 1):
                    ax = axes[j]
                    title_text = info['combo_id'].replace(f'LOSS_{loss_func}_', '').replace('_', ' ')
                    ax.set_title(f"Denoised: {title_text}", fontsize=8)
                    mappable = ax.imshow(denoised_img, cmap='bwr', vmin=vmin, vmax=vmax)
                    ax.axis('off')
                
                axes[-1].imshow(clean_np, cmap='bwr', vmin=vmin, vmax=vmax)
                axes[-1].set_title("Clean Ground Truth")
                axes[-1].axis('off')

                if mappable:
                    fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                
                # --- MODIFICATION: Save file inside the loss-specific sub-folder ---
                save_path = loss_output_dir / f"sample_{sample_id_val}_sim_{config_name_val}.png"
                plt.savefig(save_path)
                plt.close(fig)
                print(f"    - Saved comparison plot to {save_path}")
            
            processed_samples += 1

    print("\nInference and display script finished.")

if __name__ == "__main__":
    main()