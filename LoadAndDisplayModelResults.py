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

MODEL_PATHS = [
    r"training_run_results_top\best_model_LR_1e-05_WD_1e-07_ED_4_DC_256_128_64_32_LOSS_SmoothL1.pth",
    r"training_run_results_top\best_model_LR_1e-05_WD_1e-07_ED_4_DC_256_128_64_32_LOSS_SSIM.pth",
    r"training_run_results_top\best_model_LR_1e-05_WD_1e-07_ED_4_DC_256_128_64_32_LOSS_SSIM+SmoothL1.pth"
]
DATA_ROOT = r"datasets/TopographicData"
NUM_SAMPLES = 15
OUTPUT_DIR = "topo_results"

# ===================================================================


class PrecomputedNoise2NoiseDataset(Dataset):
    def __init__(self, manifest_file, root_dir, for_training=True):
        self.root_dir = Path(root_dir) # <-- MODIFICATION: Use pathlib for robustness
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
        
        # <-- MODIFICATION: Switched from os.path.join to pathlib for cross-platform compatibility
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
    filename = Path(model_path).name
    params = {}
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

    manifest_path = Path(DATA_ROOT) / 'manifest.csv'
    if not manifest_path.exists():
        print(f"Manifest file {manifest_path} not found. Cannot proceed.")
        return
    
    full_dataset = PrecomputedNoise2NoiseDataset(manifest_file=manifest_path, root_dir=DATA_ROOT, for_training=False)

    if len(full_dataset) > 0:
        total_samples = len(full_dataset)
        indices = list(range(total_samples))
        np.random.seed(42) 
        np.random.shuffle(indices)
        train_ratio, val_ratio = 0.7, 0.15
        train_split_idx = int(train_ratio * total_samples)
        val_split_idx = train_split_idx + int(val_ratio * total_samples)
        test_indices = indices[val_split_idx:] if total_samples >=3 else []
        
        if not test_indices:
            print("Warning: No test samples found. Cannot run inference.")
            return

        test_subset = Subset(full_dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=True)
        print(f"Loaded {len(test_subset)} test samples.")
    else:
        print("Dataset is empty. Cannot create test loader.")
        return

    models, model_infos = [], []
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
                encoder_weights=None, 
                encoder_depth=params['encoder_depth'], 
                decoder_channels=params['decoder_channels'], 
                in_channels=1, 
                classes=1
            )
            
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            models.append(model)
        except Exception as e:
            print(f"Error loading model {path}: {e}")
            
    if not models:
        print("No models were successfully loaded. Exiting.")
        return
        
    run_id_parts = []
    for info in model_infos:
        combo_id = info['combo_id']
        loss_match = re.search(r'LOSS_(\w+)', combo_id)
        loss_part = loss_match.group(1) if loss_match else "na"
        ed_match = re.search(r'ED_(\d+)', combo_id)
        ed_part = "ED" + ed_match.group(1) if ed_match else "na"
        run_id_parts.append(f"{loss_part}-{ed_part}")
    run_identifier = "_vs_".join(run_id_parts)
    print(f"Generated Run Identifier for filenames: {run_identifier}")

    print(f"\nStarting inference on {min(NUM_SAMPLES, len(test_loader))} samples...")
    processed_samples = 0
    with torch.no_grad():
        for noisy_tensor, clean_tensor, sample_id, time_step, config_name in test_loader:
            if processed_samples >= NUM_SAMPLES: break

            # <-- MODIFICATION: Gracefully handle samples that failed to load
            if isinstance(sample_id[0], str) and sample_id[0] == "error_id":
                print("Skipping sample due to a data loading error.")
                continue

            config_name_val, time_step_val, sample_id_val = config_name[0], time_step.item(), sample_id.item()
            print(f"Processing Sample {processed_samples + 1}/{NUM_SAMPLES} (ID: {sample_id_val}, Day: {time_step_val}, Sim: {config_name_val})")
            
            noisy_input = noisy_tensor.to(DEVICE)
            denoised_outputs_np = [model(noisy_input).cpu().squeeze().numpy() for model in models]
            noisy_np = noisy_tensor.cpu().squeeze().numpy()
            clean_np = clean_tensor.cpu().squeeze().numpy()
            
            all_images_for_sample = [noisy_np, clean_np] + denoised_outputs_np
            vmin = min(img.min() for img in all_images_for_sample)
            vmax = max(img.max() for img in all_images_for_sample)

            num_plots = 2 + len(models)
            fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6))
            
            fig.suptitle(f"Day: {time_step_val:.2f}  |  Simulation: {config_name_val}", fontsize=20)

            mappable = None
            axes[0].imshow(noisy_np, cmap='bwr', vmin=vmin, vmax=vmax)
            axes[0].set_title("Noisy Input")
            axes[0].axis('off')
            
            axes_denoised = axes[1:-1] if num_plots > 2 else []
            for j, denoised_img in enumerate(denoised_outputs_np):
                ax = axes_denoised[j]
                title_text = model_infos[j]['combo_id'].replace('_', ' ')
                ax.set_title(f"Denoised: {title_text}", fontsize=8)
                mappable = ax.imshow(denoised_img, cmap='bwr', vmin=vmin, vmax=vmax)
                ax.axis('off')
                
            axes[-1].imshow(clean_np, cmap='bwr', vmin=vmin, vmax=vmax)
            axes[-1].set_title("Clean Ground Truth")
            axes[-1].axis('off')

            if mappable:
                fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_path = Path(OUTPUT_DIR) / f"{run_identifier}_sample_{sample_id_val}.png"
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved comparison plot to {save_path}")
            processed_samples += 1

    print("\nInference and display script finished.")


if __name__ == '__main__':
    main()