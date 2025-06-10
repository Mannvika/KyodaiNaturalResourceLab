#!/usr/bin/env python
# coding: utf-8

print("Script starting now.", flush=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import product
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import os
import pandas as pd
import segmentation_models_pytorch as smp
import re
import torchvision.models as models

print("Imports successful", flush=True)

PRECOMPUTED_DATA_ROOT = r"Data"

class PrecomputedNoise2NoiseDataset(Dataset):
    def __init__(self, manifest_file, root_dir):
        self.root_dir = root_dir
        try:
            self.manifest = pd.read_csv(manifest_file)
        except FileNotFoundError:
            print(f"Error: Manifest file not found at {manifest_file}")
            print("Please ensure you have run the data generation phase first.")
            self.manifest = pd.DataFrame() # Empty dataframe

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        if idx >= len(self.manifest):
            raise IndexError("Index out of bounds")
            
        record = self.manifest.iloc[idx]
        
        noisy1_path = os.path.join(self.root_dir, record['noisy1_path'])
        noisy2_path = os.path.join(self.root_dir, record['noisy2_path'])
        clean_path = os.path.join(self.root_dir, record['clean_path'])

        time_step = record['time_step']
        config_name = record['config_name']

        try:
            noisy1_tensor = torch.load(noisy1_path)
            noisy2_tensor = torch.load(noisy2_path)
            clean_tensor = torch.load(clean_path)
        except FileNotFoundError as e:
            print(f"Error loading file for sample id {record['id']}: {e}")
            dummy_tensor = torch.zeros((1, 1500, 1500), dtype=torch.float) # Assuming IMG_SIZE is 1500x1500
            return dummy_tensor, dummy_tensor, dummy_tensor, 0.0, "error_config" 
        except Exception as e:
            print(f"Generic error loading file for sample id {record['id']}: {e}")
            dummy_tensor = torch.zeros((1, 1500, 1500), dtype=torch.float) # Assuming IMG_SIZE is 1500x1500
            return dummy_tensor, dummy_tensor, dummy_tensor, 0.0, "error_config"

        return noisy1_tensor, noisy2_tensor, clean_tensor, time_step, config_name

manifest_path = os.path.join(PRECOMPUTED_DATA_ROOT, 'manifest.csv')

if not os.path.exists(manifest_path):
    print(f"Manifest file {manifest_path} not found. Please run the data generation phase first.")
    
full_dataset = PrecomputedNoise2NoiseDataset(manifest_file=manifest_path, root_dir=PRECOMPUTED_DATA_ROOT)

train_loader, val_loader, test_loader = None, None, None

if len(full_dataset) > 0:
    total_samples = len(full_dataset)
    indices = list(range(total_samples))
    np.random.shuffle(indices)

    train_ratio = 0.7
    val_ratio = 0.15

    if total_samples < 3:
        train_indices = indices
        val_indices, test_indices = [],[]
    else:
        train_split_idx = int(train_ratio * total_samples)
        val_split_idx = train_split_idx + int(val_ratio * total_samples)
        
        train_indices = indices[:train_split_idx]
        val_indices = indices[train_split_idx:val_split_idx]
        test_indices = indices[val_split_idx:]

        if not test_indices and val_indices: test_indices = val_indices[-1:]; val_indices = val_indices[:-1]
        if not val_indices and train_indices: val_indices = train_indices[-1:]; train_indices = train_indices[:-1]

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)

    print(f"Total precomputed samples: {total_samples}")
    print(f"Train samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test samples: {len(test_subset)}")
    
    BATCH_SIZE = 1 
    if len(train_subset) > 0:
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    if len(val_subset) > 0:
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    if len(test_subset) > 0:
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

else:
    print("Full dataset is empty. Cannot create data loaders.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
IN_CHANNELS = 1
OUT_CHANNELS = 1
NUM_EPOCHS = 5

class TVLoss(nn.Module):
    """Total Variation Loss to encourage smoothness in the output."""
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size, c, h, w = x.shape
        tv_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return (tv_h + tv_w) / (batch_size * c * h * w)

class PerceptualLoss(nn.Module):
    """Perceptual Loss using a pre-trained VGG19 network."""
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.features = nn.Sequential(*list(vgg.children())[:36]) 
        self.loss_fn = nn.L1Loss()

    def forward(self, output, target):
        if output.shape[1] == 1:
            output = output.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
        
        output_features = self.features(output)
        target_features = self.features(target)
        
        return self.loss_fn(output_features, target_features)

class CombinedLoss(nn.Module):
    def __init__(self, device, w_pixel=1.0, w_perc=0.1, w_tv=1e-4):
        super(CombinedLoss, self).__init__()
        self.w_pixel = w_pixel
        self.w_perc = w_perc
        self.w_tv = w_tv
        
        self.pixel_loss = nn.SmoothL1Loss()
        self.perc_loss = PerceptualLoss(device)
        self.tv_loss = TVLoss()
        print(f"CombinedLoss initialized with weights: Pixel={w_pixel}, Perceptual={w_perc}, TV={w_tv}")

    def forward(self, output, target):
        # Calculate each loss component
        loss_p = self.pixel_loss(output, target)
        loss_g = self.perc_loss(output, target)
        loss_t = self.tv_loss(output)
        
        # Combine the losses using the weights
        total_loss = (self.w_pixel * loss_p) + (self.w_perc * loss_g) + (self.w_tv * loss_t)
        return total_loss


# --- FIX: Define valid architectures to prevent ValueError ---
architectures = [
    {"depth": 3, "channels": (256, 128, 64)},
    {"depth": 3, "channels": (128, 64, 32)},
    {"depth": 3, "channels": (192, 96, 48)},
    {"depth": 4, "channels": (128, 64, 32, 16)},
    {"depth": 4, "channels": (192, 96, 48, 24)},
    {"depth": 4, "channels": (256, 128, 64, 32)},
]

hyperparameter_settings = {
    'learning_rates': [8e-6, 1e-4, 8e-5, 1e-5],
    'weight_decays': [0, 1e-07, 1e-06, 5e-6],
}

''' loss_functions_config = {
    "MSE": nn.MSELoss,
    "L1": nn.L1Loss,
    "SmoothL1": nn.SmoothL1Loss
} '''

loss_weight_settings = {
    'w_pixel': [1.0],
    'w_perc': [0.1, 0.01],
    'w_tv': [1e-4, 1e-5]
}

output_dir = "training_run_results_combined_loss"
os.makedirs(output_dir, exist_ok=True)

def train_one_epoch(loader, model, optimizer, loss_fn, device, scaler):
    model.train()
    epoch_loss = 0
    if loader is None: return 0.0

    for batch_idx, (noisy1, noisy2, _c, _ts, _cn) in enumerate(loader):
        noisy1, noisy2 = noisy1.to(device), noisy2.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            denoised_output = model(noisy1)
            loss = loss_fn(denoised_output, noisy2)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        if batch_idx % 10 == 0: print(f"Tr B {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")

    avg_epoch_loss = epoch_loss / len(loader) if len(loader) > 0 else 0.0
    print(f"End Epoch Train Avg Loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss

def validate_one_epoch(loader, model, loss_fn, device):
    model.eval()
    epoch_loss = 0
    if loader is None: return float('inf')

    with torch.no_grad():
        for noisy1, noisy2, _c, _ts, _cn in loader:
            noisy1, noisy2 = noisy1.to(device), noisy2.to(device)
            # FIX: Updated autocast call to resolve FutureWarning
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                denoised_output = model(noisy1)
                loss = loss_fn(denoised_output, noisy2)
            epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(loader) if len(loader) > 0 else float('inf')
    print(f"Val Avg Loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss

print("\nStarting hyperparameter tuning...")

if 'train_loader' not in globals() or train_loader is None:
    print("Training cannot proceed: train_loader is None or not defined in global scope.")
else:
    param_value_lists = [
        hyperparameter_settings['learning_rates'],
        hyperparameter_settings['weight_decays'],
        loss_weight_settings['w_pixel'],
        loss_weight_settings['w_perc'],
        loss_weight_settings['w_tv']
    ]

    for arch in architectures:
        current_ed = arch['depth']
        current_dc = arch['channels']
        
        for combo_values in product(*param_value_lists):
            current_lr, current_wd, w_p, w_g, w_t = combo_values
            combo_id_parts = [
                f"LR_{current_lr}",
                f"WD_{current_wd}",
                f"ED_{current_ed}",
                f"DC_{'_'.join(map(str, current_dc))}",
                f"W_P_{w_p}_W_G_{w_g}_W_T_{w_t}" # New ID for weights
            ]
            combo_id = "_".join(combo_id_parts)
            print(f"\n\n{'='*10} Starting Run for Combination: {combo_id} {'='*10}")

            model = smp.Unet(
                encoder_name=ENCODER,
                encoder_weights=ENCODER_WEIGHTS,
                encoder_depth=current_ed,
                decoder_channels=current_dc,
                in_channels=IN_CHANNELS,
                classes=OUT_CHANNELS,
            ).to(DEVICE)

            criterion = CombinedLoss(DEVICE, w_pixel=w_p, w_perc=w_g, w_tv=w_t).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=current_lr, weight_decay=current_wd)
            # FIX: Updated GradScaler call to resolve FutureWarning
            scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

            train_losses_for_combo, val_losses_for_combo = [], []
            best_val_loss_for_combo = float('inf')
            best_model_state_for_combo = None

            print(f"\nStarting training for {combo_id}...")
            for epoch in range(NUM_EPOCHS):
                print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} for {combo_id} ---")
                train_loss = train_one_epoch(train_loader, model, optimizer, criterion, DEVICE, scaler)
                train_losses_for_combo.append(train_loss)

                current_epoch_val_loss = float('inf')
                if 'val_loader' in globals() and val_loader is not None:
                    current_epoch_val_loss = validate_one_epoch(val_loader, model, criterion, DEVICE)
                    val_losses_for_combo.append(current_epoch_val_loss)

                    if current_epoch_val_loss < best_val_loss_for_combo:
                        best_val_loss_for_combo = current_epoch_val_loss
                        best_model_state_for_combo = model.state_dict()
                        print(f"New best validation loss for {combo_id}: {best_val_loss_for_combo:.4f} at epoch {epoch+1}")
                else:
                    pass
            
            print(f"\nTraining finished for {combo_id}!")

            if best_model_state_for_combo is not None and 'val_loader' in globals() and val_loader is not None:
                model_save_path = os.path.join(output_dir, f"best_model_{combo_id}.pth")
                torch.save(best_model_state_for_combo, model_save_path)
                print(f"Saved best model for {combo_id} to {model_save_path} (Val Loss: {best_val_loss_for_combo:.4f})")
            elif 'val_loader' in globals() and val_loader is not None:
                print(f"No improvement in validation loss for {combo_id}. Best model not saved.")
            else:
                print(f"No validation loader provided for {combo_id}. Best model not saved.")

            if train_losses_for_combo or val_losses_for_combo:
                plt.figure(figsize=(6,3))
                if train_losses_for_combo:
                    plt.plot(train_losses_for_combo, label=f"Training Loss ({loss_name})")
                if val_losses_for_combo:
                    plt.plot(val_losses_for_combo, label=f"Validation Loss ({loss_name})")
                
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                title_str = f"Loss Over Epochs - {combo_id}"
                plt.title(title_str.replace('_', ' '))
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plot_save_path = os.path.join(output_dir, f"loss_plot_{combo_id}.png")
                plt.savefig(plot_save_path)
                plt.close()
                print(f"Saved loss plot for {combo_id} to {plot_save_path}")

print("\nAll hyperparameter tuning runs finished!")
