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
import functools

# Make all print statements flush by default
print = functools.partial(print, flush=True)

print("Imports successful")

# --- EarlyStopping Class Definition ---
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement. 
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

PRECOMPUTED_DATA_ROOT = r"DataTop"

class PrecomputedNoise2NoiseDataset(Dataset):
    def __init__(self, manifest_file, root_dir):
        self.root_dir = root_dir
        try:
            self.manifest = pd.read_csv(manifest_file)
        except FileNotFoundError:
            print(f"Error: Manifest file not found at {manifest_file}")
            print("Please ensure you have run the data generation phase first.")
            self.manifest = pd.DataFrame()

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
            dummy_tensor = torch.zeros((1, 1500, 1500), dtype=torch.float)
            return dummy_tensor, dummy_tensor, dummy_tensor, 0.0, "error_config" 
        except Exception as e:
            print(f"Generic error loading file for sample id {record['id']}: {e}")
            dummy_tensor = torch.zeros((1, 1500, 1500), dtype=torch.float)
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
ENCODER_WEIGHTS = None
IN_CHANNELS = 1
OUT_CHANNELS = 1
NUM_EPOCHS = 15 # Increased epochs to give EarlyStopping a chance to work

architectures = [
    {"depth": 4, "channels": (256, 128, 64, 32)},
]

hyperparameter_settings = {
    'learning_rates': [1e-2],
    'weight_decays': [1e-07],
}

loss_functions_config = {
    "SmoothL1": nn.SmoothL1Loss
}

output_dir = "training_run_results_top"
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
        if batch_idx % 10 == 0:
            print(f"Tr B {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
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
        hyperparameter_settings['weight_decays']
    ]

    patience_for_early_stopping = 5

    for arch in architectures:
        current_ed = arch['depth']
        current_dc = arch['channels']
        
        for loss_name, LossClass in loss_functions_config.items():
            for combo_values in product(*param_value_lists):
                current_lr, current_wd = combo_values
                
                combo_id_parts = [
                    f"LR_{current_lr}",
                    f"WD_{current_wd}",
                    f"ED_{current_ed}",
                    f"DC_{'_'.join(map(str, current_dc))}",
                    f"LOSS_{loss_name}"
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

                criterion = LossClass().to(DEVICE)
                optimizer = optim.Adam(model.parameters(), lr=current_lr, weight_decay=current_wd)
                scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

                model_save_path = os.path.join(output_dir, f"best_model_{combo_id}.pth")
                early_stopper = EarlyStopping(patience=patience_for_early_stopping, verbose=True, path=model_save_path)

                train_losses_for_combo, val_losses_for_combo = [], []

                print(f"\nStarting training for {combo_id}...")
                for epoch in range(NUM_EPOCHS):
                    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} for {combo_id} ---")
                    train_loss = train_one_epoch(train_loader, model, optimizer, criterion, DEVICE, scaler)
                    train_losses_for_combo.append(train_loss)

                    current_epoch_val_loss = float('inf')
                    if 'val_loader' in globals() and val_loader is not None:
                        current_epoch_val_loss = validate_one_epoch(val_loader, model, criterion, DEVICE)
                        val_losses_for_combo.append(current_epoch_val_loss)

                        early_stopper(current_epoch_val_loss, model)

                        if early_stopper.early_stop:
                            print("Early stopping triggered")
                            break 
                
                print(f"\nTraining finished for {combo_id}!")

                # Old model saving logic is removed. Best model is already saved by EarlyStopping.
                
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