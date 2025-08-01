import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
from pathlib import Path


class FluxOminiKontextDataset(Dataset):
    """Example dataset for Flux Omini Kontext training"""
    
    def __init__(self):
        self.init_files = []
        self.reference_files = []
        self.target_files = []
        

        root = '../data'
        for f in os.listdir('../data/start'):
            if not (os.path.isfile(os.path.join(root, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))):
                continue
            self.init_files.append(os.path.join(f"{root}/start", f))
            self.reference_files.append(os.path.join(f"{root}/reference", f))
            self.target_files.append(os.path.join(f"{root}/end", f))

    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_image_path = self.init_files[idx]
        target_image_path = self.target_files[idx]
        reference_image_path = self.reference_files[idx]

        input_image = Image.open(input_image_path)
        target_image = Image.open(target_image_path)
        reference_image = Image.open(reference_image_path)

        prompt = "add the character to the image"
        reference_delta = np.array([0, 512//16])
        return {
            "input_image": input_image,
            "target_image": target_image,
            "reference_image": reference_image,
            "prompt": prompt,
            "reference_delta": reference_delta,
        }


def create_dataloaders(
    train_data_path: str = "./data/train",
    val_data_path: str = "./data/val",
    batch_size: int = 2,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    """
    Create train and validation dataloaders
    
    Args:
        train_data_path: Path to training data
        val_data_path: Path to validation data
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    
    # Create datasets
    train_dataset = FluxOminiKontextDataset(train_data_path)
    val_dataset = FluxOminiKontextDataset(val_data_path)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_dataloader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_dataloader)} batches")
    
    return train_dataloader, val_dataloader
