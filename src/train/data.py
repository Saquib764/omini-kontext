from PIL import Image, ImageFilter, ImageDraw, ImageChops
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import os
from scipy.ndimage import gaussian_filter
from datasets import load_dataset, concatenate_datasets
from typing import List, Dict, Union, Optional, Any


def load_and_concatenate_datasets(
    dataset_names: List[str],
    source_field_values: List[str],
    source_field_name: str = "dataset_source",
    split: Optional[str] = "train",
    cache_dir: Optional[str] = None,
    **dataset_loading_kwargs: Any
) -> Dict[str, Any]:
    """
    Load multiple datasets from Hugging Face and concatenate them into a single dataset.
    Each dataset will have an additional field to identify its source.
    
    Args:
        dataset_names: List of dataset names/paths on Hugging Face to load
        source_field_values: List of values to add to the source field
        source_field_name: Name of the field to add for source identification
        split: Which split to load (e.g., 'train', 'validation', 'test'). If None, loads all splits.
        cache_dir: Directory to cache the downloaded datasets
        **dataset_loading_kwargs: Additional arguments to pass to load_dataset
        
    Returns:
        A dictionary mapping split names to concatenated datasets
    """
    if not dataset_names:
        raise ValueError("At least one dataset name must be provided")
    
    # Initialize dictionaries to store datasets by split
    datasets_by_split = {}
    
    # Load each dataset and add source field
    for i, dataset_name in enumerate(dataset_names):
        try:
            # Load the dataset
            dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir, **dataset_loading_kwargs)
            
            # If a specific split was requested, we have a single dataset object
            if split:
                dataset = dataset.add_column(source_field_name, [source_field_values[i]] * len(dataset))
                if split not in datasets_by_split:
                    datasets_by_split[split] = []
                datasets_by_split[split].append(dataset)
            # If no split was specified, we have a DatasetDict with multiple splits
            else:
                for split_name, split_dataset in dataset.items():
                    split_dataset = split_dataset.add_column(source_field_name, [source_field_values[i]] * len(split_dataset))
                    if split_name not in datasets_by_split:
                        datasets_by_split[split_name] = []
                    datasets_by_split[split_name].append(split_dataset)
                    
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
    
    # Concatenate datasets for each split
    concatenated_datasets = {}
    for split_name, split_datasets in datasets_by_split.items():
        if not split_datasets:
            continue
        
        try:
            # Make sure datasets have compatible features
            concatenated_datasets[split_name] = concatenate_datasets(split_datasets)
        except Exception as e:
            print(f"Error concatenating datasets for split {split_name}: {e}")
    
    return concatenated_datasets


def example_usage():
    """
    Example of how to use the load_and_concatenate_datasets function.
    """
    # Load and concatenate multiple datasets
    datasets = load_and_concatenate_datasets(
        dataset_names=["data/person-incontext/image1", "data/person-incontext/image2"],
        source_field_values=["image1", "image2"],
        split="train"
    )
    
    # Print information about the concatenated dataset
    print(f"Concatenated dataset splits: {list(datasets.keys())}")
    train_dataset = datasets["train"]
    print(f"Number of examples in concatenated train dataset: {len(train_dataset)}")
    print(f"First example: {train_dataset[0]}")
    
    # Count examples by source
    source_counts = {}
    for example in train_dataset:
        source = example["dataset_source"]
        if source not in source_counts:
            source_counts[source] = 0
        source_counts[source] += 1
    
    print(f"Source counts: {source_counts}")


if __name__ == "__main__":
    example_usage()

class FluxOminiKontextDataset(Dataset):
    """Example dataset for Flux Omini Kontext training"""
    
    def __init__(self):
        self.init_files = []
        self.reference_files = []
        self.target_files = []

        root = 'data'
        for f in os.listdir(f'{root}/start'):
            if not (os.path.isfile(os.path.join(root, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))):
                continue
            self.init_files.append(os.path.join(f"{root}/start", f))
            self.reference_files.append(os.path.join(f"{root}/reference", f))
            self.target_files.append(os.path.join(f"{root}/end", f))
        
        self.to_tensor = T.ToTensor()

    
    def __len__(self):
        return len(self.init_files)
    
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
            "input_image": self.to_tensor(input_image),
            "target_image": self.to_tensor(target_image),
            "reference_image": self.to_tensor(reference_image),
            "prompt": prompt,
            "reference_delta": reference_delta,
        }

