from PIL import Image
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
from ..pipeline_tools import optimise_image_condition


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

def select_and_load_dataset(dataset_name: str, delta: List[int] = [0, 0, 0], drop_text_prob: float = 0.1, spatial: bool = False, split: str = "train", pil=False):
    # if dataset_name folder exists, load from there
    if os.path.exists(dataset_name):
        return FluxOminiKontextDataset(dataset_name, delta=delta, drop_text_prob=drop_text_prob, spatial=spatial, pil=pil)
    # otherwise, load from huggingface
    return FluxOminiKontextDatasetHF(dataset_name, delta=delta, drop_text_prob=drop_text_prob, spatial=spatial, split=split)

class FluxOminiKontextDatasetHF(Dataset):
    def __init__(
        self,
        dataset_name: str,
        delta: List[int] = [0, 0, 0],
        drop_text_prob: float = 0.1,
        spatial: bool = False,
        split: str = "train",
    ):
        self.base_dataset = load_dataset(dataset_name, split=split)
        self.delta = delta
        self.spatial = spatial
        self.to_tensor = T.ToTensor()
        self.drop_text_prob = drop_text_prob


    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        input_image = data['input_image']
        target_image = data['target_image']
        reference_image = data['reference_image']

        # Make sure the input image is smaller than 768
        if input_image.width > 768 or input_image.height > 768:
            scale = 768 / max(input_image.width, input_image.height)
            input_image = input_image.resize((int(input_image.width*scale//16)*16, int(input_image.height*scale//16)*16))
            target_image = target_image.resize((int(target_image.width*scale//16)*16, int(target_image.height*scale//16)*16))

        # Randomly resize the reference image
        reference_image = reference_image.resize((512, 512))


        prompt = "add the subject to the image"
        if random.random() < self.drop_text_prob:
            prompt = ""
        reference_delta = np.array(self.delta)
        if self.spatial:
            reference_image, reference_delta = optimise_image_condition(reference_image, reference_delta)
        return {
            "input_image": self.to_tensor(input_image),
            "target_image": self.to_tensor(target_image),
            "reference_image": self.to_tensor(reference_image),
            "prompt": prompt,
            "reference_delta": reference_delta,
        }


class FluxOminiKontextDataset(Dataset):
    """Example dataset for Flux Omini Kontext training"""
    
    def __init__(
        self, 
        src: str = 'data/character',
        delta: List[int] = [0, 0, 0],
        drop_text_prob: float = 0.1,
        spatial: bool = False,
        pil=False
    ):
        self.init_files = []
        self.reference_files = []
        self.target_files = []
        self.delta = delta
        self.drop_text_prob = drop_text_prob
        self.pil = pil
        self.spatial = spatial
        print(f"Loading dataset from {src} with spatial={spatial}")
        root = src
        for f in os.listdir(f'{root}/start'):
            if not (os.path.isfile(os.path.join(f'{root}/start', f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))):
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

        input_image = Image.open(input_image_path).convert("RGB")
        target_image = Image.open(target_image_path).convert("RGB")
        reference_image = Image.open(reference_image_path).convert("RGB")

        # make sure the input image is smaller than 1024
        if input_image.width > 1152 or input_image.height > 1152:
            scale = 1024 / max(input_image.width, input_image.height)
            input_image = input_image.resize((int(input_image.width*scale//16)*16, int(input_image.height*scale//16)*16))
            target_image = target_image.resize((int(target_image.width*scale//16)*16, int(target_image.height*scale//16)*16))

            if self.spatial:
                reference_image = reference_image.resize((int(reference_image.width*scale//16)*16, int(reference_image.height*scale//16)*16))

        # make sure the reference image is smaller than 1024
        # if reference_image.width > 1024 or reference_image.height > 1024:
        #     scale = 1024 / max(reference_image.width, reference_image.height)
        #     reference_image = reference_image.resize((int(reference_image.width*scale//16)*16, int(reference_image.height*scale//16)*16))
        
        # Paste the reference image on white background, of same size as the reference image
        reference_image = Image.new("RGB", (reference_image.width, reference_image.height), (255, 255, 255))
        # random resize and random paste the reference image on the white background
        scale = random.uniform(0.9, 1.1)
        reference_image = reference_image.resize((int(reference_image.width*scale), int(reference_image.height*scale)))
        x = random.randint(0, 50)
        y = random.randint(0, 50)
        reference_image.paste(reference_image, (x, y))

        prompt = "add the character to the image"
        reference_delta = np.array(self.delta)
        if self.spatial:
            reference_image, reference_delta = optimise_image_condition(reference_image, reference_delta)
            # print(f"Optimised reference image with delta={reference_delta}, size={reference_image.size}")
        if self.pil:
            return {
                "input_image": input_image,
                "target_image": target_image,
                "reference_image": reference_image,
                "prompt": prompt,
                "reference_delta": reference_delta,
            }
        return {
            "input_image": self.to_tensor(input_image),
            "target_image": self.to_tensor(target_image),
            "reference_image": self.to_tensor(reference_image),
            "prompt": prompt,
            "reference_delta": reference_delta,
        }

