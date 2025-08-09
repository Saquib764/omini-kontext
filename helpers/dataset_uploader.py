# This script uploads a dataset to the Hugging Face Hub
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of current working directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.data import (
    FluxOminiKontextDataset
)
from datasets import Dataset, DatasetInfo, Features, Image

import pandas as pd


from huggingface_hub import login

# Login to Hugging Face, enter token if required
login()


dataset = FluxOminiKontextDataset("custom_data/mixed_product", pil=True)

print("Dataset length: ", len(dataset))

# Prepare data for Hugging Face Dataset using file paths (Image() feature will load them)
data = []
for i in range(len(dataset)):
    item = dataset[i]
    # print(i)
    data.append({
        "input_image": item["input_image"],  # Convert image tensors to lists
        "target_image": item["target_image"],
        "reference_image": item["reference_image"],
    })


features = Features({
    "input_image": Image(),
    "target_image": Image(),
    "reference_image": Image(),
})

# Keep df in case of future metadata usage, though we build HF dataset from dict for clarity
_df = pd.DataFrame(data)

dataset_info = DatasetInfo(
    description="Cartoon dataset for condition and target mapping.",
    homepage="https://thefluxtrain.com",
    license="CC BY 4.0",
    features=features,
)

hf_dataset = Dataset.from_dict(_df.to_dict(orient="list"), features=features)


hf_dataset.push_to_hub("saquiboye/product-triplet")

