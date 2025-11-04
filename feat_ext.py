"""
Feature Extraction from Trained SSL Model
Extract features and save as CSV file with subject IDs
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.data import DataLoader, Dataset
from architectures import sfcn_ssl2
import config as c
import json
from dataloaders import dataloader

def load_pretrained_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load pretrained SSL model.
    
    Args:
        model_path: Path to the pretrained model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model in evaluation mode
    """
    backbone = sfcn_ssl2.SFCN()
    checkpoint = torch.load(model_path, map_location=device)
    backbone.load_state_dict(checkpoint['state_dict'], strict=False)
    backbone = backbone.to(device)
    backbone.eval()
    print(f"✅ Loaded pretrained model from: {model_path}")
    return backbone


def extract_ssl_features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Extract features from SSL model backbone.
    
    Args:
        model: Pretrained SSL model
        images: Batch of images
        
    Returns:
        Flattened feature vectors
    """
    with torch.no_grad():
        features, _ = model(images, return_projection=True)
        return features.view(features.size(0), -1)


def extract_features_from_dataset(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device) -> tuple:
    """
    Extract features from entire dataset.
    
    Args:
        model: Pretrained SSL model
        data_loader: DataLoader for the dataset
        device: Device to run inference on
        
    Returns:
        Tuple of (features array, list of eids)
    """
    all_features = []
    all_eids = []
    
    with torch.no_grad():
        for eid, images in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)
            features = extract_ssl_features(model, images).cpu().numpy()
            all_features.append(features)
            all_eids.extend(eid)
    
    all_features = np.vstack(all_features)
    print(f"✅ Extracted backbone features: {all_features.shape}")
    
    return all_features, all_eids


def save_features_to_csv(features: np.ndarray, eids: list, save_path: str):
    """
    Save extracted features to CSV with eid as first column.
    
    Args:
        features: Feature array (n_samples, n_features)
        eids: List of subject IDs
        save_path: Path to save CSV file
    """
    # Create DataFrame with eid as first column
    df = pd.DataFrame(features)
    df.insert(0, 'eid', eids)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"💾 Saved features with eid to: {save_path}")


def main():
    """Main execution function."""
    # Configuration
    pretrained_model = os.path.join(c.MODEL_DIR, c.MODEL_NAME)
    tensor_dir = c.IMAGES_EXT_DIR
    feat_dir = c.FEATURES_EXT_DIR
    os.makedirs(feat_dir, exist_ok=True)
    
    # Load pretrained model
    model = load_pretrained_model(pretrained_model, c.DEVICE)
    
    # Create dataset and dataloader
    '''
    # Load data
    with open(c.JSON_PATH, "r") as json_f:
        json_data = json.load(json_f)
        data = json_data["training"]

    # DataLoaders
    dataset = Dataset(data=data)
    dataloader = DataLoader(
        dataset, batch_size=c.BATCH_SIZE, shuffle=True, 
        num_workers=c.NUM_WORKERS, drop_last=True
    )
    '''

    dataset = dataloader.BrainDataset(root_dir=tensor_dir)
    data_loader = DataLoader(
        dataset,
        batch_size=c.BATCH_SIZE,
        num_workers=8,
        drop_last=False
    )
   
    print(f"📊 Dataset size: {len(dataset)} samples")
    
    # Extract features
    features, eids = extract_features_from_dataset(model, data_loader, c.DEVICE)
    
    # Save to CSV
    save_path = os.path.join(feat_dir, '_features.csv')
    save_features_to_csv(features, eids, save_path)


if __name__ == "__main__":
    main()