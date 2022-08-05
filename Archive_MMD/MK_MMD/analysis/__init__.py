import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm

def collect_feature(data_loader: DataLoader, feature_extractor: nn.Module,
                    device: torch.device) -> torch.Tensor:
    """
    """
    feature_extractor.eval()

    all_features = []
    with torch.no_grad():
        for _, (images, target) in enumerate(tqdm.tqdm(data_loader)):
            images = images.to(device)
            feature = feature_extractor(images)
            all_features.append(feature)
    
    return torch.cat(all_features, dim=0)

