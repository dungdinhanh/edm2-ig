"""Inception-V3 and DINOv2 feature networks."""

import pickle
import PIL.Image
import torch
import torchvision
from typing import Any
import os
import dnnlib
import urllib.request

class InceptionV3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        with dnnlib.util.open_url(url, verbose=False) as f:
            self.model = pickle.load(f)
        self.feature_dim = 2048

    def forward(self,
                x: torch.Tensor,
                **kwargs) -> torch.Tensor:
        return self.model(x, return_features=True)
    
class InceptionV3cache(torch.nn.Module):
    def __init__(self, cache_folder: str = './inception_cache'):
        super().__init__()
        self.cache_folder = cache_folder
        
        # Ensure cache folder exists
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        # URL of the InceptionV3 model
        url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        model_path = os.path.join(self.cache_folder, 'inception-2015-12-05.pkl')

        # If the model is not found in the cache, download it
        if not os.path.exists(model_path):
            print(f"Model not found in cache, downloading to {model_path}...")
            # Download the model
            urllib.request.urlretrieve(url, model_path)

        # Load the model from the cached file
        print("Loading the InceptionV3 model...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.feature_dim = 2048

    def forward(self,
                x: torch.Tensor,
                **kwargs) -> torch.Tensor:
        return self.model(x, return_features=True)


class DINOv2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda')
        self.model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitl14')
        self.resize = torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        self.to_tensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.feature_dim = 1024

    def forward(self,
                x: torch.Tensor,
                **kwargs) -> torch.Tensor:
        # Pre-process images.
        # NOTE: Slow since tensors are first converted back to PIL.Images.
        # This is done because resizing PIL.Images and torch.tensors with BICUBIC
        # leads to slightly different results.
        x_np = [x_i.detach().cpu().numpy() for x_i in x]
        x_pils = [PIL.Image.fromarray(img.transpose(1, 2, 0)) for img in x_np]
        x_proc = [self.normalize(self.to_tensor(self.resize(x_pil))).unsqueeze(0) for x_pil in x_pils]
        x_proc = torch.cat(x_proc, dim=0).to(self.device)

        return self.model(x_proc)
    
class DINOv2cache(torch.nn.Module):
    def __init__(self, cache_folder: str = './dino_cache'):
        super().__init__()
        self.device = torch.device('cuda')

        # Ensure the cache folder exists, if not create it
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)

        # Define the model URL and path
        model_url = 'https://dl.fbaipublicfiles.com/dino/dino_vitl14/dino_vitl14.pth'
        model_path = os.path.join(cache_folder, 'dino_vitl14.pth')

        # Check if the model is already cached
        if not os.path.exists(model_path):
            print(f"Model not found in cache, downloading to {model_path}...")
            # Download the model to the cache folder
            torch.hub.download_url_to_file(model_url, model_path)

        # Load the model using the cached file
        print("Loading the DINOv2 model...")
        self.model = torch.hub.load_state_dict_from_url(model_url, model_dir=cache_folder, map_location=self.device)
        
        # Define image pre-processing steps
        self.resize = torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        self.to_tensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.feature_dim = 1024

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Pre-process images
        x_np = [x_i.detach().cpu().numpy() for x_i in x]
        x_pils = [PIL.Image.fromarray(img.transpose(1, 2, 0)) for img in x_np]
        x_proc = [self.normalize(self.to_tensor(self.resize(x_pil))).unsqueeze(0) for x_pil in x_pils]
        x_proc = torch.cat(x_proc, dim=0).to(self.device)

        # Forward pass through the model
        return self.model(x_proc)


def load_feature_network(name: str) -> Any:
    """Loads a feature network."""
    assert name in ['inception_v3', 'dinov2'], f'Invalid feature network name: {name}.'
    if name == 'inception_v3':
        return InceptionV3()
    else:
        return DINOv2()
    
def load_feature_network_cache(name: str, hub_folder:str) -> Any:
    """Loads a feature network."""
    assert name in ['inception_v3', 'dinov2'], f'Invalid feature network name: {name}.'
    if name == 'inception_v3':
        return InceptionV3cache(hub_folder)
    else:
        return DINOv2cache(hub_folder)
