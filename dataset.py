# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
from typing import List, \
                   Optional
import dnnlib

try:
    import pyspng
except ImportError:
    pyspng = None


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels
    
class NPZImageDataset(torch.utils.data.Dataset):
    def __init__(self, npz_file_path, transform=None, xflip=False):
        """
        Args:
            npz_file_path (str): Path to the .npz file containing the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            xflip (bool, optional): If True, doubles the dataset size with horizontally flipped images.
        """
        super().__init__()
        self.npz_file_path = npz_file_path
        self.transform = transform
        self.xflip = xflip

        # Load the entire dataset into memory
        with np.load(self.npz_file_path) as npz_file:
            keys = list(npz_file.keys())
            self.images = npz_file[keys[0]].transpose(0, 3, 1, 2)  # Convert HWC to CHW

            if len(keys) > 1:  # If there's a second matrix, it's assumed to be the labels
                self.labels = npz_file[keys[1]]
            else:
                self.labels = np.zeros((self.images.shape[0]))
        if self.labels is not None:
            assert len(self.images) == len(self.labels) , \
                "Mismatch between image and label counts."

        self.original_length = len(self.images)
        self.length = self.original_length * (2 if self.xflip else 1)  # Double size if xflip=True

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Check if this is a flipped image index
        is_flipped = self.xflip and idx >= self.original_length
        actual_idx = idx % self.original_length  # Get the actual index in the original dataset

        # Retrieve the image
        image = self.images[actual_idx]

        # Retrieve the label if available
        label = self.labels[actual_idx] if self.labels is not None else None

        # Apply horizontal flip if needed
        if is_flipped:
            image = np.flip(image, axis=2)  # Flip along the width axis (CHW)

        # Apply any additional transformations
        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloader(zip_path: str,
                   resolution: int,
                   batch_size: int,
                   num_images: int,
                   use_labels: bool = False,
                   xflip: bool = False,
                   max_size: Optional[int] = None,
                   pin_memory: bool = True,
                   num_workers: int = 1,
                   prefetch_factor: int = 2) -> torch.utils.data.DataLoader:
    """Initializes a dataloader for .zip file."""
    # Create a zip dataset object.
    dataset = ImageFolderDataset(path=zip_path,
                                 resolution=resolution,
                                 use_labels=use_labels,
                                 xflip=xflip,
                                 max_size=max_size)
    item_subset = [i % num_images for i in range(num_images)]

    return torch.utils.data.DataLoader(dataset=dataset,
                                       sampler=item_subset,
                                       batch_size=batch_size,
                                       pin_memory=pin_memory,
                                       num_workers=num_workers,
                                       prefetch_factor=prefetch_factor)

def get_dataloader_npz(npz_path: str,
                   resolution: int,
                   batch_size: int,
                   num_images: int,
                   xflip: bool = False,
                   max_size: Optional[int] = None,
                   pin_memory: bool = True,
                   num_workers: int = 1,
                   prefetch_factor: int = 2) -> torch.utils.data.DataLoader:
    """Initializes a dataloader for .npz file."""
    # Create an NPZ dataset object.
    dataset = NPZImageDataset(npz_file_path=npz_path,
                              transform=None,  # Transform can be added here if needed
                              xflip=xflip)

    # Apply max_size if provided
    # if max_size is not None and max_size < len(dataset):
    #     np.random.seed(0)  # Set random seed for reproducibility
    #     indices = np.random.choice(len(dataset), max_size, replace=False)
    #     dataset._raw_idx = indices  # Modify the indices to respect max_size

    item_subset = [i % len(dataset) for i in range(num_images)]  # Ensure indices are within the dataset length

    # Return a DataLoader for the NPZ dataset
    return torch.utils.data.DataLoader(dataset=dataset,
                      sampler=torch.utils.data.SubsetRandomSampler(item_subset),  # Use sampler for better shuffling
                      batch_size=batch_size,
                      pin_memory=pin_memory,
                      num_workers=num_workers,
                      prefetch_factor=prefetch_factor)
