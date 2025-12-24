import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.utils import time_warp

class RML2016Dataset(Dataset):
    def __init__(self, data_path, train=True, test_split=0.2, seed=42, transform=None):
        """
        Args:
            data_path (str): Path to the .pkl file.
            train (bool): If True, returns training set, else test set.
            test_split (float): Fraction of data to use for testing.
            seed (int): Random seed for splitting.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        
        # Load Data
        with open(data_path, 'rb') as f:
            # Python 2/3 compatibility
            data = pickle.load(f, encoding='latin1')
            
        self.samples = []
        self.labels = []
        self.snrs = []
        
        # Get all modulation classes
        self.classes = sorted(list(set([k[0] for k in data.keys()])))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Flatten data
        for key in data.keys():
            mod_name, snr = key
            samples = data[key] # Shape (N, 2, 128)
            
            for i in range(samples.shape[0]):
                self.samples.append(samples[i])
                self.labels.append(self.class_to_idx[mod_name])
                self.snrs.append(snr)
                
        self.samples = np.array(self.samples) # (Total_N, 2, 128)
        self.labels = np.array(self.labels)
        self.snrs = np.array(self.snrs)
        
        # Split Train/Test
        np.random.seed(seed)
        n_samples = len(self.samples)
        indices = np.random.permutation(n_samples)
        split_idx = int(n_samples * (1 - test_split))
        
        if train:
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
            
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # Original Sample
        x_orig = self.samples[real_idx] # (2, 128)
        label = self.labels[real_idx]
        snr = self.snrs[real_idx]
        
        # Augmentation (Time Warp)
        # CIRL requires (x, x_aug) pairs
        # We apply time_warp to create x_aug
        x_aug = time_warp(x_orig, sigma=0.2, num_knots=4)
        
        # Convert to Tensor
        x_orig = torch.from_numpy(x_orig).float()
        x_aug = torch.from_numpy(x_aug).float()
        label = torch.tensor(label).long()
        
        return x_orig, x_aug, label, snr
