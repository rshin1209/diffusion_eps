import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class ReactionDynamicsDataset(Dataset):
    """Dataset for loading reaction dynamics data for training."""

    def __init__(self, filename):
        """
        Initialize the dataset by loading data from a numpy file.

        Args:
            filename (str): The name of the numpy file to load data from.
        """
        self.data = self._load_from_numpy(filename)  # Load the data using a static method

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            np.ndarray: The sample at the given index.
        """
        return self.data[idx]

    @staticmethod
    def _load_from_numpy(filename):
        """
        Static method to load data from a numpy file.

        Args:
            filename (str): The name of the numpy file to load data from.

        Returns:
            np.ndarray: Loaded data as a numpy array.
        """
        return np.load(f"./dataset/{filename}/data.npy").astype(np.float32)
