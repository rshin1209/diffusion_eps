import os
import numpy as np
from torch.utils.data import Dataset
from typing import Any


class ReactionDynamicsDataset(Dataset):
    """Dataset for loading reaction dynamics data for training."""

    def __init__(self, reaction_name: str):
        """
        Initialize the dataset by loading data from a numpy file.

        Args:
            reaction_name (str): The name of the reaction directory within ./dataset/.
        """
        data_path = os.path.join("./dataset", reaction_name, "data.npy")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")

        self.data = self._load_from_numpy(data_path)

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        """
        Retrieve a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            np.ndarray: The sample at the given index.
        """
        return self.data[idx]

    @staticmethod
    def _load_from_numpy(filename: str) -> np.ndarray:
        """
        Static method to load data from a numpy file.

        Args:
            filename (str): The name of the numpy file to load data from.

        Returns:
            np.ndarray: Loaded data as a numpy array.
        """
        data = np.load(filename).astype(np.float32)
        return data
