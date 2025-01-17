# utils/shared_memory.py

import numpy as np
from multiprocessing import shared_memory
from typing import Tuple

def create_shared_memory(array: np.ndarray) -> Tuple[str, Tuple[int, ...], np.dtype]:
    """
    Create a shared memory block for a given NumPy array.

    Args:
        array (np.ndarray): The NumPy array to share.

    Returns:
        Tuple[str, Tuple[int, ...], np.dtype]: Shared memory name, shape, and data type.
    """
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
    shm_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    shm_array[:] = array[:]
    return shm.name, array.shape, array.dtype

def access_shared_memory(name: str, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """
    Access a shared memory block and return it as a NumPy array.

    Args:
        name (str): Name of the shared memory block.
        shape (Tuple[int, ...]): Shape of the array.
        dtype (np.dtype): Data type of the array.

    Returns:
        np.ndarray: The NumPy array backed by shared memory.
    """
    shm = shared_memory.SharedMemory(name=name)
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return array

