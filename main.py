# main.py

import os
import torch
import joblib
import math
import argparse
import numpy as np
import logging
from typing import List, Tuple, Any
from pathlib import Path
from torch.utils.data import DataLoader
from data import ReactionDynamicsDataset
from diffusion import Diffusion
from logger import setup_logging

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def load_scaler_and_atoms(reaction_name: str) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Load the scaler, atomic data, and run point information for a given reaction.

    Args:
        reaction_name (str): The name of the reaction directory.

    Returns:
        Tuple[Any, np.ndarray, np.ndarray]: Scaler, atoms, and runpoint arrays.
    """
    scaler_path = Path(f"./dataset/{reaction_name}/scaler.joblib")
    atoms_path = Path(f"./dataset/{reaction_name}/atoms.npy")
    runpoint_path = Path(f"./dataset/{reaction_name}/runpoint.npy")

    if (
        not scaler_path.exists()
        or not atoms_path.exists()
        or not runpoint_path.exists()
    ):
        raise FileNotFoundError(
            "One or more required files (scaler.joblib, atoms.npy, runpoint.npy) are missing."
        )

    scaler = joblib.load(scaler_path)
    atoms = np.load(atoms_path)
    runpoint = np.load(runpoint_path)
    logging.info(f"Scaler, atoms, and runpoint loaded for reaction: {reaction_name}")
    return scaler, atoms, runpoint


def initialize_model(
    output_dir: Path,
    data_loader: DataLoader,
    n_steps: int,
    lr: float,
    scaler: Any,
    atoms: np.ndarray,
    runpoint: np.ndarray,
) -> Diffusion:
    """
    Initialize the generative diffusion model with the specified parameters.

    Args:
        output_dir (Path): Directory for saving model outputs.
        data_loader (DataLoader): DataLoader for the training dataset.
        n_steps (int): Number of diffusion steps.
        lr (float): Learning rate for the optimizer.
        scaler (Any): Scaler for normalizing input data.
        atoms (np.ndarray): Atomic data for the model.
        runpoint (np.ndarray): Runpoint information from trajectories.

    Returns:
        Diffusion: Initialized diffusion model instance.
    """
    input_dim = data_loader.dataset[0].shape[0]
    model = Diffusion(
        output_dir=output_dir,
        data_loader=data_loader,
        input_dim=input_dim,
        n_steps=n_steps,
        lr=lr,
        scaler=scaler,
        atoms=atoms,
        runpoint=runpoint,
    )
    logging.info("Initialized diffusion model.")
    return model


def main(args: argparse.Namespace) -> None:
    """
    Main function for managing the training or loading of the diffusion model
    and executing subsequent workflows such as topology conversion and EPS computation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Create output directory
    output_dir = Path(f"./output/{args.reaction}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = Path(args.output_dir) / "execution.log" if args.output_dir else None
    setup_logging(log_level=args.log_level, log_file=log_file)
    logging.info(f"Output directory set to: {output_dir}")

    # Load dataset
    dataset = ReactionDynamicsDataset(args.reaction)
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    logging.info(f"Loaded dataset with {len(dataset)} samples.")

    # Load scaler, atoms, and runpoint
    scaler, atoms, runpoint = load_scaler_and_atoms(args.reaction)

    # Initialize diffusion model
    n_steps = math.ceil(600 / args.num_traj)
    model = initialize_model(
        output_dir=output_dir,
        data_loader=data_loader,
        n_steps=n_steps,
        lr=args.lr,
        scaler=scaler,
        atoms=atoms,
        runpoint=runpoint,
    )

    # Train or load the model
    data_np_path = Path(f"./dataset/{args.reaction}/data.npy")
    if not data_np_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_np_path}")

    all_data = np.load(data_np_path).astype(np.float32)

    if args.train:
        logging.info("Starting training of the diffusion model.")
        epochs = 2000 // args.num_traj
        model.train(epochs=epochs)
        num_samples = 6000 // args.num_traj
        model.diverse(all_data=all_data, num_samples=num_samples)
    else:
        model_path = output_dir / "diffusion_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model.load_model(model_path)
        logging.info(f"Loaded model from {model_path}")
        num_samples = 6000 // args.num_traj
        model.diverse(all_data=all_data, num_samples=num_samples)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train and analyze a generative diffusion model."
    )
    parser.add_argument(
        "--num_traj",
        type=int,
        default=100,
        help="Number of trajectories for the reaction.",
    )
    parser.add_argument(
        "--reaction", type=str, required=True, help="Name of the reaction directory."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Flag to indicate training mode. If not set, the model will be loaded.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )

    args = parser.parse_args()
    main(args)
