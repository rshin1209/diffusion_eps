import os
import torch
import joblib
import math
import argparse
import numpy as np
from torch.utils.data import DataLoader
from data import ReactionDynamicsDataset
from diffusion import Diffusion
from xyz2bat import XYZ2BATconverter
from eps import EntropicPathSampler

# Set random seeds for reproducibility
torch.manual_seed(42)

def load_scaler_and_atoms(reaction_name):
    """
    Load the scaler, atomic data, and run point information for a given reaction.

    Args:
        reaction_name (str): The name of the reaction directory.

    Returns:
        scaler (joblib object): Scaler for normalizing data.
        atoms (np.ndarray): Atomic coordinates or related data.
        runpoint (np.ndarray): Run point information.
    """
    scaler = joblib.load(f"./dataset/{reaction_name}/scaler.joblib")
    atoms = np.load(f"./dataset/{reaction_name}/atoms.npy")
    runpoint = np.load(f"./dataset/{reaction_name}/runpoint.npy")
    print(f"Scaler, atoms, and runpoint loaded for reaction: {reaction_name}")
    return scaler, atoms, runpoint

def initialize_model(output_dir, data_loader, n_steps, lr, scaler, atoms, runpoint):
    """
    Initialize the generative diffusion model with the specified parameters.

    Args:
        output_dir (str): Directory for saving model outputs.
        data_loader (DataLoader): DataLoader for the training dataset.
        n_steps (int): Number of diffusion steps.
        lr (float): Learning rate for the optimizer.
        scaler (joblib object): Scaler for normalizing input data.
        atoms (np.ndarray): Atomic data for the model.
        runpoint (np.ndarray): Runpoint information from trajectories.

    Returns:
        model (Diffusion): Initialized diffusion model instance.
    """
    model = Diffusion(
        output_dir,
        data_loader,
        input_dim=data_loader.dataset[0].shape[0],
        n_steps=n_steps,
        lr=lr,
        scaler=scaler,
        atoms=atoms,
        runpoint=runpoint
    )
    print("Initialized diffusion model.")
    return model

def main(args):
    """
    Main function for managing the training or loading of the diffusion model
    and executing subsequent workflows such as topology conversion and EPS computation.

    Args:
        args (Namespace): Parsed command-line arguments.
    """
    # Create output directory
    output_dir = f"./output/{args.reaction}"
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset, scaler, and atomic data
    dataset = ReactionDynamicsDataset(args.reaction)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    scaler, atoms, runpoint = load_scaler_and_atoms(args.reaction)

    # Initialize model
    n_steps = math.ceil(600 / args.num_traj)
    model = initialize_model(output_dir, data_loader, n_steps=n_steps, lr=args.lr, scaler=scaler, atoms=atoms, runpoint=runpoint)

    # Train or load the model
    if args.train == 1:
        print("Training the diffusion model...")
        model.train(epochs=(2000 // args.num_traj))
        model.diverse(
            np.load(f"./dataset/{args.reaction}/data.npy").astype(np.float32),
            num_samples=(6000 // args.num_traj)
        )
    else:
        model_path = os.path.join(output_dir, "diffusion_model.pth")
        model.load_model(model_path)
        print(f"Loaded model from {model_path}")
        model.diverse(
            np.load(f"./dataset/{args.reaction}/data.npy").astype(np.float32),
            num_samples=(6000 // args.num_traj)
        )

    # Convert XYZ to BAT format
    converter = XYZ2BATconverter(f"./dataset/{args.ts}.pdb", args.nb1, args.nb2)
    converter.save_topology(output_dir)
    converter.process_xyz_files(output_dir, f"./dataset/{args.ts}.pdb", args.atom1, args.atom2)

    # Compute entropic path sampling (EPS)
    sampler = EntropicPathSampler(
        output_dir=output_dir,
        bond_max=args.bondmax,
        bond_min=args.bondmin,
        ensemble=args.ensemble,
        temperature=args.temperature,
    )
    sampler.compute_eps()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Arguments for training and analyzing a generative model.")
    parser.add_argument("--num_traj", type=int, default=100, help="Number of trajectories for the reaction.")
    parser.add_argument("--reaction", type=str, required=True, help="Name of the reaction directory.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--train", type=int, default=1, help="Flag to indicate training (1) or loading (0) the model.")

    # Reaction coordinate arguments
    parser.add_argument("--atom1", type=int, required=True, help="First atom number in the reaction coordinate.")
    parser.add_argument("--atom2", type=int, required=True, help="Second atom number in the reaction coordinate.")
    parser.add_argument("--ts", type=str, required=True, help="Transition state structure file name (PDB, without extension).")
    parser.add_argument("--nb1", type=int, required=True, help="First atom number for bond 1.")
    parser.add_argument("--nb2", type=int, required=True, help="Second atom number for bond 1.")

    # EPS-specific arguments
    parser.add_argument("--bondmax", type=float, default=2.900, help="Maximum bond length for EPS computation.")
    parser.add_argument("--bondmin", type=float, default=1.580, help="Minimum bond length for EPS computation.")
    parser.add_argument("--ensemble", type=int, default=10, help="Number of structural ensembles for EPS.")
    parser.add_argument("--temperature", type=float, default=298.15, help="Temperature in Kelvin for EPS.")

    args = parser.parse_args()
    main(args)
