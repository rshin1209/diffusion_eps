# models/diffusion.py

import math
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from pathlib import Path
from utils.shared_memory import create_shared_memory, access_shared_memory

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for time embeddings.

    Args:
        dim (int): Dimensionality of the time embedding.
        theta (int): A scaling factor (default is 10000).
    """
    def __init__(self, dim: int, theta: int = 10000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute sinusoidal positional encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1).

        Returns:
            torch.Tensor: Positional encoded tensor of shape (batch_size, dim).
        """
        half_dim = self.dim // 2
        emb_scale = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(-emb_scale * torch.arange(half_dim, device=x.device))
        emb = x * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) model with residual connections and sinusoidal time embeddings.

    Args:
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Dimensionality of hidden layers (default is 256).
        embedding_dim (int): Dimensionality of the time embedding (default is 256).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, embedding_dim: int = 256):
        super(MLP, self).__init__()
        self.time_embedding = SinusoidalPositionalEncoding(dim=embedding_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(embedding_dim + input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)

        # Activation function
        self.gelu = nn.GELU()

        # Initialize weights
        #self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize model weights using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP model with time embedding.

        Args:
            x (torch.Tensor): Input data tensor of shape (batch_size, input_dim).
            t (torch.Tensor): Time step tensor of shape (batch_size,).

        Returns:
            torch.Tensor: Output tensor after processing of shape (batch_size, input_dim).
        """
        t_emb = self.time_embedding(t.unsqueeze(1))  # Shape: (batch_size, embedding_dim)
        x = torch.cat([x, t_emb], dim=-1)  # Concatenate time embedding with input

        x = self.gelu(self.fc1(x))  # Shape: (batch_size, hidden_dim)
        residual = x  # Residual connection

        # Residual connections through hidden layers
        x = self.gelu(self.fc2(x)) + residual  # Shape: (batch_size, hidden_dim)
        residual = x
        x = self.gelu(self.fc3(x)) + residual
        residual = x
        x = self.gelu(self.fc4(x)) + residual

        # Final output layer
        x = self.fc5(x)  # Shape: (batch_size, input_dim)
        return x


class Diffusion:
    """
    A Diffusion model class for training and generating samples using an MLP architecture.

    Args:
        output_dir (Path): Directory for saving outputs (models, plots, etc.).
        data_loader (DataLoader): DataLoader for the training dataset.
        input_dim (int): Dimensionality of the input data.
        n_steps (int): Number of diffusion steps.
        lr (float): Learning rate for optimization.
        scaler (object): Scaler for data normalization/denormalization.
        atoms (np.ndarray): Atomic data (e.g., atomic symbols).
        runpoint (np.ndarray): Runpoint information from trajectories.
    """
    def __init__(
        self,
        output_dir: Path,
        data_loader: torch.utils.data.DataLoader,
        input_dim: int,
        n_steps: int,
        lr: float,
        scaler,
        atoms: np.ndarray,
        runpoint: np.ndarray
    ):
        self.output_dir = output_dir
        self.data_loader = data_loader
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.lr = lr
        self.scaler = scaler
        self.atoms = atoms
        self.runpoint = runpoint

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_model()

    def _setup_model(self) -> None:
        """
        Initialize the model, optimizer, loss function, and noise schedule.
        """
        # Initialize model, optimizer, and loss function
        self.model = MLP(self.input_dim).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # Noise schedule
        self.betas, self.alphas, self.alpha_bar = self.noise_schedule()
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alpha_bar = self.alpha_bar.to(self.device)

        self.loss_history: List[float] = []
        logging.info("Model, optimizer, and loss function initialized.")

    def noise_schedule(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Define the noise schedule for the diffusion process.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Betas, alphas, and cumulative alpha_bar values.
        """
        betas = self.linear_beta_schedule()
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        logging.debug("Noise schedule created.")
        return betas, alphas, alpha_bar

    def linear_beta_schedule(self) -> torch.Tensor:
        """
        Create a linearly spaced beta schedule.

        Returns:
            torch.Tensor: Linearly spaced beta values of shape (n_steps,).
        """
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, self.n_steps)
        logging.debug(f"Linear beta schedule created with shape: {betas.shape}")
        return betas

    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add Gaussian noise to the input data at a specified time step.

        Args:
            x (torch.Tensor): Input data of shape (batch_size, input_dim).
            t (torch.Tensor): Time step indices of shape (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Noisy data and the noise added.
        """
        noise = torch.randn_like(x).to(self.device)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1).to(self.device)  # Shape: (batch_size, 1)
        x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    def train(self, epochs: int) -> None:
        """
        Train the diffusion model.

        Args:
            epochs (int): Number of training epochs.
        """
        self.model.train()
        logging.info(f"Starting training for {epochs} epochs.")
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(self.data_loader, 1):
                batch = batch.to(self.device)  # Shape: (batch_size, input_dim)

                # Sample random time steps for each sample in the batch
                t = torch.randint(0, self.n_steps, (batch.size(0),), device=self.device).long()

                # Add noise to the data
                noisy_x, noise = self.add_noise(batch, t)

                # Predict the noise using the model
                predicted_noise = self.model(noisy_x, t)

                # Compute loss
                loss = self.criterion(predicted_noise, noise)
                loss.backward()

                # Optimize
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()

                if batch_idx % 100 == 0:
                    logging.info(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / len(self.data_loader)
            self.loss_history.append(avg_epoch_loss)
            logging.info(f"Epoch [{epoch}/{epochs}] Average Loss: {avg_epoch_loss:.4f}")

        # Save the trained model and plot loss
        self.save_model(self.output_dir / "diffusion_model.pth")
        self.plot_loss()

    def plot_loss(self) -> None:
        """
        Plot and save the training loss over epochs.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, color="blue", label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        loss_plot_path = self.output_dir / "loss_plot.png"
        plt.savefig(loss_plot_path)
        plt.close()
        logging.info(f"Loss plot saved to {loss_plot_path}")

    def save_model(self, file_path: Path) -> None:
        """
        Save the model and optimizer states.

        Args:
            file_path (Path): Path to save the model.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history
        }
        torch.save(checkpoint, file_path)
        logging.info(f"Model and optimizer states saved to {file_path}")

    def load_model(self, file_path: Path) -> None:
        """
        Load the model and optimizer states.

        Args:
            file_path (Path): Path to load the model from.
        """
        if not file_path.exists():
            logging.error(f"Model file not found at {file_path}")
            raise FileNotFoundError(f"Model file not found at {file_path}")

        checkpoint = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_history = checkpoint.get('loss_history', [])
        self.model.to(self.device)
        logging.info(f"Model and optimizer states loaded from {file_path}")

    def diverse(self, all_data: np.ndarray, num_samples: int = 60) -> None:
        """
        Generate diverse samples by applying forward and reverse diffusion.

        Args:
            all_data (np.ndarray): Input dataset for generating samples.
            num_samples (int, optional): Number of samples to generate. Defaults to 60.
        """
        self.model.eval()
        generated_samples = []

        # Normalize the data
        all_data_tensor = torch.from_numpy(all_data).float().to(self.device)

        with torch.no_grad():
            for _ in range(num_samples):
                # Start from the last diffusion step
                t = torch.full((all_data_tensor.size(0),), self.n_steps - 1, device=self.device, dtype=torch.long)
                noisy_latent, _ = self.add_noise(all_data_tensor, t)
                x = noisy_latent

                # Reverse diffusion process
                for step in reversed(range(self.n_steps)):
                    t_step = torch.full((x.size(0),), step, device=self.device, dtype=torch.long)
                    predicted_noise = self.model(x, t_step)

                    beta_t = self.betas[step]
                    alpha_t = self.alphas[step]
                    alpha_bar_t = self.alpha_bar[step]

                    if step > 0:
                        alpha_bar_t_prev = self.alpha_bar[step - 1]
                    else:
                        alpha_bar_t_prev = torch.tensor(1.0, device=self.device)

                    # Compute mean
                    mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)

                    # Compute variance
                    beta_tilde = beta_t * ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t))

                    # Sample from Gaussian
                    if step > 0:
                        z = torch.randn_like(x).to(self.device)
                        x = mean + torch.sqrt(beta_tilde) * z
                    else:
                        x = mean

                generated_samples.append(x.cpu().numpy())

        # Concatenate all generated samples
        generated_samples = np.concatenate(generated_samples, axis=0)

        # Denormalize the data
        generated_samples = self.scaler.inverse_transform(generated_samples)

        # Separate coordinates and energies
        coordinates, energies = generated_samples[:, :-1], generated_samples[:, -1]
        coordinates = coordinates.reshape(len(coordinates), len(self.atoms), 3)

        # Save generated data
        self.save_generated_data(coordinates, energies)
        self.model.train()

    def save_generated_data(self, coordinates: np.ndarray, energies: np.ndarray) -> None:
        """
        Save generated coordinates and energies to a .xyz file.

        Args:
            coordinates (np.ndarray): Generated coordinates of atoms with shape (num_samples, num_atoms, 3).
            energies (np.ndarray): Generated energies with shape (num_samples,).
        """
        snapshot_path = self.output_dir / "snapshot.xyz"
        try:
            with open(snapshot_path, 'w') as fw:
                for i in range(len(coordinates)):
                    fw.write(f"{len(self.atoms)}\n")
                    fw.write(f"{energies[i]:.6f} runpoint {int(self.runpoint[i % len(self.runpoint)])}\n")
                    for a in range(len(self.atoms)):
                        fw.write(f"{self.atoms[a]} {coordinates[i][a][0]:.6f} {coordinates[i][a][1]:.6f} {coordinates[i][a][2]:.6f}\n")
            logging.info(f"Generated data saved to {snapshot_path}")
        except Exception as e:
            logging.error(f"Failed to save generated data: {e}")

