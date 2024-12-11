import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for time embeddings.

    Args:
        dim (int): Dimensionality of the time embedding.
        theta (int): A scaling factor (default is 10000).
    """
    def __init__(self, dim, theta=10000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
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
    def __init__(self, input_dim, hidden_dim=256, embedding_dim=256):
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

    def forward(self, x, t):
        """
        Forward pass through the MLP model with time embedding.

        Args:
            x (torch.Tensor): Input data tensor.
            t (torch.Tensor): Time step tensor.

        Returns:
            torch.Tensor: Output tensor after processing.
        """
        t_emb = self.time_embedding(t.unsqueeze(1))
        x = torch.cat([x, t_emb], dim=-1)  # Concatenate time embedding with input

        x = self.gelu(self.fc1(x))
        residual = x

        # Residual connections through hidden layers
        x = self.gelu(self.fc2(x)) + residual
        residual = x
        x = self.gelu(self.fc3(x)) + residual
        residual = x
        x = self.gelu(self.fc4(x)) + residual

        # Final output layer
        x = self.fc5(x)
        return x

class Diffusion:
    """
    A Diffusion model class for training and generating samples using an MLP architecture.

    Args:
        output_dir (str): Directory for saving outputs (models, plots, etc.).
        data_loader (DataLoader): DataLoader for the training dataset.
        input_dim (int): Dimensionality of the input data.
        n_steps (int): Number of diffusion steps.
        lr (float): Learning rate for optimization.
        scaler (object): Scaler for data normalization/denormalization.
        atoms (np.ndarray): Atomic data (e.g., atomic symbols).
        runpoint (np.ndarray): Runpoint information from trajectories.
    """
    def __init__(self, output_dir, data_loader, input_dim, n_steps, lr, scaler, atoms, runpoint):
        self.output_dir = output_dir
        self.data_loader = data_loader
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.lr = lr
        self.scaler = scaler
        self.atoms = atoms
        self.runpoint = runpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model, optimizer, and loss function
        self.model = MLP(input_dim).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # Noise schedule
        self.betas, self.alphas, self.alpha_bar = self.noise_schedule()
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alpha_bar = self.alpha_bar.to(self.device)

        self.loss = []

    def noise_schedule(self):
        """
        Define the noise schedule for the diffusion process.

        Returns:
            tuple: Betas, alphas, and cumulative alpha_bar values.
        """
        betas = self.linear_beta_schedule()
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bar

    def linear_beta_schedule(self):
        """
        Create a linearly spaced beta schedule.

        Returns:
            torch.Tensor: Linearly spaced beta values.
        """
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, self.n_steps)

    def add_noise(self, x, t):
        """
        Add Gaussian noise to the input data at a specified time step.

        Args:
            x (torch.Tensor): Input data.
            t (torch.Tensor): Time step indices.

        Returns:
            tuple: Noisy data and the noise added.
        """
        noise = torch.randn_like(x).to(self.device)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1).to(self.device)
        x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    def train(self, epochs):
        """
        Train the diffusion model.

        Args:
            epochs (int): Number of training epochs.
        """
        for epoch in range(epochs):
            avg_epoch_loss = 0.0
            for batch in self.data_loader:
                batch = batch.to(self.device)

                t = torch.randint(self.n_steps, (batch.size(0),)).to(self.device)
                noisy_x, noise = self.add_noise(batch, t.int())
                predicted_noise = self.model(noisy_x, t)

                loss = self.criterion(predicted_noise, noise)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                avg_epoch_loss += loss.item()

            avg_epoch_loss /= len(self.data_loader)
            self.loss.append(avg_epoch_loss)
            print(f"Epochs: [{epoch + 1}/{epochs}] Loss: {avg_epoch_loss:.4f}")

        self.save_model(os.path.join(self.output_dir, "diffusion_model.pth"))
        self.graph_loss()

    def graph_loss(self):
        """
        Plot and save the training loss over epochs.
        """
        plt.plot(self.loss, color="black")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(self.output_dir, "loss.png"))
        plt.show()

    def save_model(self, file_path):
        """
        Save the model and optimizer states.

        Args:
            file_path (str): Path to save the model.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, file_path)
        print(f"Model saved to {file_path}.")

    def load_model(self, file_path):
        """
        Load the model and optimizer states.

        Args:
            file_path (str): Path to load the model from.
        """
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {file_path}.")

    def diverse(self, all_data, num_samples=60):
        """
        Generate diverse samples by applying forward and reverse diffusion.

        Args:
            all_data (np.ndarray): Input dataset for generating samples.
            num_samples (int): Number of samples to generate.
        """
        self.model.eval()
        generated_samples = []

        all_data = torch.from_numpy(all_data).to(self.device)

        with torch.no_grad():
            T = self.n_steps - 1
            for _ in range(num_samples):
                noisy_latent, _ = self.add_noise(all_data, torch.full((all_data.size(0),), T, dtype=torch.long).to(self.device))
                x = noisy_latent
                for step in reversed(range(self.n_steps)):
                    t_tensor = torch.full((x.size(0),), step, dtype=torch.long).to(self.device)
                    predicted_noise = self.model(x, t_tensor)

                    beta_t = self.betas[step]
                    alpha_t = self.alphas[step]
                    alpha_bar_t = self.alpha_bar[step]

                    alpha_bar_t_prev = self.alpha_bar[step - 1] if step > 0 else torch.tensor(1.0).to(self.device)

                    mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
                    beta_tilde = beta_t * ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t))

                    if step > 0:
                        z = torch.randn_like(x).to(self.device)
                        x = mean + torch.sqrt(beta_tilde) * z
                    else:
                        x = mean

                generated_samples.append(x.cpu().numpy())

        generated_samples = np.concatenate(generated_samples, axis=0)
        generated_samples = self.scaler.inverse_transform(generated_samples)
        coordinates, energies = generated_samples[:, :-1], generated_samples[:, -1]
        coordinates = coordinates.reshape(len(coordinates), len(self.atoms), 3)

        self.save_generated_data(coordinates, energies)
        self.model.train()

    def save_generated_data(self, coordinates, energies):
        """
        Save generated coordinates and energies to a .xyz file.

        Args:
            coordinates (np.ndarray): Generated coordinates of atoms.
            energies (np.ndarray): Generated energies.
        """
        snapshot_path = os.path.join(self.output_dir, "snapshot.xyz")
        with open(snapshot_path, 'w') as fw:
            for i in range(len(coordinates)):
                fw.write(f"{len(self.atoms)}\n")
                fw.write(f"{energies[i]} runpoint {int(self.runpoint[i % len(self.runpoint)])}\n")
                for a in range(len(self.atoms)):
                    fw.write(f"{self.atoms[a]} {coordinates[i][a][0]:.6f} {coordinates[i][a][1]:.6f} {coordinates[i][a][2]:.6f}\n")
        print(f"Generated data saved to {snapshot_path}")
