# entropy/eps.py

import os
import math
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from multiprocessing import shared_memory
from joblib import Parallel, delayed

from utils.shared_memory import create_shared_memory, access_shared_memory

def entropy_2D_shared(shm_name: str, shape: Tuple[int, ...], dtype: np.dtype,
                      dof_index1: int, dof_index2: int,
                      xjtype: str, yjtype: str, indices: np.ndarray) -> float:
    """
    Standalone function to compute mutual information for a pair of DOFs using shared memory.

    Args:
        shm_name (str): Name of the shared memory block.
        shape (Tuple[int, ...]): Shape of the DOFs array.
        dtype (np.dtype): Data type of the DOFs array.
        dof_index1 (int): Index of the first DOF.
        dof_index2 (int): Index of the second DOF.
        xjtype (str): Type of the first DOF (bond, angle, torsion).
        yjtype (str): Type of the second DOF (bond, angle, torsion).
        indices (np.ndarray): Indices of the samples to consider.

    Returns:
        float: Computed mutual information.
    """
    # Access the shared memory block
    dofs = access_shared_memory(shm_name, shape, dtype)
    
    # Extract the relevant data
    data1 = dofs[dof_index1][indices]
    data2 = dofs[dof_index2][indices]
    
    # Compute 2D histogram
    H_XY, X_bin_edges, Y_bin_edges = np.histogram2d(data1, data2, bins=50)
    sample_size = H_XY.sum()

    if sample_size == 0:
        logging.warning(f"Sample size for DOF indices ({dof_index1 + 1}, {dof_index2 + 1}) is zero. Returning mutual information 0.")
        return 0.0

    H_XY_normalized = H_XY / sample_size
    dx = (data1.max() - data1.min()) / 50
    dy = (data2.max() - data2.min()) / 50
    X_bin_centers = np.linspace(data1.min() + dx / 2, data1.max() - dx / 2, 50)
    Y_bin_centers = np.linspace(data2.min() + dy / 2, data2.max() - dy / 2, 50)

    entropy_sum = 0.0
    for row in range(H_XY_normalized.shape[0]):
        for col in range(H_XY_normalized.shape[1]):
            p = H_XY_normalized[row, col]
            if p > 0:
                # Calculate Jacobian
                if yjtype is None:
                    jacobian = EntropicPathSampler.get_jacobian_static(X_bin_centers[row], xjtype)
                else:
                    jacobian = EntropicPathSampler.get_jacobian_static(
                        X_bin_centers[row], xjtype,
                        Y_bin_centers[col], yjtype
                    )
                entropy_sum -= p * math.log(p / (jacobian * dx * dy))

    # Correction term
    entropy_sum += (np.count_nonzero(H_XY_normalized) - 1) / (2 * sample_size)

    # Mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
    # Assuming H(X) and H(Y) are precomputed and stored
    # Here, entropy_sum represents H(X,Y)
    # This function should receive H(X) and H(Y) if needed
    # For simplicity, we'll return entropy_sum which represents H(X,Y)
    # The caller is responsible for computing I(X;Y)

    return entropy_sum


class EntropicPathSampler:
    """
    Entropic Path Sampling (EPS) class for computing entropy profiles.

    Args:
        output_dir (Path): Output directory for saving EPS results.
        bond_max (float): Maximum bond length for the reaction coordinate.
        bond_min (float): Minimum bond length for the reaction coordinate.
        ensemble (int): Number of structural ensembles for EPS.
        temperature (float): Temperature for EPS calculations.
    """
    def __init__(
        self,
        output_dir: Path,
        bond_max: float,
        bond_min: float,
        ensemble: int,
        temperature: float
    ):
        self.output_dir = output_dir
        self.bond_max = bond_max
        self.bond_min = bond_min
        self.ensemble = ensemble + 1  # Including the initial ensemble
        self.temperature = temperature
        self.conversion_factor = -1.987204259e-3 * temperature  # kcal/mol
        self.reaction_coordinate = np.linspace(bond_max, bond_min, self.ensemble)
        self.dof_list = self._load_topology()
        self.dofs = np.load(os.path.join(self.output_dir, "dof.npy")).T  # Shape: (num_dofs, num_samples)
        self.atom_num = int(self.dof_list[0][0])
        self.dof_list.pop(0)  # Remove atom count
        self.bond_num = len([items for items in self.dof_list if len(items) == 2])
        self.angle_num = len([items for items in self.dof_list if len(items) == 3])
        self.torsion_num = len([items for items in self.dof_list if len(items) == 4])
        logging.info(f"Initialized EntropicPathSampler with {self.bond_num} bonds, "
                     f"{self.angle_num} angles, and {self.torsion_num} torsions.")

        # Initialize shared memory for self.dofs
        self.shm = self._initialize_shared_memory()

    def _load_topology(self) -> List[List[str]]:
        """
        Load topology information from the output directory.

        Returns:
            List[List[str]]: List of degrees of freedom (DOFs) extracted from the topology file.
        """
        topology_file = os.path.join(self.output_dir, "topology.txt")
        if not os.path.isfile(topology_file):
            logging.error(f"Topology file not found at {topology_file}")
            raise FileNotFoundError(f"Topology file not found at {topology_file}")

        with open(topology_file, "r") as file:
            dof_list = [line.strip().split() for line in file.readlines()]
        logging.info(f"Loaded topology with {len(dof_list)} DOFs.")
        return dof_list

    def _initialize_shared_memory(self) -> shared_memory.SharedMemory:
        """
        Initialize shared memory for the DOFs array.

        Returns:
            shared_memory.SharedMemory: Shared memory object containing the DOFs array.
        """
        shm = create_shared_memory(self.dofs)
        logging.info(f"Shared memory created with name {shm.name}")
        return shm

    @staticmethod
    def get_jacobian_static(x_bin_edge: float, xjtype: str,
                            y_bin_edge: Optional[float] = None, yjtype: Optional[str] = None) -> float:
        """
        Static method to calculate the Jacobian based on bin edges and types.

        Args:
            x_bin_edge (float): Bin edge value for x.
            xjtype (str): Type of x (bond, angle, torsion).
            y_bin_edge (float, optional): Bin edge value for y.
            yjtype (str, optional): Type of y (bond, angle, torsion).

        Returns:
            float: Jacobian value based on the provided types.
        """
        if y_bin_edge is None and yjtype is None:
            if xjtype == "bond":
                return x_bin_edge ** 2
            elif xjtype == "torsion":
                return 1.0
            elif xjtype == "angle":
                return math.sin(x_bin_edge)
            else:
                logging.warning(f"Unknown jtype: {xjtype}. Defaulting Jacobian to 1.")
                return 1.0
        else:
            jacobian_dict = {
                ("bond", "bond"): (x_bin_edge ** 2) * (y_bin_edge ** 2),
                ("bond", "angle"): (x_bin_edge ** 2) * math.sin(y_bin_edge),
                ("bond", "torsion"): x_bin_edge ** 2,
                ("angle", "angle"): math.sin(x_bin_edge) * math.sin(y_bin_edge),
                ("angle", "torsion"): math.sin(x_bin_edge),
                ("torsion", "torsion"): 1.0
            }
            return jacobian_dict.get((xjtype, yjtype), 1.0)

    def entropy_1D(self, dof_index: int, jtype: str, indices: np.ndarray) -> float:
        """
        Compute 1D entropy for a given degree of freedom.

        Args:
            dof_index (int): Index of the DOF.
            jtype (str): Type of the DOF (bond, angle, torsion).
            indices (np.ndarray): Indices of the samples to consider.

        Returns:
            float: Computed 1D entropy.
        """
        data = self.dofs[dof_index][indices]
        counts, bin_edges = np.histogram(data, bins=50)
        sample_size = counts.sum()

        if sample_size == 0:
            logging.warning(f"Sample size for DOF index {dof_index + 1} is zero. Returning entropy 0.")
            return 0.0

        prob_density = counts / sample_size
        dx = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:-1] + dx / 2

        # Calculate entropy sum
        entropy_sum = 0.0
        for i in range(len(prob_density)):
            p = prob_density[i]
            if p > 0:
                jacobian = self.get_jacobian_static(bin_centers[i], jtype)
                entropy_sum -= p * math.log(p / (jacobian * dx))

        # Correction term
        entropy_sum += (np.count_nonzero(prob_density) - 1) / (2 * sample_size)

        return entropy_sum

    def entropy_2D(self, dof_index1: int, dof_index2: int,
                  xjtype: str, yjtype: str, indices: np.ndarray) -> float:
        """
        Compute mutual information for a pair of degrees of freedom.

        Args:
            dof_index1 (int): Index of the first DOF.
            dof_index2 (int): Index of the second DOF.
            xjtype (str): Type of the first DOF (bond, angle, torsion).
            yjtype (str): Type of the second DOF (bond, angle, torsion).
            indices (np.ndarray): Indices of the samples to consider.

        Returns:
            float: Computed mutual information.
        """
        # The mutual information is calculated as:
        # I(X; Y) = H(X) + H(Y) - H(X, Y)
        # Since H(X, Y) is computed in entropy_2D_shared, we'll assume H(X) and H(Y) are precomputed.

        # Compute H(X, Y)
        h_xy = entropy_2D_shared(
            shm_name=self.shm.name,
            shape=self.dofs.shape,
            dtype=self.dofs.dtype,
            dof_index1=dof_index1,
            dof_index2=dof_index2,
            xjtype=xjtype,
            yjtype=yjtype,
            indices=indices
        )

        # Compute mutual information
        mi = self.entropy1D[dof_index1] + self.entropy1D[dof_index2] - h_xy

        return mi

    def compute_eps(self) -> None:
        """
        Perform Entropic Path Sampling and compute entropy profiles.
        """
        react = self.dofs[0]  # Assuming the first DOF is the reaction coordinate
        s_profile = []
        sample_sizes = []
        bat = {2: "bond", 3: "angle", 4: "torsion"}

        for e in range(self.ensemble - 1):
            eps_file_path = self.output_dir / f"eps_{self.reaction_coordinate[e]:.3f}_{self.reaction_coordinate[e+1]:.3f}.log"
            indices = np.where(
                (react >= self.reaction_coordinate[e + 1]) &
                (react < self.reaction_coordinate[e])
            )[0]
            sample_sizes.append(len(indices))
            logging.info(f"Processing EPS for range {self.reaction_coordinate[e+1]:.3f} - {self.reaction_coordinate[e]:.3f} with {len(indices)} samples.")

            if len(indices) == 0:
                logging.warning(f"No samples found in range {self.reaction_coordinate[e+1]:.3f} - {self.reaction_coordinate[e]:.3f}. Skipping.")
                continue

            # Precompute 1D entropies
            self.entropy1D = [
                self.entropy_1D(i, bat[len(self.dof_list[i])], indices)
                for i in range(len(self.dof_list))
            ]

            # Prepare arguments for 2D entropy calculations
            args = [
                (i, j, bat[len(self.dof_list[i])], bat[len(self.dof_list[j])], indices)
                for i in range(len(self.dof_list) - 1)
                for j in range(i + 1, len(self.dof_list))
            ]

            # Compute mutual information in parallel using Joblib
            mutual_info_values = Parallel(n_jobs=-1, backend='loky', batch_size=1000)(
                delayed(self.entropy_2D)(*arg) for arg in args
            )

            MI, MI_list, MIST_list, index = 0.0, [], [], 0
            with open(eps_file_path, "w") as f_eps:
                # Write 1D entropies
                for i, dof in enumerate(self.dof_list):
                    f_eps.write(f"{bat[len(dof)]} {' '.join(str(x) for x in dof)}\t\t{self.entropy1D[i]:.6f}\n")

                # Write 2D entropies and compute mutual information
                for i in range(len(self.dof_list) - 1):
                    for j in range(i + 1, len(self.dof_list)):
                        mutual_info = mutual_info_values[index]
                        f_eps.write(
                            f"{bat[len(self.dof_list[i])]} {' '.join(str(x) for x in self.dof_list[i])}\t"
                            f"{bat[len(self.dof_list[j])]} {' '.join(str(x) for x in self.dof_list[j])}\t"
                            f"\t{mutual_info_values[index]:.6f}\t{mutual_info_values[index]:.6f}\n"
                        )
                        index += 1
                        MI += mutual_info

                # Calculate entropies and mutual information statistics
                bond_entropy = sum(self.entropy1D[:self.bond_num]) * (self.atom_num - 1) / self.bond_num
                angle_entropy = sum(self.entropy1D[self.bond_num:self.bond_num + self.angle_num]) * (self.atom_num - 2) / self.angle_num
                torsion_entropy = sum(self.entropy1D[self.bond_num + self.angle_num:]) * (self.atom_num - 3) / self.torsion_num
                bond_mist = sum(mutual_info_values[:self.bond_num]) * (self.atom_num - 2) / (self.bond_num - 1)
                angle_mist = sum(mutual_info_values[self.bond_num:self.bond_num + self.angle_num]) * (self.atom_num - 3) / (self.angle_num - 1)
                torsion_mist = sum(mutual_info_values[self.bond_num + self.angle_num:]) * (self.atom_num - 4) / (self.torsion_num - 1)
                total_entropy = bond_entropy + angle_entropy + torsion_entropy - bond_mist - angle_mist - torsion_mist

                # Write summary to EPS log file
                f_eps.write(f"Bond Entropy: {bond_entropy:.6f}\n")
                f_eps.write(f"Angle Entropy: {angle_entropy:.6f}\n")
                f_eps.write(f"Torsion Entropy: {torsion_entropy:.6f}\n")
                f_eps.write(f"MIST: {bond_mist + angle_mist + torsion_mist:.6f}\n")
                f_eps.write(f"Bond Entropy (kcal/mol): {bond_entropy * self.conversion_factor:.6f}\n")
                f_eps.write(f"Angle Entropy (kcal/mol): {angle_entropy * self.conversion_factor:.6f}\n")
                f_eps.write(f"Torsion Entropy (kcal/mol): {torsion_entropy * self.conversion_factor:.6f}\n")
                f_eps.write(f"MIST (kcal/mol): {(bond_mist + angle_mist + torsion_mist) * self.conversion_factor:.6f}\n")
                f_eps.write(f"MIST Entropy (kcal/mol): {total_entropy * self.conversion_factor:.6f}\n")
                s_profile.append(self.conversion_factor * total_entropy)

            # Save EPS summary
            eps_summary_path = self.output_dir / "eps.log"
            with open(eps_summary_path, "w") as fw:
                fw.write("Bond Length Range \t Entropy(kcal/mol) \t Snapshots\n")
                for i, s in enumerate(s_profile):
                    fw.write(
                        f"{self.reaction_coordinate[i]:.3f} \t "
                        f"{self.reaction_coordinate[i + 1]:.3f} \t "
                        f"{s:.4f} \t "
                        f"{s - s_profile[0]:.4f} \t "
                        f"{sample_sizes[i]}\n"
                    )
            logging.info(f"EPS summary saved to {eps_summary_path}")

            # Clean up shared memory
            self.shm.close()
            self.shm.unlink()
            logging.info(f"Shared memory {self.shm.name} closed and unlinked.")

            logging.info("Entropic Path Sampling Completed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Perform Entropic Path Sampling.")
    parser.add_argument("--reaction", type=str, required=True, help="Name of the reaction")
    parser.add_argument("--bondmax", type=float, default=2.900, help="Maximum value of bond")
    parser.add_argument("--bondmin", type=float, default=1.580, help="Minimum value of bond")
    parser.add_argument("--ensemble", type=int, default=10, help="Number of structural ensembles for EPS")
    parser.add_argument("--temperature", type=float, default=298.15, help="Temperature for EPS")

    args = parser.parse_args()

    output_dir = Path(f"./output/{args.reaction}")

    if not output_dir.exists():
        logging.error(f"Output directory {output_dir} does not exist.")
        raise FileNotFoundError(f"Output directory {output_dir} does not exist.")

    sampler = EntropicPathSampler(
        output_dir=output_dir,
        bond_max=args.bondmax,
        bond_min=args.bondmin,
        ensemble=args.ensemble,
        temperature=args.temperature,
    )

    sampler.compute_eps()

