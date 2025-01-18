import os
import math
import numpy as np
import argparse
import logging
from joblib import Parallel, delayed
from multiprocessing import shared_memory
from typing import List, Tuple, Optional

def setup_logging(log_file_path: Optional[str] = None):
    """
    Configure logging for the script.

    Args:
        log_file_path (str, optional): Path to the log file. If None, logs are only printed to the console.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file_path:
        fh = logging.FileHandler(log_file_path, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

def get_jacobian(x_bin_edge: float, xjtype: str,
                y_bin_edge: Optional[float] = None, yjtype: Optional[str] = None) -> float:
    """
    Calculate the Jacobian based on bin edges and types.

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
            ("bond", "bond"): x_bin_edge**2 * y_bin_edge**2,
            ("bond", "angle"): x_bin_edge**2 * math.sin(y_bin_edge),
            ("bond", "torsion"): x_bin_edge**2,
            ("angle", "angle"): math.sin(x_bin_edge) * math.sin(y_bin_edge),
            ("angle", "torsion"): math.sin(x_bin_edge),
            ("torsion", "torsion"): 1.0  # Assuming torsion-torsion has Jacobian 1
        }
        return jacobian_dict.get((xjtype, yjtype), 1.0)

def entropy_1D(dofs: np.ndarray, dof_index: int, indices: np.ndarray, jtype: str) -> float:
    """
    Compute 1D entropy for a given degree of freedom.

    Args:
        dofs (np.ndarray): DOFs array of shape (num_dofs, num_samples).
        dof_index (int): Index of the DOF.
        indices (np.ndarray): Indices of the samples to consider.
        jtype (str): Type of the DOF (bond, angle, torsion).

    Returns:
        float: Computed 1D entropy.
    """
    # Extract relevant data
    data = dofs[dof_index][indices]

    # Calculate histogram
    counts, bin_edges = np.histogram(data, bins=50)
    sample_size = counts.sum()

    if sample_size == 0:
        logging.warning(f"Sample size for DOF index {dof_index} is zero. Returning entropy 0.")
        return 0.0

    prob_density = counts / sample_size
    dx = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + dx / 2

    # Calculate entropy
    entropy_sum = 0.0
    for i in range(len(prob_density)):
        p = prob_density[i]
        if p > 0:
            jacobian = get_jacobian(bin_centers[i], jtype)
            entropy_sum -= p * math.log(p / (jacobian * dx))

    # Correction term
    entropy_sum += (np.count_nonzero(prob_density) - 1) / (2 * sample_size)

    return entropy_sum

def entropy_2D(dofs: np.ndarray, dof_index1: int, dof_index2: int,
              indices: np.ndarray, xjtype: str, yjtype: str) -> float:
    """
    Compute 2D entropy for a pair of degrees of freedom.

    Args:
        dofs (np.ndarray): DOFs array of shape (num_dofs, num_samples).
        dof_index1 (int): Index of the first DOF.
        dof_index2 (int): Index of the second DOF.
        indices (np.ndarray): Indices of the samples to consider.
        xjtype (str): Type of the first DOF (bond, angle, torsion).
        yjtype (str): Type of the second DOF (bond, angle, torsion).

    Returns:
        float: Computed 2D entropy.
    """
    # Extract relevant data
    data1 = dofs[dof_index1][indices]
    data2 = dofs[dof_index2][indices]

    # Calculate 2D histogram
    H, X_bin_edges, Y_bin_edges = np.histogram2d(data1, data2, bins=50)
    sample_size = H.sum()

    if sample_size == 0:
        logging.warning(f"Sample size for DOF indices ({dof_index1}, {dof_index2}) is zero. Returning joint entropy 0.")
        return 0.0

    # Normalize histogram
    H_normalized = H / sample_size
    dx = X_bin_edges[1] - X_bin_edges[0]
    dy = Y_bin_edges[1] - Y_bin_edges[0]
    X_bin_centers = X_bin_edges[:-1] + dx / 2
    Y_bin_centers = Y_bin_edges[:-1] + dy / 2

    # Calculate entropy
    entropy_sum = 0.0
    for row in range(H_normalized.shape[0]):
        for col in range(H_normalized.shape[1]):
            p = H_normalized[row, col]
            if p > 0:
                jacobian = get_jacobian(X_bin_centers[row], xjtype, Y_bin_centers[col], yjtype)
                entropy_sum -= p * math.log(p / (jacobian * dx * dy))

    # Correction term
    entropy_sum += (np.count_nonzero(H_normalized) - 1) / (2 * sample_size)

    return entropy_sum

class EntropicPathSampler:
    """
    A class to perform Entropic Path Sampling (EPS) for computational chemistry simulations.

    Args:
        output_dir (str): Directory where output files are stored.
        bond_max (float): Maximum bond length for the reaction coordinate.
        bond_min (float): Minimum bond length for the reaction coordinate.
        ensemble (int): Number of structural ensembles for EPS.
        temperature (float): Temperature for EPS calculations.
    """
    def __init__(self, output_dir: str, bond_max: float, bond_min: float,
                 ensemble: int, temperature: float):
        self.output_dir = output_dir
        self.bond_max = bond_max
        self.bond_min = bond_min
        self.ensemble = ensemble + 1  # Including the initial ensemble
        self.temperature = temperature
        self.conversion_factor = -1.987204259e-3 * temperature  # kcal/mol

        # Reaction coordinate range
        self.reaction_coordinate = np.linspace(bond_max, bond_min, self.ensemble)

        # Load DOF list
        self.dof_list = self._load_topology()

        # Load DOFs array and transpose to shape (num_dofs, num_samples)
        dof_np_path = os.path.join(self.output_dir, "dof.npy")
        if not os.path.isfile(dof_np_path):
            logging.error(f"dof.npy not found at {dof_np_path}")
            raise FileNotFoundError(f"dof.npy not found at {dof_np_path}")
        self.dofs = np.load(dof_np_path).T

        # Extract atom number and remove from DOF list
        self.atom_num = int(self.dof_list[0][0])
        self.dof_list.pop(0)  # Remove atom count

        # Count the number of bonds, angles, and torsions
        self.bond_num = len([items for items in self.dof_list if len(items) == 2])
        self.angle_num = len([items for items in self.dof_list if len(items) == 3])
        self.torsion_num = len([items for items in self.dof_list if len(items) == 4])

        logging.info(f"Initialized EntropicPathSampler with {self.bond_num} bonds, "
                     f"{self.angle_num} angles, and {self.torsion_num} torsions.")

        # Initialize shared memory for dofs
        self.shm = self._initialize_shared_memory()

    def _load_topology(self) -> List[List[str]]:
        """
        Load topology information from the output directory.

        Returns:
            list: List of degrees of freedom (DOFs) extracted from the topology file.
        """
        topology_file = os.path.join(self.output_dir, "topology.txt")
        if not os.path.isfile(topology_file):
            logging.error(f"topology.txt not found at {topology_file}")
            raise FileNotFoundError(f"topology.txt not found at {topology_file}")

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
        shm = shared_memory.SharedMemory(create=True, size=self.dofs.nbytes)
        # Create a NumPy array backed by shared memory
        shm_dofs = np.ndarray(self.dofs.shape, dtype=self.dofs.dtype, buffer=shm.buf)
        shm_dofs[:] = self.dofs[:]
        logging.info(f"Shared memory created with name {shm.name}")
        return shm

    def compute_eps(self):
        """
        Perform Entropic Path Sampling and compute entropy profiles.

        This function calculates the entropy profiles for each ensemble and outputs the results
        to log files for further analysis.
        """
        react = self.dofs[0]  # Reaction coordinate
        s_profile = []
        sample_sizes = []
        bat = {2: "bond", 3: "angle", 4: "torsion"}

        for e in range(self.ensemble - 1):
            eps_file_name = f"eps_{self.reaction_coordinate[e]:.3f}_{self.reaction_coordinate[e+1]:.3f}.log"
            eps_file_path = os.path.join(self.output_dir, eps_file_name)

            # Select samples within the current ensemble range
            indices = np.argwhere(
                (react >= self.reaction_coordinate[e + 1]) &
                (react < self.reaction_coordinate[e])
            ).flatten()
            sample_sizes.append(len(indices))

            logging.info(f"Processing EPS for range {self.reaction_coordinate[e+1]:.3f} - "
                         f"{self.reaction_coordinate[e]:.3f} with {len(indices)} samples.")

            if len(indices) == 0:
                logging.warning(f"No samples found in range {self.reaction_coordinate[e+1]:.3f} - "
                                f"{self.reaction_coordinate[e]:.3f}. Skipping.")
                continue

            # Compute 1D entropies sequentially
            entropy1d = []
            for i, dof in enumerate(self.dof_list):
                jtype = bat.get(len(dof), "unknown")
                entropy = entropy_1D(self.dofs, i, indices, jtype)
                entropy1d.append(entropy)

            # Prepare DOF pairs for 2D entropy calculations
            dof_pairs = []
            for i in range(len(self.dof_list) - 1):
                for j in range(i + 1, len(self.dof_list)):
                    dof_pairs.append((i, j,
                                      bat.get(len(self.dof_list[i]), "unknown"),
                                      bat.get(len(self.dof_list[j]), "unknown")))

            # Compute 2D entropies in parallel using Joblib
            try:
                entropy2d = Parallel(n_jobs=-1, backend='loky', batch_size=100)(
                    delayed(entropy_2D)(self.dofs, pair[0], pair[1], indices, pair[2], pair[3])
                    for pair in dof_pairs
                )
            except Exception as e:
                logging.error(f"Error during parallel 2D entropy computation: {e}")
                continue

            MI, MI_list, MIST_list, index = 0.0, [], [], 0
            with open(eps_file_path, "w") as f_eps:
                # Write 1D entropies
                for i, dof in enumerate(self.dof_list):
                    f_eps.write(f"{bat.get(len(dof), 'unknown')} {' '.join(str(x) for x in dof)}\t\t{entropy1d[i]:.6f}\n")

                # Write 2D entropies and compute mutual information
                for i in range(len(self.dof_list) - 1):
                    dummy = []
                    for j in range(i + 1, len(self.dof_list)):
                        mutual_info = entropy1d[i] + entropy1d[j] - entropy2d[index]
                        dummy.append(mutual_info)
                        f_eps.write(
                            f"{bat.get(len(self.dof_list[i]), 'unknown')} {' '.join(str(x) for x in self.dof_list[i])}\t"
                            f"{bat.get(len(self.dof_list[j]), 'unknown')} {' '.join(str(x) for x in self.dof_list[j])}\t"
                            f"\t{entropy2d[index]:.6f}\t{mutual_info:.6f}\n"
                        )
                        index += 1
                    MI_list.append(dummy)
                    MIST_list.append(max(dummy))
                    MI += sum(dummy)

                # Calculate entropies and mutual information statistics
                bond_entropy = sum(entropy1d[:self.bond_num]) * (self.atom_num - 1) / self.bond_num
                angle_entropy = sum(entropy1d[self.bond_num:self.bond_num + self.angle_num]) * (self.atom_num - 2) / self.angle_num
                torsion_entropy = sum(entropy1d[self.bond_num + self.angle_num:]) * (self.atom_num - 3) / self.torsion_num
                bond_mist = sum(MIST_list[:self.bond_num]) * (self.atom_num - 2) / (self.bond_num - 1)
                angle_mist = sum(MIST_list[self.bond_num:self.bond_num + self.angle_num]) * (self.atom_num - 3) / (self.angle_num - 1)
                torsion_mist = sum(MIST_list[self.bond_num + self.angle_num:]) * (self.atom_num - 4) / (self.torsion_num - 1)
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
            summary_file_path = os.path.join(self.output_dir, "eps.log")
            with open(summary_file_path, "w") as fw:
                fw.write("Bond Length Range \t Entropy(kcal/mol) \t Snapshots\n")
                for i, s in enumerate(s_profile):
                    fw.write(
                        f"{self.reaction_coordinate[i]:.3f} \t "
                        f"{self.reaction_coordinate[i + 1]:.3f} \t "
                        f"{s:.4f} \t "
                        f"{s - s_profile[0]:.4f} \t "
                        f"{sample_sizes[i]}\n"
                    )
        logging.info(f"EPS summary saved to {summary_file_path}")

        # Clean up shared memory
        try:
            self.shm.close()
            self.shm.unlink()
            logging.info(f"Shared memory {self.shm.name} closed and unlinked.")
        except Exception as e:
            logging.error(f"Error during shared memory cleanup: {e}")

        logging.info("Entropic Path Sampling Completed Successfully.")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Perform Entropic Path Sampling.")
    parser.add_argument("--reaction", type=str, required=True, help="Name of the reaction")
    parser.add_argument("--bondmax", type=float, default=2.900, help="Maximum bond length for EPS")
    parser.add_argument("--bondmin", type=float, default=1.580, help="Minimum bond length for EPS")
    parser.add_argument("--ensemble", type=int, default=10, help="Number of structural ensembles for EPS")
    parser.add_argument("--temperature", type=float, default=298.15, help="Temperature for EPS calculations")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory. Defaults to ./output/{reaction}")
    args = parser.parse_args()

    # Define output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join("./output", args.reaction)
    if not os.path.isdir(output_dir):
        logging.error(f"Output directory '{output_dir}' does not exist. Please create it and add the necessary files.")
        sys.exit(1)

    # Setup logging with both console and file handlers
    log_file = os.path.join(output_dir, "eps_execution.log")
    setup_logging(log_file_path=log_file)

    # Initialize EntropicPathSampler
    sampler = EntropicPathSampler(
        output_dir=output_dir,
        bond_max=args.bondmax,
        bond_min=args.bondmin,
        ensemble=args.ensemble,
        temperature=args.temperature,
    )

    # Perform EPS computation
    sampler.compute_eps()

if __name__ == "__main__":
    main()

