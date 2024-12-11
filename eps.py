import os
import sys
import math
import numpy as np
import argparse
from multiprocessing import Pool

class EntropicPathSampler:
    def __init__(self, output_dir, bond_max, bond_min, ensemble, temperature):
        """
        Initialize the EntropicPathSampler class with parameters for entropy calculation.

        Args:
            output_dir (str): Output directory.
            bond_max (float): Maximum bond length for the reaction coordinate.
            bond_min (float): Minimum bond length for the reaction coordinate.
            ensemble (int): Number of structural ensembles for EPS.
            temperature (float): Temperature for EPS calculations.
        """
        self.output_dir = output_dir
        self.bond_max = bond_max
        self.bond_min = bond_min
        self.ensemble = ensemble + 1
        self.temperature = temperature
        self.conversion_factor = -1.987204259e-3 * temperature  # kcal/mol
        self.reaction_coordinate = np.linspace(bond_max, bond_min, self.ensemble)
        self.dof_list = self._load_topology()
        self.dofs = np.load(os.path.join(self.output_dir, "dof.npy")).T
        self.atom_num = int(self.dof_list[0][0])
        self.dof_list.pop(0)
        self.bond_num = len([items for items in self.dof_list if len(items) == 2])
        self.angle_num = len([items for items in self.dof_list if len(items) == 3])
        self.torsion_num = len([items for items in self.dof_list if len(items) == 4])

    def _load_topology(self):
        """
        Load topology information from the output directory.

        Returns:
            list: List of degrees of freedom (DOFs) extracted from the topology file.
        """
        topology_file = os.path.join(self.output_dir, "topology.txt")
        with open(topology_file, "r") as file:
            return [line.split() for line in file.readlines()]

    def get_jacobian(self, x_bin_edge, xjtype, y_bin_edge=None, yjtype=None):
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
            return x_bin_edge**2 if xjtype == "bond" else 1 if xjtype == "torsion" else math.sin(x_bin_edge)
        else:
            jacobian_dict = {
                ("bond", "bond"): x_bin_edge**2 * y_bin_edge**2,
                ("bond", "angle"): x_bin_edge**2 * math.sin(y_bin_edge),
                ("bond", "torsion"): x_bin_edge**2,
                ("angle", "angle"): math.sin(x_bin_edge) * math.sin(y_bin_edge),
                ("angle", "torsion"): math.sin(x_bin_edge),
            }
            return jacobian_dict.get((xjtype, yjtype), 1)

    def entropy_1D(self, opt):
        """
        Compute 1D entropy for a given degree of freedom.

        Args:
            opt (tuple): Contains dof_index, indices, and jtype for entropy calculation.

        Returns:
            float: Computed 1D entropy.
        """
        dof_index, indices, jtype = opt  # Unpack options for clarity

        # Calculate histogram for the specified DOF and indices
        counts, bin_edges = np.histogram(self.dofs[dof_index][indices], bins=50)
        sample_size = np.sum(counts)

        if sample_size == 0:  # Check to avoid division by zero
            return 0

        prob_density = counts / sample_size
        dx = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:-1] + dx / 2  # Shift to bin centers

        # Calculate entropy sum
        entropy_sum = -np.sum([
            prob_density[i] * math.log(prob_density[i] / (self.get_jacobian(bin_centers[i], jtype) * dx))
            for i in range(len(prob_density)) if prob_density[i] != 0
        ])

        # Return total entropy with correction
        return entropy_sum + (np.count_nonzero(prob_density) - 1) / (2 * sample_size)

    def entropy_2D(self, opt):
        """
        Compute 2D entropy for a pair of degrees of freedom.

        Args:
            opt (tuple): Contains dof_index1, dof_index2, indices, xjtype, and yjtype for entropy calculation.

        Returns:
            float: Computed 2D entropy.
        """
        dof_index1, dof_index2, indices, xjtype, yjtype = opt  # Unpack options for clarity

        # Calculate 2D histogram for the specified DOFs and indices
        H, X_bin_edges, Y_bin_edges = np.histogram2d(
            self.dofs[dof_index1][indices],
            self.dofs[dof_index2][indices],
            bins=50
        )
        sample_size = np.sum(H)

        if sample_size == 0:  # Check to avoid division by zero
            return 0

        H /= sample_size  # Normalize the histogram
        dx = X_bin_edges[1] - X_bin_edges[0]
        dy = Y_bin_edges[1] - Y_bin_edges[0]
        X_bin_centers = X_bin_edges[:-1] + dx / 2
        Y_bin_centers = Y_bin_edges[:-1] + dy / 2

        # Calculate entropy sum
        entropy_sum = -np.sum([
            H[row][col] * math.log(H[row][col] / (
                self.get_jacobian(X_bin_centers[row], xjtype, Y_bin_centers[col], yjtype) * dx * dy
            ))
            for row in range(H.shape[0]) for col in range(H.shape[1]) if H[row][col] != 0
        ])

        # Return total entropy with correction
        return entropy_sum + (np.count_nonzero(H) - 1) / (2 * sample_size)

    def compute_eps(self):
        """
        Perform Entropic Path Sampling and compute entropy profiles.

        This function calculates the entropy profiles for each ensemble and outputs the results
        to log files for further analysis.
        """
        react = self.dofs[0]
        s_profile = []
        sample_sizes = []
        bat = {2: "bond", 3: "angle", 4: "torsion"}

        for e in range(self.ensemble - 1):
            eps_file_path = os.path.join(self.output_dir, f"eps_{self.reaction_coordinate[e]:.3f}_{self.reaction_coordinate[e+1]:.3f}.log")
            indices = np.argwhere((react >= self.reaction_coordinate[e + 1]) & (react < self.reaction_coordinate[e])).flatten()
            sample_sizes.append(len(indices))

            entropy1d = [self.entropy_1D((i + 1, indices, bat[len(self.dof_list[i])])) for i in range(len(self.dof_list))]
            with Pool() as pool:
                entropy2d = pool.map(self.entropy_2D, [
                    (i + 1, j + 1, indices, bat[len(self.dof_list[i])], bat[len(self.dof_list[j])])
                    for i in range(len(self.dof_list) - 1) for j in range(i + 1, len(self.dof_list))
                ])

            MI, MI_list, MIST_list, index = 0.0, [], [], 0
            with open(eps_file_path, "w") as f_eps:
                for i, dof in enumerate(self.dof_list):
                    f_eps.write(f"{bat[len(dof)]} {' '.join(str(x) for x in dof)}\t\t{entropy1d[i]:.6f}\n")

                for i in range(len(self.dof_list) - 1):
                    dummy = []
                    for j in range(i + 1, len(self.dof_list)):
                        mutual_info = entropy1d[i] + entropy1d[j] - entropy2d[index]
                        dummy.append(mutual_info)
                        f_eps.write(f"{bat[len(self.dof_list[i])]} {' '.join(str(x) for x in self.dof_list[i])}\t"
                                    f"{bat[len(self.dof_list[j])]} {' '.join(str(x) for x in self.dof_list[j])}\t"
                                    f"\t{entropy2d[index]:.6f}\t{mutual_info:.6f}\n")
                        index += 1
                    MI_list.append(dummy)
                    MIST_list.append(max(dummy))
                    MI += sum(dummy)

                bond_entropy = sum(entropy1d[:self.bond_num]) * (self.atom_num - 1) / self.bond_num
                angle_entropy = sum(entropy1d[self.bond_num:self.bond_num + self.angle_num]) * (self.atom_num - 2) / self.angle_num
                torsion_entropy = sum(entropy1d[self.bond_num + self.angle_num:]) * (self.atom_num - 3) / self.torsion_num
                bond_mist = sum(MIST_list[:self.bond_num]) * (self.atom_num - 2) / (self.bond_num - 1)
                angle_mist = sum(MIST_list[self.bond_num:self.bond_num + self.angle_num]) * (self.atom_num - 3) / (self.angle_num - 1)
                torsion_mist = sum(MIST_list[self.bond_num + self.angle_num:]) * (self.atom_num - 4) / (self.torsion_num - 1)
                total_entropy = bond_entropy + angle_entropy + torsion_entropy - bond_mist - angle_mist - torsion_mist

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

        with open(os.path.join(self.output_dir, "eps.log"), "w") as fw:
            fw.write("Bond Length Range \t Entropy(kcal/mol) \t Snapshots\n")
            for i, s in enumerate(s_profile):
                fw.write(f"{self.reaction_coordinate[i]:.3f} \t {self.reaction_coordinate[i + 1]:.3f} \t {s:.4f} \t {s - s_profile[0]:.4f} \t {sample_sizes[i]}\n")
        os.remove(os.path.join(self.output_dir, "dof.npy"))
        print("Entropic Path Sampling Completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Entropic Path Sampling.")
    parser.add_argument("--reaction", type=str, required=True, help="Name of the reaction")
    parser.add_argument("--bondmax", type=float, default=2.900, help="Maximum value of bond")
    parser.add_argument("--bondmin", type=float, default=1.580, help="Minimum value of bond")
    parser.add_argument("--ensemble", type=int, default=10, help="Number of structural ensembles for EPS")
    parser.add_argument("--temperature", type=float, default=298.15, help="Temperature for EPS")

    args = parser.parse_args()

    output_dir = "./output/{args.reaction}"

    sampler = EntropicPathSampler(
        output_dir=output_dir,
        bond_max=args.bondmax,
        bond_min=args.bondmin,
        ensemble=args.ensemble,
        temperature=args.temperature,
    )

    sampler.compute_eps()

