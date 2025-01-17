# converters/xyz2bat.py

import os
import numpy as np
import networkx as nx
import logging
from pathlib import Path
import mdtraj as md
from typing import List, Tuple

class XYZ2BATconverter:
    """
    Converter class to transform XYZ files into BAT format based on bond information.

    Args:
        ts_file (str): Path to the transition state PDB file.
        nb1 (int): First atom number in bond 1.
        nb2 (int): Second atom number in bond 1.
    """
    def __init__(self, ts_file: str, nb1: int, nb2: int):
        self.reference = md.load(ts_file)
        self.nb1 = nb1
        self.nb2 = nb2
        self.graph = self._generate_connectivity()
        self.atom_count, self.topology = self._generate_topology()
        logging.info("XYZ2BATconverter initialized.")

    def _generate_connectivity(self) -> dict:
        """
        Generate connectivity graph from the transition state structure.

        Returns:
            dict: Connectivity graph with atom indices as keys and connected atom indices as values.
        """
        graph = {}
        for bond in self.reference.top.bonds:
            index1, index2 = bond.atom1.index, bond.atom2.index
            graph.setdefault(index1, []).append(index2)
            graph.setdefault(index2, []).append(index1)

        # Ensure bond between nb1 and nb2 exists
        graph.setdefault(self.nb1 - 1, []).append(self.nb2 - 1)
        graph.setdefault(self.nb2 - 1, []).append(self.nb1 - 1)

        logging.debug("Connectivity graph generated.")
        return graph

    def _generate_topology(self) -> Tuple[int, List[Tuple[int, ...]]]:
        """
        Generate topology paths based on connectivity.

        Returns:
            Tuple[int, List[Tuple[int, ...]]]: Number of atoms and list of DOF paths.
        """
        G = nx.Graph(self.graph)
        all_paths = [
            path for i in range(len(self.graph)) for j in range(len(self.graph))
            for path in nx.all_simple_paths(G, source=i, target=j) if len(path) in [2, 3, 4]
        ]

        unique_paths = []
        for items in all_paths:
            path_tuple = tuple(items)
            if path_tuple not in unique_paths and tuple(reversed(path_tuple)) not in unique_paths:
                unique_paths.append(path_tuple)
        unique_paths.sort()
        unique_paths.sort(key=len)

        logging.info(f"Generated topology with {len(unique_paths)} paths.")
        return len(self.graph), unique_paths

    def save_topology(self, output_dir: Path) -> None:
        """
        Save the generated topology to a file.

        Args:
            output_dir (Path): Directory to save the topology file.
        """
        output_path = output_dir / "topology.txt"
        try:
            with open(output_path, "w") as file:
                file.write(f"{self.atom_count}\n")
                for path in self.topology:
                    file.write(" ".join(str(atom + 1) for atom in path) + "\n")
            logging.info(f"Topology saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save topology: {e}")

    def process_xyz_files(self, output_dir: Path, ts: str, atom1: int, atom2: int) -> None:
        """
        Process XYZ files to compute internal coordinates and save DOFs.

        Args:
            output_dir (Path): Directory containing generated XYZ files.
            ts (str): Transition state PDB file name.
            atom1 (int): First atom number in bond 1.
            atom2 (int): Second atom number in bond 1.
        """
        trajectory = md.load_xyz(output_dir / "snapshot.xyz", top=ts)
        self.topology.insert(0, (atom1 - 1, atom2 - 1))  # Add reacting bond to topology

        # Separate DOFs based on their lengths
        bond_list = [path for path in self.topology if len(path) == 2]
        angle_list = [path for path in self.topology if len(path) == 3]
        torsion_list = [path for path in self.topology if len(path) == 4]

        # Compute internal coordinates
        b = md.compute_distances(trajectory, bond_list) * 10.0  # nm to angstrom
        a = md.compute_angles(trajectory, angle_list, periodic=True)  # radians
        t = md.compute_dihedrals(trajectory, torsion_list, periodic=True)  # radians

        # Concatenate all DOFs
        dofs = np.hstack((b, a, t))
        np.save(output_dir / "dof.npy", dofs)
        logging.info(f"DOFs saved to {output_dir / 'dof.npy'}")

