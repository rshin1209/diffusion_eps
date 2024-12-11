import os
import numpy as np
import networkx as nx
import argparse
import mdtraj as md
from concurrent.futures import ProcessPoolExecutor

class XYZ2BATconverter:
    def __init__(self, ts_file, nb1, nb2):
        self.reference = md.load(ts_file)
        self.nb1 = nb1
        self.nb2 = nb2
        self.graph = self._generate_connectivity()
        self.atom_count, self.topology = self._generate_topology()

    def _generate_connectivity(self):
        graph = {}
        for atom1, atom2 in self.reference.top.bonds:
            index1, index2 = atom1.index, atom2.index
            graph.setdefault(index1, []).append(index2)
            graph.setdefault(index2, []).append(index1)

        graph[self.nb1 - 1].append(self.nb2 - 1)
        graph[self.nb2 - 1].append(self.nb1 - 1)
        return graph

    def _generate_topology(self):
        G = nx.Graph(self.graph)
        all_paths = [
            path for i in range(len(self.graph)) for j in range(len(self.graph))
            for path in nx.all_simple_paths(G, source=i, target=j) if len(path) in [2, 3, 4]
        ]

        unique_paths = []
        for items in all_paths:
            if tuple(items) not in unique_paths and tuple(reversed(items)) not in unique_paths:
                unique_paths.append(tuple(items))
        unique_paths.sort()
        unique_paths.sort(key=len)
        return len(self.graph), unique_paths

    def save_topology(self, output_dir):
        output_path = os.path.join(output_dir, "topology.txt")
        with open(output_path, "w") as file:
            file.write(f"{self.atom_count}\n")
            for path in self.topology:
                file.write(" ".join(str(atom + 1) for atom in path) + "\n")

    def process_xyz_files(self, output_dir, ts, atom1, atom2):
        trajectory = md.load_xyz(os.path.join(output_dir, "snapshot.xyz"), top=ts)
        self.topology.insert(0, (atom1-1, atom2-1))    
    
        bond_list = [items for items in self.topology if len(items) == 2]
        angle_list = [items for items in self.topology if len(items) == 3]
        torsion_list = [items for items in self.topology if len(items) == 4]

        b = md.compute_distances(trajectory, bond_list) * 10.0  # nm2ang
        a = md.compute_angles(trajectory, angle_list)
        t = md.compute_dihedrals(trajectory, torsion_list)

        dofs = np.hstack((b, a, t))
        np.save(os.path.join(output_dir, "dof.npy"), dofs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process XYZ files for internal coordinate conversion.")
    parser.add_argument("--atom1", type=int, required=True, help="First atom number in reaction coordinate.")
    parser.add_argument("--atom2", type=int, required=True, help="Second atom number in reaction coordinate.")
    parser.add_argument("--reaction", type=str, required=True, help="Name of the reaction file without format tag.")
    parser.add_argument("--ts", type=str, required=True, help="Name of the optimized transition state structure file (PDB) without format tag.")
    parser.add_argument("--nb1", type=int, required=True, help="First atom number in bond 1.")
    parser.add_argument("--nb2", type=int, required=True, help="Second atom number in bond 1.")

    args = parser.parse_args()

    output_dir = f"./output/{args.reaction}"

    converter = XYZ2BATconverter(f"./dataset/{args.ts}.pdb", args.nb1, args.nb2)
    converter.save_topology(output_dir)
    converter.process_xyz_files(output_dir, f"./dataset/{args.ts}.pdb", args.atom1, args.atom2)

