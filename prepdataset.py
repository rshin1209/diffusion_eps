import os
import shutil
import numpy as np
import mdtraj as md
from sklearn.preprocessing import StandardScaler
import joblib

def parse_snapshots(reaction_dir, runpoint_threshold=1):
    """Parse XYZ trajectory files and collect snapshots where runpoint > threshold after second runpoint = 1 occurrence."""
    snapshot_data = []
    runpoints = []
    filelist = [f for f in os.listdir(reaction_dir) if f.endswith('.xyz')]

    for file in filelist:
        with open(os.path.join(reaction_dir, file), 'r') as fr:
            lines = fr.readlines()
            frame_length = int(lines[0]) + 2
            runpoint_one_count = 0

            for i in range(len(lines) // frame_length):
                block = lines[i * frame_length : (i + 1) * frame_length]
                runpoint_idx = block[1].split().index('runpoint') + 1
                runpoint_val = int(block[1].split()[runpoint_idx])

                if runpoint_val == 1:
                    runpoint_one_count += 1

                if runpoint_val > runpoint_threshold and runpoint_one_count >= 2:
                    snapshot_data.append(block)
                    runpoints.append(runpoint_val)

    return snapshot_data, np.array(runpoints)

def save_combined_xyz(snapshot_data, output_xyz):
    """Save combined snapshots into a single XYZ file."""
    with open(output_xyz, 'w') as fw:
        for block in snapshot_data:
            fw.writelines(block)

def extract_energy_from_xyz(xyz_file, eindex=0):
    """Extract energy values from XYZ trajectory file headers."""
    energy_values = []
    with open(xyz_file, 'r') as fr:
        lines = fr.readlines()
        frame_length = int(lines[0]) + 2

        for i in range(len(lines) // frame_length):
            block = lines[i * frame_length : (i + 1) * frame_length]
            energy_values.append(float(block[1].split()[eindex]))

    atom_names = [line.split()[0] for line in lines[2:frame_length]]
    return np.array(energy_values), atom_names

def process_trajectory(tsfile, combined_xyz_file):
    """Load trajectory, align to transition state, convert units, and flatten coordinates."""
    ts = md.load(tsfile)
    trajectory = md.load_xyz(combined_xyz_file, top=tsfile)
    trajectory.superpose(ts)

    return np.array([frame.flatten() * 10.0 for frame in trajectory.xyz])  # Convert nm to Å

def main():
    reaction_name = 'dta_r2p_1'
    tsfile_name = 'dta_r2p_TS.pdb'
    energy_index = -1

    reaction_dir = os.path.join('trajectory', reaction_name)
    dataset_dir = os.path.join('dataset', reaction_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Copy TSFILE to dataset directory
    shutil.copy(os.path.join(reaction_dir, tsfile_name), dataset_dir)

    snapshot_data, runpoints = parse_snapshots(reaction_dir)
    np.save(os.path.join(dataset_dir, 'runpoint.npy'), runpoints)

    combined_xyz = os.path.join(dataset_dir, f'{reaction_name}.xyz')
    save_combined_xyz(snapshot_data, combined_xyz)

    energy, atom_names = extract_energy_from_xyz(combined_xyz, eindex=energy_index)
    tsfile_path = os.path.join(dataset_dir, tsfile_name)
    xyz = process_trajectory(tsfile_path, combined_xyz)

    data = np.hstack((xyz, energy.reshape(-1, 1)))
    scaler = StandardScaler().fit(data)
    scaled_data = scaler.transform(data)

    np.save(os.path.join(dataset_dir, 'data.npy'), scaled_data)
    np.save(os.path.join(dataset_dir, 'atoms.npy'), np.array(atom_names))
    joblib.dump(scaler, os.path.join(dataset_dir, 'scaler.joblib'))

    print(f"Processing completed. Dataset and TS file saved in {dataset_dir}.")

if __name__ == "__main__":
    main()
