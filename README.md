# Diffusion Models for Efficient Energy and Entropy Analysis in Post-Transition-State Bifurcations
## ACS Spring 2025 Abstract
Post-transition-state bifurcation (PTSB) is a phenomenon where a single transition state (TS) can lead to the formation of multiple products. Understanding the mechanisms underlying PTSB requires a comprehensive analysis of both post-TS energy and entropy profiles for each bifurcating product. In previous work, we developed an [accelerated entropic path sampling (EPS)](https://pubs.acs.org/doi/10.1021/acs.jpcb.3c01202) approach using a bidirectional generative adversarial network (BGAN) model, which allowed for entropy evaluation with 100–200 reaction trajectories per product. However, as ambimodal selectivity becomes more skewed, the method’s effectiveness decreases. Collecting over 100 trajectories for the minor product becomes computationally expensive, and with fewer trajectories, the BGAN model’s ability to accurately generate molecular configurations diminishes, leading to incomplete entropy convergence. To address these limitations, we applied a Denoising Probabilistic Diffusion Model to generate molecular structures and corresponding energies that are statistically indistinguishable from those sampled during reaction dynamics simulations. With limited datasets, diffusion models offer more stable training, better mode coverage, and higher-quality synthetic data generation than adversarial neural networks, enabling more robust sampling of molecular structures. Crucially, this approach requires only 100–200 trajectories in total, covering both major and minor product pathways, thereby significantly reducing computational costs. Using asymmetric PTSB reactions as a case study, we demonstrate how this model enhances the efficiency of entropic path sampling and enables effective energy and entropy analysis.

## requirement.txt
        numpy
        scipy
        torch
        matplotlib
        networkx
        mdtraj
        joblib
        tqdm
        scikit-learn

## How to perform Diffusion-EPS

## Example Reaction: Diene/Triene Cycloaddition (provided in "dataset" folder)
<p align="center">
<img src="https://github.com/rshin1209/bgan_eps/assets/25111091/e78b318a-37d5-40ee-a6d3-b747457b03f3", width=50%>
</p>

The diene/triene cycloaddition is an ambimodal pericyclic reaction involving butadiene with hexatriene. It yields two products with asynchronous bond formations: 4+2-adduct (bond 1 and bond 2) and 6+4-adduct (bond 1 and bond 3)

<p align="center">
<img src="https://github.com/rshin1209/bgan_eps/assets/25111091/45e297e2-09dc-403d-908d-0f97f43d66bb", width=50%>
</p>

## The overview of Diffusion-EPS

<p align="center">
<img src="https://github.com/user-attachments/assets/55995b3a-0d4d-4308-ba7a-b8ee06864691" width=100%>
</p>
Figure 1. The overview of Diffusion-EPS

### Step 1: Dataset Preparation
#### Step 1.1: Quasiclassical Trajectory Simulation
        Functional/Basis Set: B3LYP-D3/6-31G(d)
        Integration Time Step: 1 fs
        Temperature: 298.15 K
        Bond Formation Cutoff: < 1.550 Å

Files to prepare:
1. Reaction Dynamics Trajectories in .xyz extension (e.g., ./trajectory/dta_r2p_1/traj2.xyz).
2. Optimized TS structure file in pdb format (e.g., ./trajectory/dta_r2p_TS.pdb).

#### Step 1.2: Dataset Scaling and Preparation
prepdataset.py extracts transition-state structure aligned cartesian coordinates of structures and corresponding energies for each snapsho. Then, it uses sklearn.preprocessing.StandardScaler to scale each column of dataset for training.

reaction_name, tsfile_name, and energy_index must be specified in prepdataset.py.

        def main():
          reaction_name = 'dta_r2p_1'
          tsfile_name = 'dta_r2p_TS.pdb'
          energy_index = -1 # The last item (0-indexed) of 'Progdyn_2017   dynamics   trajectory   diene_triene_cycloaddition   runpoint    50   runisomer    21   E:    -389.299517092'

        python prepdataset.py

### Step 2: Diffusion-assisted Configurational Sampling
main.py performs Diffusion Model training.

        python main.py --reaction dta_r2p_1 --num_traj 10 --train
        [reaction] -- The name of reaction directory
        [num_traj] -- The number of trajectories sampled from quasiclassical trajectory simulation
        [train] -- Flag to indicate training mode. If not set, the model will be loaded.

### Step 3: Entropy Analysis
#### Step 3.1: Coordinate Conversion

xyz2bat.py converts Cartesian coordinates of generated snapshots into redundant internal coordinates based on bonding connectivity. The resulting internal coordinates are saved in a 2D numpy array (e.g., ./output/dta_r2p_1/dofs.npy) with rows of snapshots and columns of internal coordinates.

        python xyz2bat.py --nb1 1 --nb2 10 --atom1 2 --atom2 5 --reaction dta_r2p_1
        [nb1] -- first atom number in bond 1
        [nb2] -- second atom number in bond 1
        [atom1] -- first atom number in reaction coordinate (e.g., bond 2 or bond 3)
        [atom2] -- second atom number in reaction coordinate (e.g., bond 2 or bond 3)
        [reaction] -- The name of reaction directory

#### Step 3.2: Entropy Sampling
entropy_sampler.py computes 1D-entropies of each DoF and Mutual Information (MI) of the pairs of DoFs for each structural window and store as a numpy matrix.

        python entropy_sampler.py --reaction dta_r2p_1 --bondmax 2.790 --bondmin 1.602 --ensemble 10
        [reaction] -- The name of reaction directory
        [bondmax] -- Maximum value of the reacting bond.
        [bondmin] -- Minimum value of the reacting bond.
        [ensemble] -- Number of ensembles to divide the reaction coordinate into.

#### Step 3.3: Entropy Compiling
entropy_compiler.py computes configurational entropy profiles based on entropy matrices for each structural window sampled in Step 3.2.

        python entropy_sampler.py --reaction dta_r2p_1 --temperature 298.15
        [reaction] -- The name of reaction directory
        [temperature] -- Temperature in Kelvin.

<p align="center">
<img src = "https://github.com/user-attachments/assets/7beb9d89-765b-40f3-afa7-e6f2c2121015", width=50%>
</p>

Figure 2. Benchmark of Diffusion-EPS. The entropy and energy profiles of dta_r2p_1 and dta_r2p_2 were calculated with EPS protocol using 1961 trajectories for each bond formation. Those of dta_r2p_1_gen and dta_r2p_2_gen were calculated using Diffusion-EPS.

## Contact
Please open an issue on GitHub or contact wook.shin@vanderbilt.edu if you encounter any issues or have concerns.

## Citation
Shin, W.; Ran, X.; Yang, Z. J. Accelerated Entropic Path Sampling with a Bidirectional Generative Adversarial Network. The Journal of Physical Chemistry B 2023, 127 (19), 4254-4260. DOI: 10.1021/acs.jpcb.3c01202.
Wook Shin, Yaning Hou, Xin Wang*, and Zhongyue J. Yang*. Interplay between Energy and Entropy Mediates Ambimodal Selectivity of Cycloadditions.” J. Chem. Theory Comput. 2024, 20, 24, 10942–10951.


## License
This project is licensed under the MIT License - see the LICENSE file for details.
