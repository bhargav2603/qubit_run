"""Everything this study takes from the paper, in one place.

Reference
---------
Danil Kaliakin, Akhil Shajan, Javier Robledo Moreno, Zhen Li, Arnab Mitra,
Mario Motta, Caleb Johnson, Abdullah Ash Saki, Susanta Das, Iskandar Sitdikov,
Antonio Mezzacapo, Kenneth M. Merz Jr.,
*Accurate quantum-centric simulations of supramolecular interactions*,
arXiv:2410.09209v2 (2024); Communications Physics (2025).
https://arxiv.org/abs/2410.09209

The rule for this file: a value belongs here **only** if it is printed in that
paper.  Anything this folder had to choose for itself lives in ``geometry.py``
or is a command-line flag, and is reported as ours rather than theirs.  The
distinction matters, because the paper does not publish the one input that most
strongly determines the answer -- see ``GEOMETRY_GAP`` below.
"""

from __future__ import annotations

ARXIV_ID = "2410.09209"
ARXIV_VERSION = "v2"
TITLE = "Accurate quantum-centric simulations of supramolecular interactions"
JOURNAL = "Communications Physics (2025)"
URL = "https://arxiv.org/abs/2410.09209"

# --------------------------------------------------------------------------
# Table I -- active spaces
# --------------------------------------------------------------------------
# "species | active space | AOs"
#   methane dimer | (16e,16o) | C[2s,2p], H[1s]
#
# The AO list is the input to PySCF's AVAS routine, not a post-hoc description.
AVAS_AO_LABELS = ("C 2s", "C 2p", "H 1s")

N_ELECTRONS = 16
N_ORBITALS = 16
N_ALPHA = 8
N_BETA = 8

# 16 spatial orbitals -> 32 qubits carry occupation numbers.  Transpiling the
# LUCJ ansatz onto heavy-hex adds 4 ancillas for the alpha-beta density-density
# interactions, which is how the paper arrives at its stated figure.
N_QUBITS_OCCUPATION = 2 * N_ORBITALS          # 32
N_QUBITS_ANCILLA = 4                          # Figure 2B
N_QUBITS_TOTAL = N_QUBITS_OCCUPATION + N_QUBITS_ANCILLA   # 36

BASIS = "aug-cc-pvqz"
# Chosen in the paper because Metz et al. and Li et al. showed
# CCSD(T)/aug-cc-pVQZ reproduces the CCSD(T)/CBS limit for these dimers.

# --------------------------------------------------------------------------
# Supplementary Information, section I -- the geometry grid
# --------------------------------------------------------------------------
# "The PES for the methane dimer is calculated for the distances between two
#  carbon atoms ranging between 2.500 and 6.000 Angstrom."
PES_DISTANCES = (
    2.500, 2.750, 3.000, 3.167, 3.334, 3.500,
    3.667, 3.834, 4.000, 4.250, 4.500, 4.750,
    5.000, 6.000,
)

# "The methane dimer CASCI (16e,16o) simulations and SQD (16e,16o) simulations
#  with |chi_b| = 20.0e3 are also performed for an additional distance of
#  3.638 Angstrom."
#
# 3.638 is also the single distance used for the SQD (16e,24o) 54-qubit run and
# the anchor of SI Figure S1, which is what marks it as the equilibrium point.
EQUILIBRIUM_DISTANCE = 3.638

# "To calculate the total energy of unbound dimer we utilize the distance of
#  48.000 Angstrom for both water and methane dimers."
UNBOUND_DISTANCE = 48.000

# Distances carrying a full CASCI + SQD(|chi_b|=20e3) treatment.
FULL_TREATMENT_DISTANCES = tuple(
    sorted(PES_DISTANCES + (EQUILIBRIUM_DISTANCE, UNBOUND_DISTANCE))
)

# "The SQD (16e,16o) energy extrapolations using |chi_b| of 9.0e3, 11.0e3 and
#  14.0e3 are done for 4.000, 4.250, 4.500, 4.750, 5.000, 6.000 and 48.000 A."
EXTRAPOLATION_DISTANCES = (4.000, 4.250, 4.500, 4.750, 5.000, 6.000, 48.000)
EXTRAPOLATION_SAMPLE_SIZES = (9_000, 11_000, 14_000)

# --------------------------------------------------------------------------
# Table II -- SQD parameters for the (16e,16o) methane dimer row
# --------------------------------------------------------------------------
#   species        AS         |chi|[10^3]  K   |chi_b|[10^3]  d        CPUs, code  steps
#   methane dimer  (16e,16o)  200          10  20             12.6e7   10, PySCF   10
TOTAL_SAMPLES = 200_000          # |chi|   : shots drawn from the device
N_BATCHES = 10                   # K       : independent subspaces per recovery step
SAMPLES_PER_BATCH = 20_000       # |chi_b| : configurations drawn into each batch
SUBSPACE_DIMENSION = 126_000_000  # d      : 12.6e7, reported at |chi_b| = 20e3
RECOVERY_STEPS = 10              # iterations of self-consistent configuration recovery
PAPER_CPUS = 10                  # one CPU per batch, via Ray 2.33.0
PAPER_SCI_SOLVER = "PySCF"

# --------------------------------------------------------------------------
# Error mitigation (Methods, section II.2)
# --------------------------------------------------------------------------
# "gate twirling (but not measurement twirling) over random 2-qubit Clifford
#  gates and dynamical decoupling, as available via the SamplerV2 primitive"
GATE_TWIRLING = True
MEASUREMENT_TWIRLING = False
DYNAMICAL_DECOUPLING = True
PAPER_DEVICE = "ibm_cleveland"

# --------------------------------------------------------------------------
# Results the reproduction is measured against
# --------------------------------------------------------------------------
# Equation (2): the binding energy is a difference of *dimer* energies, never
# E_AB - E_A - E_B.  Holding the active space fixed between the bound and the
# unbound geometry is what makes the active-space error cancel.  Getting this
# wrong silently changes every number downstream, so binding.py enforces it.
BINDING_ENERGY_DEFINITION = "E_bind(R) = E_dimer(R) - E_dimer(48.000 A)"

# "the highest deviation being observed at 1.400 A and corresponding to
#  2.263 kcal/mol" -- that figure is for the *water* dimer against CCSD(T),
#  and is recorded here only so it is not mistaken for a methane result.
WATER_DIMER_MAX_CCSDT_DEVIATION_KCAL = 2.263

# "a |chi_b| = 20.0e3 is necessary to reach agreement within 0.010 kcal/mol
#  when compared against CASCI" -- methane dimer at 3.638 A.  This is the
# headline number for the 36-qubit experiment and the target of `run.py verify`.
SQD_VS_CASCI_TARGET_KCAL = 0.010
SQD_VS_CASCI_TARGET_DISTANCE = 3.638

# Agreement claimed across the equilibrium region of both dimers.
CHEMICAL_ACCURACY_CLAIM_KCAL = 1.0

# HCI settings, used only for the 54-qubit (16e,24o) comparison we do not run.
HCI_EPSILON_1_INITIAL = 5e-6
HCI_EPSILON_1_FINAL = 1e-6
HCI_PERTURBATIVE_CORRECTION = False   # deliberately variational, for fair comparison

# --------------------------------------------------------------------------
# What the paper does NOT publish
# --------------------------------------------------------------------------
GEOMETRY_GAP = """\
The paper fixes the C-C distance grid but never publishes Cartesian
coordinates, the monomer C-H bond length, or -- decisively -- the relative
orientation of the two methane units.  All it says is:

    "To produce the geometries studied in this work, we start from the
     equilibrium geometries and change the distance between the centers of the
     monomers, with the geometries of the individual monomers fixed."

Methane dimer has several near-degenerate stationary points (D3d, D3h, D2d)
separated by only a few tenths of a kcal/mol -- comparable to the binding
energy itself.  Orientation is therefore not a detail; it is the dominant
uncertainty in this reproduction, and it is larger than the 0.010 kcal/mol
SQD-vs-CASCI agreement being reproduced.

This folder does not paper over that.  `geometry.py` implements the candidate
orientations explicitly, `run.py` requires one to be named, and every result
file records which was used.  Because both the SQD and the CASCI number are
computed at the *same* geometry, the SQD-vs-CASCI agreement -- the actual claim
under test -- is unaffected by the choice.  Absolute energies and the depth of
the binding well are not.
"""

SQD_HYPERPARAMETER_GAP = """\
Table II fixes |chi|, K, |chi_b|, d and the number of recovery steps, but not
the energy or occupancy convergence tolerances, the carryover threshold, the
LUCJ repetition count n_reps, or the random seed.  Defaults in `sqd.py` are
this folder's, are exposed as flags, and are recorded in every result file.
"""
