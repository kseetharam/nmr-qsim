# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**nmr-qsim** simulates NMR (Nuclear Magnetic Resonance) spectroscopy experiments using quantum circuits. The core goal is to encode spin time-evolution (e^{-iHt}) into variational quantum circuits, enabling NMR spectral simulation on quantum hardware (Cirq, IBM, QuEra) or classical simulators.

## Environment Setup

The project uses a conda environment named `c_sym` (defined in `csyn.yml`), installed at `/Users/luismartinezmartinez/pyenvs/csyn`.

```bash
# Create environment from spec
conda env create -f csyn.yml

# Or install from requirements.txt
pip install -r requirements.txt
```

**Python version:** 3.11.7

## Running Scripts

```bash
# Train 10-qubit dynamical decoupling circuit on Heisenberg dynamics
python circ_sim/scripts/train_circuit.py [--n_layers N] [--maxiter M] [--seed S] [--step_size LR] [--data PATH]

# Test HEA circuit on 4-qubit random Heisenberg model
python circ_sim/scripts/test_hea_4spin.py [--n_layers N] [--maxiter M] [--seed S] [--step_size LR]

# Run acetonitrile test suite
python acetonitrile/tests.py
```

There is no pytest configuration. Tests are script-based (`acetonitrile/tests.py`) or notebook-based.

## Architecture

The simulation pipeline flows as:

1. **Hamiltonian construction** (`circ_sim/utils/ham_comp_utils.py`) — parse Gaussian quantum chemistry output files, build spin-spin coupling matrices, construct Heisenberg/Ising models
2. **Circuit generation** — Trotterized evolution, variational Hardware-Efficient Ansatz (`hea_utils.py`), or dynamical decoupling sequences (`exp_circ_utils.py`)
3. **Simulation backend** — direct density matrix via QuTiP (`direct_sim_utils.py`), Linbladian open-system evolution (`linblad_utils.py`), or Cirq circuit simulation (`simulation_utils.py`)
4. **Observable extraction** — magnetization expectation values, spin correlation functions, FFT to frequency domain → NMR spectrum

### Key modules in `circ_sim/utils/`

| File | Role |
|------|------|
| `basis_utils.py` | Spin operator basis (S+, S−, Sz) and irreducible spherical tensor (IST) basis definitions |
| `ham_comp_utils.py` | Hamiltonian construction from Gaussian log files; Trotterized evolution |
| `hea_utils.py` | Hardware-Efficient Ansatz (HEA) variational circuits on hypercube graphs |
| `exp_circ_utils.py` | Dynamical decoupling (DD) circuits with PennyLane/JAX optimization |
| `linblad_utils.py` | Linbladian master equation evolution for open quantum systems |
| `direct_sim_utils.py` | Direct QuTiP/Cirq-based quantum simulation |
| `simulation_utils.py` | BQSKit circuit compilation and synthesis |
| `nmr_spec_utils.py` | NMR spectroscopy post-processing utilities |

### Molecular system modules

- **`circ_sim/`** — main development area; fluorinated molecules (19F/13C), general spin systems
- **`methyl_circuits/`** — aromatic/methyl compounds; ZULF (Zero/Ultra-Low Field) NMR
- **`acetonitrile/`** — acetonitrile dynamics with xarray multi-dimensional data support
- **`autodiff_sim/`** — JAX-based automatic differentiation for open quantum system evolution

### Circuit ansätze

- **HEA (Hardware-Efficient Ansatz):** Parameterized single-qubit rotations + entangling layers on a hypercube graph topology; trained by minimizing overlap cost with target unitary
- **Dynamical Decoupling (DD):** Fixed pulse sequences for noise suppression, optimized via PennyLane/JAX
- **Trotter:** Systematic product-formula approximation to e^{-iHt}

## Key Dependencies

- **cirq** — primary quantum circuit framework
- **openfermion** — quantum operator algebra and Hamiltonian construction
- **qutip** — reference classical simulation of quantum systems
- **JAX** — autodiff and JIT compilation for parameter optimization
- **bqskit** — Berkeley Quantum Synthesis Toolkit for circuit compilation
- **pennylane** — variational quantum circuit optimization
- **quspin** — spin chain dynamics

## Data Files

`circ_sim/data/` contains:
- `.mat` files — MATLAB relaxation matrices and spin system definitions (from Spinach NMR toolbox)
- `.pkl` files — serialized Python objects (spectra, circuit parameters)
- `.npy` files — NumPy arrays (transition rates)
- `.txt` files — experimental NMR spectra for comparison
- Gaussian log file outputs for Hamiltonian extraction
