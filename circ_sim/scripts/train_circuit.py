#!/usr/bin/env python3
"""
Circuit training script based on circ_training_dynamics.ipynb.

Minimizes overlap cost 1 - (1/N) Σ_n |⟨n|U(θ)|n⟩| where |n⟩ are computational
basis states evolved under e^{-iHt}, and U(θ) is the DD circuit. Saves optimized
params, final cost, and the circuit in Cirq format to output/<run>.pkl.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import scipy.io as spio
from scipy.sparse.linalg import expm

# Resolve path so we can import from circ_sim.utils from repo root or from circ_sim/scripts
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CIRC_SIM_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _CIRC_SIM_ROOT not in sys.path:
    sys.path.insert(0, _CIRC_SIM_ROOT)

from utils.exp_circ_utils import (
    N_QUBITS,
    minimize_overlap_cost,
    n_params_for_layers,
    pennylane_dd_circuit_to_cirq,
)


def generate_heisenberg_hamiltonian(Jmat):
    """Build Heisenberg H from coupling matrix Jmat (OpenFermion QubitOperator)."""
    from openfermion import QubitOperator

    N = Jmat.shape[0]
    H = QubitOperator()
    for i in range(N):
        for j in range(i + 1, N):
            if Jmat[i, j] != 0:
                H += Jmat[i, j] * (
                    QubitOperator(f"Z{i} Z{j}")
                    + QubitOperator(f"X{i} X{j}")
                    + QubitOperator(f"Y{i} Y{j}")
                )
    return H


def load_hamiltonian_and_evolved_states(data_path: str, time: float):
    """Load Jmat from .mat file, build H, return evolved states and comp basis."""
    import openfermion as of

    load_mat = spio.loadmat(data_path, squeeze_me=True)
    Jmat = load_mat["inter"]["coupling"].item().item()[0].astype("float")
    Jmat_norm = Jmat / np.max(np.abs(Jmat))
    H = generate_heisenberg_hamiltonian(Jmat_norm)
    sp_H = of.get_sparse_operator(H)
    tar_unitary = expm(-1j * time * sp_H)
    comp_basis_states = np.eye(2**N_QUBITS, dtype=complex)
    evolved_states = [tar_unitary @ comp_basis_states[i] for i in range(2**N_QUBITS)]
    return (
        np.asarray(evolved_states),
        comp_basis_states,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train DD circuit to match e^{-iHt} on computational basis."
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=1,
        help="Number of circuit layers (default: 1)",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=100,
        help="Maximum optimization steps (default: 100)",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=1.0,
        help="Evolution time for target e^{-iHt} (default: 1.0)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to .mat file with inter.coupling. Default: circ_sim/data/big_fluo_mols/gemcitabine_inter.mat",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initial parameters (default: 42)",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.01,
        help="Optimizer step size (default: 0.01)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output",
        help="Output directory for pickle file (default: output)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print optimization progress (default: True)",
    )
    parser.add_argument(
        "--no_verbose",
        action="store_false",
        dest="verbose",
        help="Disable verbose output",
    )
    args = parser.parse_args()

    if args.data is None:
        args.data = os.path.join(
            _CIRC_SIM_ROOT, "data", "big_fluo_mols", "gemcitabine_inter.mat"
        )
    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")

    evolved_states, comp_basis_states = load_hamiltonian_and_evolved_states(
        args.data, args.time
    )

    print(
        f"Training: n_layers={args.n_layers}, maxiter={args.maxiter}, time={args.time}"
    )
    params_opt, cost_final = minimize_overlap_cost(
        args.n_layers,
        evolved_states,
        states_ket=comp_basis_states,
        params=None,
        seed=args.seed,
        maxiter=args.maxiter,
        step_size=args.step_size,
        verbose=args.verbose,
    )

    circuit_cirq = pennylane_dd_circuit_to_cirq(params_opt, args.n_layers)

    os.makedirs(args.out_dir, exist_ok=True)
    out_name = (
        f"train_circuit_nl{args.n_layers}_maxiter{args.maxiter}_t{args.time}.pkl"
    )
    out_path = os.path.join(args.out_dir, out_name)

    results = {
        "params_opt": params_opt,
        "cost_final": cost_final,
        "circuit_cirq": circuit_cirq,
        "n_layers": args.n_layers,
        "maxiter": args.maxiter,
        "time": args.time,
        "seed": args.seed,
    }

    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved results to {out_path}")
    print(f"  cost_final = {cost_final:.6f}, params_opt.shape = {params_opt.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
