"""
PennyLane implementation of the 10-qubit DD (dynamical decoupling) experiment circuit
from Notebooks/10_qubits_dd_notebook.ipynb.

One layer = 3 offset_gate blocks (offset1_left, offset1_right, offset3_left).
Each offset_gate uses 22 parameters: 10 Lθ, 10 Lϕ, 2 global (Gθ, Gϕ).
Total parameters = n_layers * 66.

Optimization: JAX-enabled QNode and cost 1 - (1/N) sum_n |⟨n|U(θ)|n⟩| over user-provided
initial states |n⟩, for minimization (e.g. with JAX optimizers).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pennylane as qml

try:
    import cirq
except ImportError:
    cirq = None  # optional; required only for pennylane_dd_circuit_to_cirq

# Fixed for this experiment
N_QUBITS = 10
PARAMS_PER_OFFSET = 22   # 10 Lθ + 10 Lϕ + 2 global
OFFSETS_PER_LAYER = 3
PARAMS_PER_LAYER = OFFSETS_PER_LAYER * PARAMS_PER_OFFSET  # 66


def n_params_for_layers(n_layers: int) -> int:
    """Total number of parameters for a circuit with n_layers."""
    return n_layers * PARAMS_PER_LAYER


# ----- CZ-pattern "offset" gates (no parameters) -----

def offset1_left():
    """CZ on pairs (0,1), (2,3), (4,5), (6,7), (8,9)."""
    for i in range(0, N_QUBITS - 1, 2):
        qml.CZ(wires=[i, i + 1])


def offset1_right():
    """CZ on pairs (1,2), (3,4), (5,6), (7,8)."""
    for i in range(1, N_QUBITS - 1, 2):
        qml.CZ(wires=[i, i + 1])


def offset3_left():
    """CZ on pairs (0,3), (2,5), (4,7), (6,9)."""
    for i in range(0, N_QUBITS - 2, 2):
        qml.CZ(wires=[i, i + 3])


def pi_echo():
    """Global π pulse about X (RX(π) on all qubits)."""
    for w in range(N_QUBITS):
        qml.RX(np.pi, wires=w)


def _pi_pulse_x():
    """Single π pulse about X: Rz(0)-Rx(π)-Rz(0) on all qubits."""
    for w in range(N_QUBITS):
        qml.RX(np.pi, wires=w)


def _pi_pulse_y():
    """Single π pulse about Y: Rz(-π/2)-Rx(π)-Rz(π/2) on all qubits."""
    for w in range(N_QUBITS):
        qml.RZ(-np.pi / 2, wires=w)
    for w in range(N_QUBITS):
        qml.RX(np.pi, wires=w)
    for w in range(N_QUBITS):
        qml.RZ(np.pi / 2, wires=w)


def xy8_gate(gate_fn, num_pi_pulses: int = 4):
    """
    XY8 block: num_pi_pulses pulses (X, Y, X, Y, ...), with gate_fn inserted after the 2nd pulse (i==1).
    Matches notebook: i % 8 in [0,2,5,7] -> X; else -> Y; gate inserted when i == 1.
    """
    for i in range(num_pi_pulses):
        if i % 8 in (0, 2, 5, 7):
            _pi_pulse_x()
        else:
            _pi_pulse_y()
        if i == 1:
            gate_fn()


def local_rot_layer_x(params_slice: np.ndarray):
    """Apply Rz(params_slice[q]) on qubit q, then Rz(π), pi_echo, Rz(π)."""
    for q in range(N_QUBITS):
        qml.RZ(params_slice[q], wires=q)
    for w in range(N_QUBITS):
        qml.RZ(np.pi, wires=w)
    pi_echo()
    for w in range(N_QUBITS):
        qml.RZ(np.pi, wires=w)


def local_rot_layer_z(params_slice: np.ndarray):
    """Apply Rz(params_slice[q]) on qubit q, then Rz(π), pi_echo, Rz(π)."""
    for q in range(N_QUBITS):
        qml.RZ(params_slice[q], wires=q)
    for w in range(N_QUBITS):
        qml.RZ(np.pi, wires=w)
    pi_echo()
    for w in range(N_QUBITS):
        qml.RZ(np.pi, wires=w)


def offset_gate(
    params_slice: np.ndarray,
    gate_fn,
):
    """
    One offset_gate block using 22 parameters from params_slice.
    Layout: params_slice[0:10]=Lθ, params_slice[10:20]=Lϕ, params_slice[20]=Gθ, params_slice[21]=Gϕ.
    """
    # Rx(π/2) - local_rot_x - Rx(π/2)
    for w in range(N_QUBITS):
        qml.RX(np.pi / 2, wires=w)
    local_rot_layer_x(params_slice[0:10])
    for w in range(N_QUBITS):
        qml.RX(np.pi / 2, wires=w)
    # Rx(π)
    for w in range(N_QUBITS):
        qml.RX(np.pi, wires=w)
    # local_rot_z
    local_rot_layer_z(params_slice[10:20])
    # XY8
    xy8_gate(gate_fn, num_pi_pulses=4)
    # Global
    g_theta, g_phi = params_slice[20], params_slice[21]
    for w in range(N_QUBITS):
        qml.RX(g_theta, wires=w)
    for w in range(N_QUBITS):
        qml.RZ(g_phi, wires=w)
    # XY8 again
    xy8_gate(gate_fn, num_pi_pulses=4)


def apply_full_circuit(params, n_layers: int):
    """
    Apply the full DD circuit: n_layers layers, each with 3 offset_gate blocks
    (offset1_left, offset1_right, offset3_left), using the 1D parameter array.

    params: shape (n_layers * 66,) — NumPy or JAX array (avoid np.asarray here so
    JAX tracing works when this is called from a JAX QNode).
    """
    # Use .ravel() only — do not use np.asarray(params) when params may be a JAX array/tracer
    if hasattr(params, "ravel"):
        params_flat = params.ravel()
    else:
        params_flat = np.asarray(params).ravel()
    expected = n_params_for_layers(n_layers)
    if len(params_flat) != expected:
        raise ValueError(
            f"params length must be {expected} for n_layers={n_layers}, got {len(params_flat)}"
        )
    gates = [offset1_left, offset1_right, offset3_left]
    for layer in range(n_layers):
        for offset in range(OFFSETS_PER_LAYER):
            base = layer * PARAMS_PER_LAYER + offset * PARAMS_PER_OFFSET
            offset_gate(params_flat[base : base + PARAMS_PER_OFFSET], gates[offset])


def build_dd_circuit(
    n_layers: int,
    params: np.ndarray | None = None,
    *,
    seed: int | None = None,
):
    """
    Build the 10-qubit DD experiment circuit as a PennyLane QNode and return
    parameters to use.

    Parameters
    ----------
    n_layers : int
        Number of circuit layers (each layer = 3 offset_gate blocks).
    params : array-like, optional
        1D array of length n_layers * 66. If provided, it is validated and used;
        if None, random angles in [-π, π] are generated.
    seed : int, optional
        Random seed used only when params is None.

    Returns
    -------
    qnode : pennylane.QNode
        A QNode that takes a single 1D parameter array and returns the state vector.
        Call with qnode(params) to get the final state.
    params : np.ndarray
        The parameter array to use (either the provided one or the randomly generated one).

    Raises
    ------
    ValueError
        If params is provided and its length is not n_layers * 66.
    """
    expected = n_params_for_layers(n_layers)
    if params is not None:
        params = np.asarray(params, dtype=float)
        if params.ndim != 1 or len(params) != expected:
            raise ValueError(
                f"params must be 1D of length {expected} (n_layers * 66), got shape {params.shape}"
            )
    else:
        rng = np.random.default_rng(seed)
        params = rng.uniform(-np.pi, np.pi, size=expected)

    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def circuit(par):
        apply_full_circuit(par, n_layers)
        return qml.state()

    return circuit, params


# ---------------------------------------------------------------------------
# JAX-based overlap cost and optimization
# ---------------------------------------------------------------------------

def make_overlap_qnode(
    n_layers: int,
    interface: str = "jax",
):
    """
    Build a QNode that takes an initial state |n⟩ and circuit parameters, prepares
    |n⟩, applies the DD circuit U(θ), and returns the state vector U(θ)|n⟩.
    Used to compute ⟨n|U(θ)|n⟩ for the overlap cost.

    Parameters
    ----------
    n_layers : int
        Number of circuit layers.
    interface : str
        PennyLane interface; use "jax" for JAX differentiation.

    Returns
    -------
    qnode : callable
        qnode(initial_state, params) -> state_vector (shape (2**N_QUBITS,), complex).
        initial_state and params must be arrays in the chosen interface (e.g. jax.numpy).
    """
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface=interface)
    def circuit(initial_state, params):
        qml.StatePrep(initial_state, wires=range(N_QUBITS))
        apply_full_circuit(params, n_layers)
        return qml.state()

    return circuit


def make_overlap_cost(
    n_layers: int,
    states_bra: np.ndarray,
    states_ket: np.ndarray | None = None,
    *,
    interface: str = "jax",
) -> Callable:
    """
    Build the scalar cost function for minimization:

        cost(θ) = 1 - (1/N) Σ_n |⟨a_n|U(θ)|b_n⟩|

    where |a_n⟩ are the bra states and |b_n⟩ the ket states. If states_ket is None,
    states_bra is used for both (cost = 1 - (1/N) Σ_n |⟨n|U(θ)|n⟩|).

    Parameters
    ----------
    n_layers : int
        Number of circuit layers.
    states_bra : array-like, shape (N, 2**N_QUBITS)
        N "bra" state vectors; each row is one ⟨a_n| (used as conjugated in vdot).
    states_ket : array-like, shape (N, 2**N_QUBITS), optional
        N "ket" state vectors; each row is one |b_n⟩. Circuit prepares |b_n⟩ then
        applies U(θ). If None, states_bra is used for both.
    interface : str
        Must be "jax" for JAX-based optimization.

    Returns
    -------
    cost_fn : callable
        cost_fn(params) -> scalar. When interface is "jax", params should be
        a JAX array so that jax.grad(cost_fn)(params) works.
    """
    try:
        import jax.numpy as jnp
    except ImportError as e:
        raise ImportError("JAX is required for make_overlap_cost; install jax and jaxlib.") from e

    states_bra = np.asarray(states_bra)
    if states_bra.ndim != 2 or states_bra.shape[1] != 2**N_QUBITS:
        raise ValueError(
            f"states_bra must have shape (N, {2**N_QUBITS}), got {states_bra.shape}"
        )
    n_states = states_bra.shape[0]
    if states_ket is None:
        states_ket = states_bra
    else:
        states_ket = np.asarray(states_ket)
        if states_ket.shape != states_bra.shape:
            raise ValueError(
                f"states_ket must have the same shape as states_bra ({states_bra.shape}), got {states_ket.shape}"
            )
    states_bra_jax = jnp.array(states_bra)
    states_ket_jax = jnp.array(states_ket)

    qnode = make_overlap_qnode(n_layers, interface=interface)

    def cost_fn(params):
        overlaps = jnp.zeros((n_states,), dtype=jnp.float64)
        for i in range(n_states):
            # Prepare |b_n⟩, apply U(θ) -> state_after = U(θ)|b_n⟩
            state_after = qnode(states_ket_jax[i], params)
            # ⟨a_n|U(θ)|b_n⟩; vdot conjugates first arg so ⟨a_n|ψ⟩
            overlap_complex = jnp.vdot(states_bra_jax[i], state_after)
            overlaps = overlaps.at[i].set(overlap_complex)
        return 1.0 - jnp.abs(jnp.mean(overlaps))
    return cost_fn


def minimize_overlap_cost(
    n_layers: int,
    states_bra: np.ndarray,
    states_ket: np.ndarray | None = None,
    params: np.ndarray | None = None,
    *,
    seed: int | None = None,
    maxiter: int = 200,
    step_size: float = 0.01,
    tol: float = 1e-6,
    verbose: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Minimize cost(θ) = 1 - (1/N) Σ_n |⟨a_n|U(θ)|b_n⟩| using JAX and a simple
    gradient descent optimizer (optax.adam if available, else manual SGD).

    Parameters
    ----------
    n_layers : int
        Number of circuit layers.
    states_bra : array-like, shape (N, 2**N_QUBITS)
        N "bra" state vectors ⟨a_n|.
    states_ket : array-like, shape (N, 2**N_QUBITS), optional
        N "ket" state vectors |b_n⟩. If None, states_bra is used for both.
    params : array-like, optional
        Initial parameter vector; length n_layers * 66. If None, random in [-π, π].
    seed : int, optional
        Random seed for initial params when params is None.
    maxiter : int
        Maximum number of optimization steps.
    step_size : float
        Learning rate (used as-is for SGD; for Adam this is the initial step size).
    tol : float
        Stop if cost change is below tol (for simple SGD).
    verbose : bool
        If True, print cost every 50 steps.

    Returns
    -------
    params_opt : np.ndarray
        Optimized parameter vector.
    cost_final : float
        Final cost value.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as e:
        raise ImportError("JAX is required for minimize_overlap_cost; install jax and jaxlib.") from e

    try:
        import optax
    except ImportError:
        optax = None

    expected = n_params_for_layers(n_layers)
    if params is not None:
        params = np.asarray(params).ravel()
        if len(params) != expected:
            raise ValueError(f"params length must be {expected}, got {len(params)}")
    else:
        rng = np.random.default_rng(seed)
        params = rng.uniform(-np.pi, np.pi, size=expected)

    cost_fn = make_overlap_cost(
        n_layers, states_bra, states_ket=states_ket, interface="jax"
    )
    grad_fn = jax.grad(cost_fn)
    params_jax = jnp.array(params)

    if optax is not None:
        optimizer = optax.adam(step_size)
        opt_state = optimizer.init(params_jax)
        last_cost = float(jnp.inf)

        for step in range(maxiter):
            cost_val = cost_fn(params_jax)
            grads = grad_fn(params_jax)
            updates, opt_state = optimizer.update(grads, opt_state)
            params_jax = optax.apply_updates(params_jax, updates)
            if verbose and (step % 50 == 0 or step == maxiter - 1):
                print(f"  step {step}: cost = {float(cost_val):.6f}")
            if step > 0 and abs(float(cost_val) - last_cost) < tol:
                if verbose:
                    print(f"  converged at step {step} (cost change < {tol})")
                break
            last_cost = float(cost_val)

        params_opt = np.asarray(params_jax)
        cost_final = float(cost_fn(params_jax))
    else:
        # Fallback: vanilla gradient descent
        last_cost = float("inf")
        for step in range(maxiter):
            cost_val = cost_fn(params_jax)
            grads = grad_fn(params_jax)
            params_jax = params_jax - step_size * grads
            if verbose and (step % 50 == 0 or step == maxiter - 1):
                print(f"  step {step}: cost = {float(cost_val):.6f}")
            if step > 0 and abs(float(cost_val) - last_cost) < tol:
                if verbose:
                    print(f"  converged at step {step} (cost change < {tol})")
                break
            last_cost = float(cost_val)
        params_opt = np.asarray(params_jax)
        cost_final = float(cost_fn(params_jax))

    return params_opt, cost_final


# ---------------------------------------------------------------------------
# PennyLane DD circuit -> Cirq
# ---------------------------------------------------------------------------

def _cirq_offset1_left(qubits):
    """CZ on pairs (0,1), (2,3), (4,5), (6,7), (8,9)."""
    return [cirq.CZ(qubits[i], qubits[i + 1]) for i in range(0, N_QUBITS - 1, 2)]


def _cirq_offset1_right(qubits):
    """CZ on pairs (1,2), (3,4), (5,6), (7,8)."""
    return [cirq.CZ(qubits[i], qubits[i + 1]) for i in range(1, N_QUBITS - 1, 2)]


def _cirq_offset3_left(qubits):
    """CZ on pairs (0,3), (2,5), (4,7), (6,9)."""
    return [cirq.CZ(qubits[i], qubits[i + 3]) for i in range(0, N_QUBITS - 2, 2)]


def _cirq_pi_echo(qubits):
    """Global π pulse about X (RX(π) on all qubits)."""
    return [cirq.rx(np.pi).on(qubits[w]) for w in range(N_QUBITS)]


def _cirq_pi_pulse_x(qubits):
    return [cirq.rx(np.pi).on(qubits[w]) for w in range(N_QUBITS)]


def _cirq_pi_pulse_y(qubits):
    ops = []
    for w in range(N_QUBITS):
        ops.append(cirq.rz(-np.pi / 2).on(qubits[w]))
    for w in range(N_QUBITS):
        ops.append(cirq.rx(np.pi).on(qubits[w]))
    for w in range(N_QUBITS):
        ops.append(cirq.rz(np.pi / 2).on(qubits[w]))
    return ops


def _cirq_xy8_block(qubits, insert_czs_fn, num_pi_pulses: int = 4):
    """XY8 block: X, Y, X, Y with insert_czs_fn() after the 2nd pulse."""
    ops = []
    for i in range(num_pi_pulses):
        if i % 8 in (0, 2, 5, 7):
            ops.extend(_cirq_pi_pulse_x(qubits))
        else:
            ops.extend(_cirq_pi_pulse_y(qubits))
        if i == 1:
            ops.extend(insert_czs_fn(qubits))
    return ops


def _cirq_local_rot_layer_x(qubits, params_slice: np.ndarray):
    for q in range(N_QUBITS):
        yield cirq.rz(params_slice[q]).on(qubits[q])
    for w in range(N_QUBITS):
        yield cirq.rz(np.pi).on(qubits[w])
    yield from _cirq_pi_echo(qubits)
    for w in range(N_QUBITS):
        yield cirq.rz(np.pi).on(qubits[w])


def _cirq_local_rot_layer_z(qubits, params_slice: np.ndarray):
    for q in range(N_QUBITS):
        yield cirq.rz(params_slice[q]).on(qubits[q])
    for w in range(N_QUBITS):
        yield cirq.rz(np.pi).on(qubits[w])
    yield from _cirq_pi_echo(qubits)
    for w in range(N_QUBITS):
        yield cirq.rz(np.pi).on(qubits[w])


def _cirq_offset_gate(
    qubits,
    params_slice: np.ndarray,
    insert_czs_fn,
):
    """One offset_gate block in Cirq (same structure as PennyLane offset_gate)."""
    ops = []
    for w in range(N_QUBITS):
        ops.append(cirq.rx(np.pi / 2).on(qubits[w]))
    ops.extend(list(_cirq_local_rot_layer_x(qubits, params_slice[0:10])))
    for w in range(N_QUBITS):
        ops.append(cirq.rx(np.pi / 2).on(qubits[w]))
    for w in range(N_QUBITS):
        ops.append(cirq.rx(np.pi).on(qubits[w]))
    ops.extend(list(_cirq_local_rot_layer_z(qubits, params_slice[10:20])))
    ops.extend(_cirq_xy8_block(qubits, insert_czs_fn, num_pi_pulses=4))
    g_theta, g_phi = params_slice[20], params_slice[21]
    for w in range(N_QUBITS):
        ops.append(cirq.rx(g_theta).on(qubits[w]))
    for w in range(N_QUBITS):
        ops.append(cirq.rz(g_phi).on(qubits[w]))
    ops.extend(_cirq_xy8_block(qubits, insert_czs_fn, num_pi_pulses=4))
    return ops


def pennylane_dd_circuit_to_cirq(
    params: np.ndarray,
    n_layers: int,
    qubits: list | None = None,
):
    """
    Build the Cirq circuit equivalent to the PennyLane DD circuit U(θ) with
    the given params and n_layers (no state prep; circuit only).

    Parameters
    ----------
    params : array-like, shape (n_layers * 66,)
        Parameter vector from minimize_overlap_cost or build_dd_circuit.
    n_layers : int
        Number of layers.
    qubits : list of cirq.Qid, optional
        Qubit register. If None, use cirq.LineQubit.range(N_QUBITS).

    Returns
    -------
    cirq.Circuit
        Circuit implementing the same unitary as apply_full_circuit(params, n_layers).
    """
    if cirq is None:
        raise ImportError("cirq is required for pennylane_dd_circuit_to_cirq; install cirq.")

    params_flat = np.asarray(params, dtype=float).ravel()
    if len(params_flat) != n_params_for_layers(n_layers):
        raise ValueError(
            f"params length must be {n_params_for_layers(n_layers)} for n_layers={n_layers}, got {len(params_flat)}"
        )
    if qubits is None:
        qubits = cirq.LineQubit.range(N_QUBITS)
    gates = [_cirq_offset1_left, _cirq_offset1_right, _cirq_offset3_left]
    circuit = cirq.Circuit()
    for layer in range(n_layers):
        for offset in range(OFFSETS_PER_LAYER):
            base = layer * PARAMS_PER_LAYER + offset * PARAMS_PER_OFFSET
            circuit.append(
                _cirq_offset_gate(
                    qubits,
                    params_flat[base : base + PARAMS_PER_OFFSET],
                    gates[offset],
                )
            )
    return circuit
