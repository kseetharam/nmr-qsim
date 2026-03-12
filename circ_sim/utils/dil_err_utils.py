"""
Dilated-error channel utilities.

Implements the quantum channel E(ρ) from the expansion (in t, θ) and computes
the matrix of overlaps Tr(op_i† E(op_j)) for a list of operators {op_i}.

Inputs: OpenFermion QubitOperator H, list of QubitOperators {L_α}, scalars t and θ,
        and list of QubitOperators {op_i}.

Performance: For ~10 spins (Hilbert dimension 2^10=1024), OpenFermion + scipy.sparse
is used; the trace Tr(op_i† E(op_j)) is computed via a dense inner product by default
(use_dense=True), which is stable and fast at this scale. For larger systems, set
use_dense=False to use a sparse trace and consider pre-converting H and L_list to
sparse matrices once to avoid repeated conversion.
"""

import numpy as np
import openfermion as of
from openfermion import hermitian_conjugated
from scipy import sparse


# ---------------------------------------------------------------------------
# Operator norms (for UB_error_dilation)
# ---------------------------------------------------------------------------

def _norm_qubit_op_L1(op):
    """
    L1 norm of a QubitOperator: ||O|| = sum_i |c_i| over Pauli product coefficients.
    """
    if hasattr(op, "terms"):
        return float(sum(abs(c) for c in op.terms.values()))
    raise TypeError("L1 norm expects openfermion.QubitOperator.")


def _norm_qubit_op_L2(op, n_qubits):
    """Frobenius (L2) norm of operator as matrix: ||O||_F = sqrt(Tr(O†O))."""
    if hasattr(op, "terms"):
        sp = of.get_sparse_operator(op, n_qubits=n_qubits)
    else:
        sp = op if sparse.issparse(op) else sparse.csr_matrix(op)
    return np.sqrt((sp.conj().T.multiply(sp)).sum().real)


def _norm_qubit_op_spectral(op, n_qubits):
    """Spectral (operator) norm: largest singular value."""
    if hasattr(op, "terms"):
        sp = of.get_sparse_operator(op, n_qubits=n_qubits)
    else:
        sp = op if sparse.issparse(op) else sparse.csr_matrix(op)
    try:
        from scipy.sparse.linalg import norm as _spnorm
        return _spnorm(sp, ord=2)
    except Exception:
        return np.linalg.norm(sp.toarray(), ord=2)


def _get_norm_fn(norm_spec, n_qubits):
    """
    Return a callable norm(op) or norm(op, n_qubits) for the given norm spec.
    norm_spec: 'L1', 'L2', 'fro', 'spectral', or callable(op, n_qubits=None).
    """
    if callable(norm_spec):
        def norm(op):
            return norm_spec(op, n_qubits) if n_qubits is not None else norm_spec(op)
        return norm
    s = str(norm_spec).strip().lower()
    if s in ("l1", "1"):
        return lambda op: _norm_qubit_op_L1(op)
    if s in ("l2", "fro", "frobenius", "2"):
        if n_qubits is None:
            raise ValueError("n_qubits required for L2/Frobenius norm.")
        return lambda op: _norm_qubit_op_L2(op, n_qubits)
    if s in ("spectral", "op", "operator"):
        if n_qubits is None:
            raise ValueError("n_qubits required for spectral norm.")
        return lambda op: _norm_qubit_op_spectral(op, n_qubits)
    raise ValueError(
        f"Unknown norm: {norm_spec}. Use 'L1', 'L2', 'spectral', or a callable(op, n_qubits=None)."
    )


def UB_error_dilation(H, L_list, t, theta, n_qubits=None, norm="L1"):
    r"""
    Upper bound on the dilation error (norm of the error channel expansion).

    Computes:
      (t³θ²/6)*4*||H|| * ∑_α ( ||L_α† L_α|| + ||L_α|| ||L_α†|| )
      + (t³θ²/6)*4*||H|| * ∑_α ( ||L_α|| ||L_α†|| + ||L_α† L_α|| )
      + (t³θ²/6)*||H|| * ∑_α ( 5||L_α|| ||L_α†|| + 2||L_α† L_α|| )
      + (t⁴θ⁴/24) * ∑_{α≠β} ( 2||L_α L_β|| ||L_α† L_β†|| + 2||L_α L_β|| ||L_β† L_α†|| )
      + (t⁴θ⁴/24) * ∑_{α,β} ( ||L_α|| ||L_α† L_β† L_β|| + ... )  [4 terms]
      + (t⁴θ⁴/24) * ∑_{α,β} ( ||L_β|| ||L_β† L_α† L_α|| + ... )  [4 terms]
      + (t⁴θ⁴/24) * ∑_{α≠β} ( ||L_β† L_α† L_β L_α|| + ||L_α† L_β† L_α L_β|| + 2||L_α† L_β† L_β L_α|| )

    Parameters
    ----------
    H : openfermion.QubitOperator
        Hamiltonian.
    L_list : list of openfermion.QubitOperator
        Jump operators {L_α}.
    t, theta : float
        Scalars.
    n_qubits : int, optional
        Number of qubits. Required for norms that need a matrix (L2, spectral);
        inferred from H if None and norm needs it.
    norm : str or callable, optional
        Norm to use: 'L1' (default, sum of |c_i| over Pauli coefficients),
        'L2'/'fro' (Frobenius), 'spectral', or a callable(op, n_qubits=None)
        returning a non-negative float.

    Returns
    -------
    float
        Upper bound value (non-negative).
    """
    if n_qubits is None and hasattr(H, "terms"):
        n_qubits = of.count_qubits(H)
    nrm = _get_norm_fn(norm, n_qubits)

    # Precompute L† and products we need
    L_dags = [hermitian_conjugated(L) for L in L_list]
    nL = len(L_list)

    def n(op):
        return nrm(op)

    # Precompute norms of single operators and L†L
    norms_L = [n(L) for L in L_list]
    norms_Ld = [n(Ld) for Ld in L_dags]
    LdL = [L_dags[a] * L_list[a] for a in range(nL)]
    norms_LdL = [n(op) for op in LdL]

    norm_H = n(H)

    c3 = (t ** 3) * (theta ** 2) / 6.0
    c4 = (t ** 4) * (theta ** 4) / 24.0

    ub = 0.0

    # Term 1: 4 ||H|| ∑_α ( ||L†L|| + ||L|| ||L†|| )
    for a in range(nL):
        ub += 4.0 * norm_H * (norms_LdL[a] + norms_L[a] * norms_Ld[a])
    ub *= c3

    # Term 2: 4 ||H|| ∑_α ( ||L|| ||L†|| + ||L†L|| )
    t2 = 0.0
    for a in range(nL):
        t2 += norms_L[a] * norms_Ld[a] + norms_LdL[a]
    ub += c3 * 4.0 * norm_H * t2

    # Term 3: ||H|| ∑_α ( 5||L|| ||L†|| + 2||L†L|| )
    t3 = 0.0
    for a in range(nL):
        t3 += 5.0 * norms_L[a] * norms_Ld[a] + 2.0 * norms_LdL[a]
    ub += c3 * norm_H * t3

    # Term 4: ∑_{α≠β} ( 2||L_α L_β|| ||L_α† L_β†|| + 2||L_α L_β|| ||L_β† L_α†|| )
    for a in range(nL):
        for b in range(nL):
            if a == b:
                continue
            LaLb = L_list[a] * L_list[b]
            LdaLdb = L_dags[a] * L_dags[b]
            LdbLda = L_dags[b] * L_dags[a]
            ub += c4 * (2.0 * n(LaLb) * n(LdaLdb) + 2.0 * n(LaLb) * n(LdbLda))

    # Term 5: ∑_{α,β} ( ||L_α|| ||L_α† L_β† L_β|| + ||L_β† L_β L_α|| ||L_α†|| + ||L_α|| ||L_β† L_α† L_β|| + ||L_β† L_α L_β|| ||L_α†|| )
    for a in range(nL):
        for b in range(nL):
            LdaLdbLb = L_dags[a] * L_dags[b] * L_list[b]
            LdbLbLa = L_dags[b] * L_list[b] * L_list[a]
            LdbLdaLb = L_dags[b] * L_dags[a] * L_list[b]
            LdbLaLb = L_dags[b] * L_list[a] * L_list[b]
            ub += c4 * (
                norms_L[a] * n(LdaLdbLb)
                + n(LdbLbLa) * norms_Ld[a]
                + norms_L[a] * n(LdbLdaLb)
                + n(LdbLaLb) * norms_Ld[a]
            )

    # Term 6: ∑_{α,β} ( ||L_β|| ||L_β† L_α† L_α|| + ||L_β†|| ||L_α† L_α L_β|| + ||L_β|| ||L_α† L_β† L_α|| + ||L_β†|| ||L_α† L_β L_α|| )
    for a in range(nL):
        for b in range(nL):
            LdbLdaLa = L_dags[b] * L_dags[a] * L_list[a]
            LdaLaLb = L_dags[a] * L_list[a] * L_list[b]
            LdaLdbLa = L_dags[a] * L_dags[b] * L_list[a]
            LdaLbLa = L_dags[a] * L_list[b] * L_list[a]
            ub += c4 * (
                norms_L[b] * n(LdbLdaLa)
                + norms_Ld[b] * n(LdaLaLb)
                + norms_L[b] * n(LdaLdbLa)
                + norms_Ld[b] * n(LdaLbLa)
            )

    # Term 7: ∑_{α≠β} ( ||L_β† L_α† L_β L_α|| + ||L_α† L_β† L_α L_β|| + 2||L_α† L_β† L_β L_α|| )
    for a in range(nL):
        for b in range(nL):
            if a == b:
                continue
            LdbLdaLbLa = L_dags[b] * L_dags[a] * L_list[b] * L_list[a]
            LdaLdbLaLb = L_dags[a] * L_dags[b] * L_list[a] * L_list[b]
            LdaLdbLbLa = L_dags[a] * L_dags[b] * L_list[b] * L_list[a]
            ub += c4 * (n(LdbLdaLbLa) + n(LdaLdbLaLb) + 2.0 * n(LdaLdbLbLa))

    return float(ub)


# ---------------------------------------------------------------------------
# Conversion and trace helpers
# ---------------------------------------------------------------------------

def _to_sparse(op, n_qubits):
    """QubitOperator or sparse matrix -> sparse (csr)."""
    if sparse.issparse(op):
        return op.tocsr()
    return of.get_sparse_operator(op, n_qubits=n_qubits).tocsr()


def _trace_dag_A_B(A, B, use_dense=True):
    """
    Compute Tr(A† B). use_dense=True (default) uses dense arrays for the trace and
    is correct for any sparsity. use_dense=False uses (A.conj().multiply(B)).sum(),
    which is correct only when A and B have identical sparsity (e.g. diagonal).
    """
    if use_dense:
        Ad = A.toarray() if sparse.issparse(A) else np.asarray(A)
        Bd = B.toarray() if sparse.issparse(B) else np.asarray(B)
        return np.trace(Ad.conj().T @ Bd)
    return (A.conj().multiply(B)).sum()


def _herm(M):
    """Hermitian conjugate of matrix (dense or sparse)."""
    if sparse.issparse(M):
        return M.conj().T
    return np.asarray(M).conj().T


# ---------------------------------------------------------------------------
# Channel E(ρ): E(rho) = sum of terms from the expansion
# All terms applied to rho (sparse matrix). L_list and H are sparse.
# ---------------------------------------------------------------------------

def _apply_channel_E_sparse(rho, H, L_list, t, theta):
    """
    Apply the dilated-error channel E to operator rho (sparse matrix).

    E(ρ) = (it³θ²/6) * term1 + (it³θ²/6) * term2 + (it³θ²/6) * term3 + (t⁴θ⁴/24) * term4

    Term1: ∑_α [H, L_α† L_α ρ - L_α ρ L_α†] + h.c.
    Term2: ∑_α ( -L_α ρ H L_α† + L_α H ρ L_α† + ρ H L_α† L_α - H ρ L_α† L_α - h.c. )
    Term3: ∑_α ( L_α ρ L_α† H - L_α H ρ L_α† - ρ L_α† H L_α + H ρ L_α† L_α - h.c. )
    Term4: quartic in L's with Kronecker-delta coefficients.

    rho, H, and L_list entries are sparse (csr) matrices.
    """
    n = rho.shape[0]
    out = sparse.csr_matrix((n, n), dtype=complex)
    c1 = 1j * (t ** 3) * (theta ** 2) / 6.0
    c2 = (t ** 4) * (theta ** 4) / 24.0

    # Precompute L† for each L
    L_dags = [L.conj().T.tocsr() for L in L_list]
    L_dag_L = [L_dags[a] @ L_list[a] for a in range(len(L_list))]
    L_L_dag = [L_list[a] @ L_dags[a] for a in range(len(L_list))]

    # ----- Term 1: (it³θ²/6) ∑_α [H, L_α† L_α ρ - L_α ρ L_α†] + h.c. -----
    for a in range(len(L_list)):
        L, Ld, LdL, LLd = L_list[a], L_dags[a], L_dag_L[a], L_L_dag[a]
        A = LdL @ rho - L @ rho @ Ld
        T1a = H @ A - A @ H
        out = out + c1 * (T1a + _herm(T1a))

    # ----- Term 2: (it³θ²/6) ∑_α ( -L ρ H L† + L H ρ L† + ρ H L†L - H ρ L†L - h.c. ) -----
    for a in range(len(L_list)):
        L, Ld, LdL = L_list[a], L_dags[a], L_dag_L[a]
        part = -L @ rho @ H @ Ld + L @ H @ rho @ Ld + rho @ H @ LdL - H @ rho @ LdL
        out = out + c1 * (part - _herm(part))

    # ----- Term 3: (it³θ²/6) ∑_α ( L ρ L† H - L H ρ L† - ρ L† H L + H ρ L†L - h.c. ) -----
    for a in range(len(L_list)):
        L, Ld, LdL = L_list[a], L_dags[a], L_dag_L[a]
        part = L @ rho @ Ld @ H - L @ H @ rho @ Ld - rho @ Ld @ H @ L + H @ rho @ LdL
        out = out + c1 * (part - _herm(part))

    # ----- Term 4: (t⁴θ⁴/24) ∑_{α,β,γ,δ} [ ... ] -----
    nL = len(L_list)
    dum = 0.0j
    for a in range(nL):
        for b in range(nL):
            for g in range(nL):
                for d in range(nL):
                    # (1-δ_{α,β})(1-δ_{δ,γ})(δ_{α,δ}δ_{β,γ}+δ_{α,γ}δ_{β,δ}) L_α L_β ρ L_δ† L_γ†
                    coef1 = (1 - (1 if a == b else 0)) * (1 - (1 if d == g else 0))
                    coef1 *= ((1 if a == d and b == g else 0) + (1 if a == g and b == d else 0))
                    if coef1:
                        #out = out + c2 * coef1 * (L_list[a] @ L_list[b] @ rho @ L_dags[d] @ L_dags[g])
                        dum += c2 * coef1 * (L_list[a] @ L_list[b] @ rho @ L_dags[d] @ L_dags[g])

                    # -(δ_{γ,β}δ_{α,δ}+δ_{δ,β}δ_{α,γ}) L_α ρ L_δ† L_γ† L_β
                    coef2 = -((1 if g == b and a == d else 0) + (1 if d == b and a == g else 0))
                    if coef2:
                        #out = out + c2 * coef2 * (L_list[a] @ rho @ L_dags[d] @ L_dags[g] @ L_list[b])
                        dum += c2 * coef2 * (L_list[a] @ rho @ L_dags[d] @ L_dags[g] @ L_list[b])

                    # -(δ_{β,δ}δ_{α,γ}+δ_{β,γ}δ_{δ,α}) L_β ρ L_δ† L_γ† L_α
                    coef3 = -((1 if b == d and a == g else 0) + (1 if b == g and d == a else 0))
                    if coef3:
                        #out = out + c2 * coef3 * (L_list[b] @ rho @ L_dags[d] @ L_dags[g] @ L_list[a])
                        dum += c2 * coef3 * (L_list[b] @ rho @ L_dags[d] @ L_dags[g] @ L_list[a])

                    # (1-δ_{δ,γ})(1-δ_{β,α})(δ_{δ,β}δ_{γ,α}+δ_{δ,α}δ_{γ,β}) ρ L_δ† L_γ† L_β L_α
                    coef4 = (1 - (1 if d == g else 0)) * (1 - (1 if b == a else 0))
                    coef4 *= ((1 if d == b and g == a else 0) + (1 if d == a and g == b else 0))
                    if coef4:
                        #out = out + c2 * coef4 * (rho @ L_dags[d] @ L_dags[g] @ L_list[b] @ L_list[a])
                        dum += c2 * coef4 * (rho @ L_dags[d] @ L_dags[g] @ L_list[b] @ L_list[a])
    out = out + dum + _herm(dum)

    return out.tocsr()


# ---------------------------------------------------------------------------
# Public API: QubitOperator inputs, Tr(op_i† E(op_j)) matrix
# ---------------------------------------------------------------------------

def apply_channel_E(rho, H, L_list, t, theta, n_qubits):
    """
    Apply the dilated-error channel E to an operator rho. 

    Parameters
    ----------
    rho : openfermion.QubitOperator or scipy.sparse matrix
        Input operator (e.g. ρ₀(0) or op_j).
    H : openfermion.QubitOperator or scipy.sparse matrix
        Hamiltonian.
    L_list : list of openfermion.QubitOperator or list of sparse matrices
        Jump operators {L_α}.
    t, theta : float
        Time and angle parameters.
    n_qubits : int
        Number of qubits (required if operators are QubitOperators).

    Returns
    -------
    scipy.sparse.csr_matrix
        E(rho).
    """
    H_sp = _to_sparse(H, n_qubits)
    L_sp = [_to_sparse(L, n_qubits) for L in L_list]
    rho_sp = _to_sparse(rho, n_qubits)
    return _apply_channel_E_sparse(rho_sp, H_sp, L_sp, t, theta)


def trace_op_i_dag_E_op_j(op_i, op_j, H, L_list, t, theta, n_qubits, use_dense=True):
    """
    Compute Tr(op_i† E(op_j)).

    Parameters
    ----------
    op_i, op_j : openfermion.QubitOperator or sparse matrix
    H, L_list, t, theta, n_qubits : as in apply_channel_E
    use_dense : bool
        If True, use dense arrays for the trace (recommended for 2^n ≤ 2048).

    Returns
    -------
    complex
        Tr(op_i† E(op_j)).
    """
    op_i_sp = _to_sparse(op_i, n_qubits)
    E_op_j = apply_channel_E(op_j, H, L_list, t, theta, n_qubits)
    return _trace_dag_A_B(op_i_sp, E_op_j, use_dense=use_dense)


def dil_err_overlap_matrix(ops, H, L_list, t, theta, n_qubits=None, use_dense=True):
    """
    Compute the matrix M[i,j] = Tr(op_i† E(op_j)) for all operators in `ops`.

    Parameters
    ----------
    ops : list of openfermion.QubitOperator or list of sparse matrices
        List of operators {op_i}.
    H : openfermion.QubitOperator or sparse matrix
        Hamiltonian.
    L_list : list of openfermion.QubitOperator or list of sparse matrices
        Jump operators {L_α}.
    t, theta : float
        Scalars in the channel expansion.
    n_qubits : int, optional
        Number of qubits. If None, inferred from H when it is a QubitOperator.
    use_dense : bool
        Use dense arrays for trace (recommended for moderate n_qubits).

    Returns
    -------
    np.ndarray, shape (len(ops), len(ops)), dtype=complex
        M[i,j] = Tr(op_i† E(op_j)).
    """
    if n_qubits is None:
        if hasattr(H, "terms"):
            n_qubits = of.count_qubits(H)
        else:
            n_qubits = int(round(np.log2(H.shape[0])))
    n = len(ops)
    M = np.zeros((n, n), dtype=complex)
    for j in range(n):
        E_op_j = apply_channel_E(ops[j], H, L_list, t, theta, n_qubits)
        for i in range(n):
            op_i_sp = _to_sparse(ops[i], n_qubits)
            M[i, j] = _trace_dag_A_B(op_i_sp, E_op_j, use_dense=use_dense)
    return M


def _dil_err_column_j(args):
    """Worker: compute column j of overlap matrix (all Tr(op_i† E(op_j)) for fixed j)."""
    j, ops, H, L_list, t, theta, n_qubits, use_dense = args
    E_op_j = apply_channel_E(ops[j], H, L_list, t, theta, n_qubits)
    n = len(ops)
    return j, [_trace_dag_A_B(_to_sparse(ops[i], n_qubits), E_op_j, use_dense=use_dense) for i in range(n)]


def dil_err_overlap_matrix_parallel(ops, H, L_list, t, theta, n_qubits=None, use_dense=True, num_workers=None):
    """
    Same as dil_err_overlap_matrix but parallel over columns (each column
    corresponds to E(op_j); inner products with all op_i are computed in one worker).

    Optional: requires concurrent.futures. If num_workers is None, runs sequentially.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if n_qubits is None:
        if hasattr(H, "terms"):
            n_qubits = of.count_qubits(H)
        else:
            n_qubits = int(round(np.log2(H.shape[0])))

    n = len(ops)
    M = np.zeros((n, n), dtype=complex)

    if num_workers is None or num_workers <= 1:
        return dil_err_overlap_matrix(ops, H, L_list, t, theta, n_qubits=n_qubits, use_dense=use_dense)

    task_args = [
        (j, ops, H, L_list, t, theta, n_qubits, use_dense)
        for j in range(n)
    ]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_dil_err_column_j, ta): ta[0] for ta in task_args}
        for fut in as_completed(futures):
            j, col = fut.result()
            M[:, j] = col
    return M
