"""
pauli_algebra.py

General N-qubit Pauli-string dictionary algebra: spin operators are built
symbolically, without ever constructing a 2^N dense matrix, so cost scales
with the sparsity of the underlying coupling graph rather than Hilbert-space
dimension. This is what makes it possible to build jump operators for the
~50-atom fluo_suite subsets, not just the small (<=15-spin) systems dense
matrices can handle.

Convention (matches gemcitabine_jump_operators.py and linblad_utils.py):
    N-char strings from {I,X,Y,Z}^N, ordered by a fixed spin_order;
    M = sum_P c_P (P[0] otimes P[1] otimes ... otimes P[N-1]);
    c_P = Tr(M P) / 2^N
Single-spin operators use the spin-1/2 convention Sx=X/2, Sy=Y/2, Sz=Z/2,
S+ = Sx + i Sy, S- = Sx - i Sy.
"""

_PM = {
    ('I', 'I'): ('I', 1 + 0j), ('I', 'X'): ('X', 1 + 0j), ('I', 'Y'): ('Y', 1 + 0j), ('I', 'Z'): ('Z', 1 + 0j),
    ('X', 'I'): ('X', 1 + 0j), ('X', 'X'): ('I', 1 + 0j), ('X', 'Y'): ('Z', 1j), ('X', 'Z'): ('Y', -1j),
    ('Y', 'I'): ('Y', 1 + 0j), ('Y', 'X'): ('Z', -1j), ('Y', 'Y'): ('I', 1 + 0j), ('Y', 'Z'): ('X', 1j),
    ('Z', 'I'): ('Z', 1 + 0j), ('Z', 'X'): ('Y', 1j), ('Z', 'Y'): ('X', -1j), ('Z', 'Z'): ('I', 1 + 0j),
}


class PauliAlgebra:
    """Pauli-string dictionary algebra over a fixed N-spin register."""

    def __init__(self, n_total):
        self.n = n_total
        self.i_n = 'I' * n_total

    def _single(self, idx, char, coeff):
        return {self.i_n[:idx] + char + self.i_n[idx + 1:]: coeff}

    def ix(self, idx): return self._single(idx, 'X', 0.5 + 0j)
    def iy(self, idx): return self._single(idx, 'Y', 0.5 + 0j)
    def iz(self, idx): return self._single(idx, 'Z', 0.5 + 0j)

    def ip(self, idx):
        return {**self._single(idx, 'X', 0.5 + 0j), **self._single(idx, 'Y', 0.5j)}

    def im(self, idx):
        return {**self._single(idx, 'X', 0.5 + 0j), **self._single(idx, 'Y', -0.5j)}

    @staticmethod
    def dagger(d):
        """Hermitian conjugate: each Pauli string is itself Hermitian, so
        only the coefficients get conjugated."""
        return {k: v.conjugate() for k, v in d.items()}

    @staticmethod
    def add(*dicts, thr=1e-12):
        r = {}
        for d in dicts:
            for k, v in d.items():
                r[k] = r.get(k, 0j) + v
        return {k: v for k, v in r.items() if abs(v) > thr}

    @staticmethod
    def scale(d, c):
        if abs(c) < 1e-30:
            return {}
        return {k: v * c for k, v in d.items()}

    def product(self, d1, d2, thr=1e-12):
        r = {}
        for s1, c1 in d1.items():
            for s2, c2 in d2.items():
                chars, ph = [], 1 + 0j
                for ch1, ch2 in zip(s1, s2):
                    ch, p = _PM[(ch1, ch2)]
                    chars.append(ch)
                    ph *= p
                key = ''.join(chars)
                r[key] = r.get(key, 0j) + c1 * c2 * ph
        return {k: v for k, v in r.items() if abs(v) > thr}

    def commutator(self, d1, d2):
        return self.add(self.product(d1, d2), self.scale(self.product(d2, d1), -1))

    def trace_product(self, d1, d2):
        """Normalized trace <d1.d2> = Tr[d1.d2]/2^N (coefficient of the
        all-identity string in the product)."""
        return self.product(d1, d2).get(self.i_n, 0j)
