"""
Exact (no approximations at all) FID and its Fourier-transform spectrum for
truncated Gemcitabine's default-field ZULF Lindbladian, on the time grid
this project's trajectory-based exploration has been using throughout
(delta_t = 1/J_max, J_max = 226.85 Hz = |J_F0-C0|, truncated Gemcitabine's
strongest coupling), out to a total time close to 1 s.

Purely diagnostic, run BEFORE building the delta_t=1/J_max, 100-trajectories-
per-point Trotterized/trajectory-sampled FID (a deferred follow-up): the
question here is only whether this (delta_t, T_total) choice is even
adequate to resolve a reasonable spectrum from the EXACT dynamics, before
spending any effort asking whether the approximate method reproduces it.

Two things this delta_t/T_total choice trades off, both visible directly in
the plot below:
  - Sampling rate / Nyquist limit: delta_t=1/J_max means a sample rate of
    J_max itself, i.e. a Nyquist frequency of ONLY J_max/2 ~ 113.4 Hz
    (marked as vertical dashed lines in the spectrum panel). The F0-C0
    coupling that DEFINES J_max (226.85 Hz) therefore sits at exactly twice
    the Nyquist limit -- any real spectral feature there does not fall
    inside the plotted window and aliases back into it, folded about the
    Nyquist edge, rather than appearing at its true frequency.
  - Frequency resolution: T_total ~ 1 s gives a frequency resolution
    Delta_f = 1/T_total ~ 1 Hz -- enough to separate this molecule's larger
    couplings (order 1-227 Hz) but not its smallest ones (down to 0.02 Hz
    in the raw 10-atom coupling table), which would need a much longer
    T_total to resolve as distinct peaks rather than blur into the same
    bin.

FID(t_n) = Tr(coil * exact_rho(t_n)), t_n = n*delta_t, n=0..N_t-1, via
single_step_trotter_tightness.exact_rho (Liouville-space eigendecomposition,
done once; each additional time point is then a cheap eigenbasis evaluation,
not a new propagation). coil = Sx_coll + i*Sy_coll is genuinely complex
(quadrature-detected, per this project's own ZULF readout convention, not a
real-valued signal) -- so the full complex FFT (not just its magnitude or a
half-spectrum) is the physically meaningful transform, exactly as for
quadrature-detected experimental NMR data.
"""
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')

from trajectory_convergence import sstt  # noqa: E402

J_MAX_HZ = 226.85
DT = 1.0 / J_MAX_HZ
N_T = round(1.0 / DT)  # ~227 points -> total duration ~1.0007 s, "close to 1 s"

if __name__ == '__main__':
    T_total = N_T * DT
    print(f'delta_t = 1/J_max = {DT:.6e} s, N_t = {N_T}, T_total = {T_total:.6f} s')
    print(f'Nyquist frequency = 1/(2*delta_t) = {1.0 / (2 * DT):.4f} Hz')
    print(f'frequency resolution = 1/T_total = {1.0 / T_total:.4f} Hz')

    t_grid = DT * np.arange(N_T)
    fid = np.array([sstt.expect(sstt.coil, sstt.exact_rho(t)) for t in t_grid])
    print(f'FID(0) = {fid[0]:.6f}  (sanity check vs. Tr(coil*rho0) = '
          f'{sstt.expect(sstt.coil, sstt.rho0):.6f})')
    assert np.isclose(fid[0], sstt.expect(sstt.coil, sstt.rho0), atol=1e-9)

    freqs = np.fft.fftshift(np.fft.fftfreq(N_T, d=DT))
    spectrum = np.fft.fftshift(np.fft.fft(fid))

    out_pkl = os.path.join(DATA_DIR, 'plot5_exact_fid_spectrum.pkl')
    with open(out_pkl, 'wb') as fh:
        pickle.dump(dict(DT=DT, N_T=N_T, T_total=T_total, t_grid=t_grid, fid=fid,
                          freqs=freqs, spectrum=spectrum, J_MAX_HZ=J_MAX_HZ), fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved -> {out_pkl}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))

    ax1.plot(t_grid, fid.real, '-', color='C0', label=r'Re$\langle\mathrm{coil}(t)\rangle$', linewidth=1)
    ax1.plot(t_grid, fid.imag, '-', color='C1', label=r'Im$\langle\mathrm{coil}(t)\rangle$', linewidth=1)
    ax1.plot(t_grid, np.abs(fid), '--', color='gray', label=r'$|\langle\mathrm{coil}(t)\rangle|$', linewidth=1)
    ax1.set_xlabel('$t$ (s)')
    ax1.set_ylabel(r'$\langle\mathrm{coil}(t)\rangle$')
    ax1.set_title(rf'Exact FID, $\Delta t=1/J_{{\max}}={DT:.4e}$ s, $N_t={N_T}$, $T={T_total:.4f}$ s')
    ax1.legend(fontsize=9)

    ax2.plot(freqs, np.abs(spectrum), '-', color='C2', linewidth=1)
    nyq = 1.0 / (2 * DT)
    ax2.axvline(nyq, color='k', linestyle=':', linewidth=1, label=f'Nyquist $=\\pm${nyq:.1f} Hz')
    ax2.axvline(-nyq, color='k', linestyle=':', linewidth=1)
    ax2.set_xlabel('frequency (Hz)')
    ax2.set_ylabel(r'$|\mathrm{FFT}[\langle\mathrm{coil}\rangle](f)|$')
    ax2.set_title(rf'Spectrum (complex FFT), resolution $\Delta f=1/T={1.0 / T_total:.3f}$ Hz')
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out_png = os.path.join(DATA_DIR, 'plot5_exact_fid_spectrum.png')
    fig.savefig(out_png, dpi=150)
    print(f'Saved plot -> {out_png}')
