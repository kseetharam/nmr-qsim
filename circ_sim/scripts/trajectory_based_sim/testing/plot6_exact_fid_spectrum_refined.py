"""
Refined version of plot5_exact_fid_spectrum.py's exact FID/spectrum check,
after that first pass showed likely aliasing (dominant spectral cluster near
-17 to -25 Hz matching the F1-C0 coupling's 203.41 Hz alias,
203.41-226.85=-23.44 Hz, folded back because delta_t=1/J_max gave a Nyquist
frequency of only J_max/2) and marginal frequency resolution/decay
completeness at T~1 s.

Two changes, both requested directly:
  - delta_t -> (1/J_max)/2, halving the step doubles the sampling rate and
    therefore the Nyquist frequency to exactly J_max = 226.85 Hz. F0-C0
    (the strongest coupling, numerically equal to J_max) now sits right at
    the Nyquist edge instead of at 2x it; F1-C0 (203.41 Hz) now sits safely
    inside the Nyquist window instead of aliasing to ~-23 Hz.
  - T_total -> 2 s (roughly double), giving ~2x finer frequency resolution
    (~0.5 Hz instead of ~1 Hz) and more of the dissipative decay captured
    before truncation.

Otherwise identical in structure/conventions to plot5_exact_fid_spectrum.py
(same molecule/system, same exact_rho-based construction, same complex-FFT
quadrature convention for the non-Hermitian coil observable) -- see that
module's docstring for the reasoning not repeated here.
"""
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')

from trajectory_convergence import sstt  # noqa: E402

J_MAX_HZ = 226.85
DT = (1.0 / J_MAX_HZ) / 2.0
T_TARGET = 2.0
N_T = round(T_TARGET / DT)

if __name__ == '__main__':
    T_total = N_T * DT
    print(f'delta_t = (1/J_max)/2 = {DT:.6e} s, N_t = {N_T}, T_total = {T_total:.6f} s')
    print(f'Nyquist frequency = 1/(2*delta_t) = {1.0 / (2 * DT):.4f} Hz  (= J_max exactly)')
    print(f'frequency resolution = 1/T_total = {1.0 / T_total:.4f} Hz')

    t_grid = DT * np.arange(N_T)
    fid = np.array([sstt.expect(sstt.coil, sstt.exact_rho(t)) for t in t_grid])
    print(f'FID(0) = {fid[0]:.6f}  (sanity check vs. Tr(coil*rho0) = '
          f'{sstt.expect(sstt.coil, sstt.rho0):.6f})')
    assert np.isclose(fid[0], sstt.expect(sstt.coil, sstt.rho0), atol=1e-9)
    print(f'|FID(T_total)| / |FID(0)| = {abs(fid[-1]) / abs(fid[0]):.4f} (decay completeness)')

    freqs = np.fft.fftshift(np.fft.fftfreq(N_T, d=DT))
    spectrum = np.fft.fftshift(np.fft.fft(fid))

    out_pkl = os.path.join(DATA_DIR, 'plot6_exact_fid_spectrum_refined.pkl')
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
    ax1.set_title(rf'Exact FID, $\Delta t=(1/J_{{\max}})/2={DT:.4e}$ s, $N_t={N_T}$, $T={T_total:.4f}$ s')
    ax1.legend(fontsize=9)

    ax2.plot(freqs, np.abs(spectrum), '-', color='C2', linewidth=1)
    nyq = 1.0 / (2 * DT)
    ax2.axvline(nyq, color='k', linestyle=':', linewidth=1, label=f'Nyquist $=\\pm${nyq:.1f} Hz')
    ax2.axvline(-nyq, color='k', linestyle=':', linewidth=1)
    ax2.axvline(203.41, color='r', linestyle='--', linewidth=1, alpha=0.6, label='F1-C0 = 203.41 Hz')
    ax2.axvline(-203.41, color='r', linestyle='--', linewidth=1, alpha=0.6)
    ax2.set_xlabel('frequency (Hz)')
    ax2.set_ylabel(r'$|\mathrm{FFT}[\langle\mathrm{coil}\rangle](f)|$')
    ax2.set_title(rf'Spectrum (complex FFT), resolution $\Delta f=1/T={1.0 / T_total:.3f}$ Hz')
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out_png = os.path.join(DATA_DIR, 'plot6_exact_fid_spectrum_refined.png')
    fig.savefig(out_png, dpi=150)
    print(f'Saved plot -> {out_png}')
