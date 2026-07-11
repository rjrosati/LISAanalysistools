"""Minimal check: does the instrument-noise model used by
LISAanalysistools/global_fit_input/noise_sgwb_global_fit_settings.py
(CompositeSensitivityBackend, TDI-2, PSD_INJECTION = [Soms_d=15e-12, Sa_a=3e-15])
fit the mojito CD1L noise-only file NOISE_731d_2.5s_...h5?

Two comparisons, both on the analysis band 0.3-8 mHz:
  1. FD  : Welch cross-spectral matrix of TDI X2/Y2/Z2 vs the model covariance,
           plus a 2-parameter (Soms_d, Sa_a) refit via the backend's linear basis.
  2. WDM : full WDM transform of the data vs the model WDM covariance
           (exact "fold" evaluation), whitened per-pixel chi2.

TDI streams in the file are in Hz; divide by the laser_frequency attribute to
get the fractional-frequency convention the lisatools model uses.
"""

import numpy as np
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lisatools.domains import FDSettings, TDSettings, TDSignal, WDMSettings
from lisatools.sensitivity import CompositeSensitivityBackend

NOISE_FILE = "NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5"
PSD_INJECTION = np.array([15e-12, 3e-15])  # (Soms_d, Sa_a), sqrt units
FMIN, FMAX = 3e-4, 8e-3  # analysis band of the noise_sgwb settings script
CHANNELS = ["X2", "Y2", "Z2"]

# ------------------------------------------------------------------ load data
with h5py.File(NOISE_FILE, "r") as f:
    laser_freq = f.attrs["laser_frequency"]
    dt = float(f["tdis/sampling"].attrs["dt"])
    xyz = np.array([f[f"tdis/{ch}"][:] for ch in CHANNELS]) / laser_freq
print(f"loaded {xyz.shape[1]} samples x 3 channels, dt={dt}s "
      f"({xyz.shape[1] * dt / 86400:.1f} days)")

# --------------------------------------------------- FD: Welch CSD vs model
nper = 2**21                      # 60.7-day segments, df = 1.9e-7 Hz
step = nper // 2                  # 50% overlap
win = np.hanning(nper)
nseg = (xyz.shape[1] - nper) // step + 1
acc = 0.0
for i in range(nseg):
    seg = xyz[:, i * step : i * step + nper]
    F = np.fft.rfft((seg - seg.mean(axis=1, keepdims=True)) * win, axis=1)
    acc = acc + F[:, None] * F[None, :].conj()
csd = acc * (2.0 * dt / (win**2).sum() / nseg)  # one-sided density, (3,3,Nf)
df = 1.0 / (nper * dt)
print(f"Welch: {nseg} segments of {nper} samples")

fd_set = FDSettings(N=nper // 2 + 1, df=df, min_freq=FMIN, max_freq=FMAX,
                    force_backend="cpu")
backend_fd = CompositeSensitivityBackend(fd_set, tdi_generation=2)
S_fd = np.asarray(backend_fd("check", PSD_INJECTION).sens_mat)  # (3,3,Nf_act)
f_act = np.arange(fd_set.ind_min, fd_set.ind_max + 1) * df
D_fd = csd[:, :, fd_set.active_slice]
assert S_fd.shape == D_fd.shape, (S_fd.shape, D_fd.shape)

# per-channel ratios in sub-bands + full-matrix whitened trace
bands = [(3e-4, 1e-3), (1e-3, 3e-3), (3e-3, 8e-3)]
print("\n--- FD: data/model PSD ratio (median per band) ---")
for lo, hi in bands:
    m = (f_act >= lo) & (f_act < hi)
    r = [np.median(D_fd[i, i, m].real / S_fd[i, i, m]) for i in range(3)]
    print(f"  {lo * 1e3:4.1f}-{hi * 1e3:3.1f} mHz : "
          + "  ".join(f"{ch}={ri:.3f}" for ch, ri in zip(CHANNELS, r)))
Sinv_D = np.einsum("ijf,jkf->ikf", np.moveaxis(np.linalg.inv(
    np.moveaxis(S_fd, -1, 0)), 0, -1), D_fd)
tr_ratio = np.einsum("iif->f", Sinv_D).real / 3.0
print(f"  full-matrix whitened trace/3 over band: mean={tr_ratio.mean():.4f} "
      f"(expect 1 within ~{1 / np.sqrt(nseg * len(f_act)):.0e})")

# 2-parameter refit: covariance is linear in (Soms_d^2, Sa_a^2); weighted LSQ
# on the auto-spectra using the backend's cached basis matrices.
B_oms, B_acc = (np.asarray(b) for b in backend_fd._instrument_basis())
diag = lambda M: np.array([M[i, i].real for i in range(3)]).ravel()
w = 1.0 / diag(S_fd)  # ~constant relative errors
A = np.column_stack([diag(B_oms) * w, diag(B_acc) * w])
coef, *_ = np.linalg.lstsq(A, diag(D_fd) * w, rcond=None)
soms_fit, sa_fit = np.sqrt(coef)
print(f"\n--- FD: 2-parameter refit ---\n"
      f"  Soms_d = {soms_fit:.4e}  (model {PSD_INJECTION[0]:.1e}, "
      f"{100 * (soms_fit / PSD_INJECTION[0] - 1):+.1f}%)\n"
      f"  Sa_a   = {sa_fit:.4e}  (model {PSD_INJECTION[1]:.1e}, "
      f"{100 * (sa_fit / PSD_INJECTION[1] - 1):+.1f}%)")

# --------------------------------------------- WDM: transform data vs model
# The record's ends don't match (periodic-extension jump ~2.6e-18 in Y2),
# so the full-record FFT inside wdmtransform leaks ~f^-2 power that swamps
# the in-band noise. Detrend + 1-day cosine taper each end, then exclude the
# tapered edge columns (plus a wavelet-support margin) from the statistics.
Nf, Nt = 1536, 16384  # layer_df = 0.13 mHz, N = 25,165,824 <= data length
N = Nf * Nt
wdm_set = WDMSettings(Nf=Nf, Nt=Nt, dt=dt, min_freq=FMIN, max_freq=FMAX,
                      force_backend="cpu")
data = xyz[:, :N].copy()
t_lin = np.arange(N, dtype=float)
data -= (np.polynomial.polynomial.polyval(
    t_lin, np.polynomial.polynomial.polyfit(t_lin[::100], data[:, ::100].T, 1)))
n_tap = int(86400.0 / dt)
taper = np.ones(N)
ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_tap) / n_tap))
taper[:n_tap] = ramp
taper[-n_tap:] = ramp[::-1]
td = TDSignal(data * taper, TDSettings(N=N, dt=dt, force_backend="cpu"))
w_mn = np.asarray(td.wdmtransform(settings=wdm_set).arr).real  # (3,mf,nt)
edge_cols = int(np.ceil(n_tap * dt / wdm_set.layer_dt)) + 4  # taper + support
interior = slice(edge_cols, w_mn.shape[-1] - edge_cols)
w_mn = w_mn[:, :, interior]

backend_wdm = CompositeSensitivityBackend(wdm_set, tdi_generation=2)
S_wdm = np.asarray(backend_wdm("check", PSD_INJECTION).sens_mat)[:, :, :, interior]
layer_f = np.arange(wdm_set.ind_min_f, wdm_set.ind_max_f + 1) * wdm_set.layer_df
n_layers, n_time = w_mn.shape[1], w_mn.shape[2]
assert S_wdm.shape == (3, 3, n_layers, n_time), S_wdm.shape

# per-layer whitened variance (diagonal) and full-matrix chi2/dof
layer_var = (w_mn**2 / np.einsum("iimn->imn", S_wdm)).mean(axis=-1)  # (3,mf)
Cinv = np.linalg.inv(np.moveaxis(S_wdm, (0, 1), (-2, -1)))  # (mf,nt,3,3)
wv = np.moveaxis(w_mn, 0, -1)  # (mf,nt,3)
chi2_pix = np.einsum("mni,mnij,mnj->mn", wv, Cinv, wv)
chi2_layer = chi2_pix.mean(axis=-1) / 3.0
chi2_dof = chi2_pix.mean() / 3.0
npix = 3 * n_layers * n_time
print(f"\n--- WDM ({n_layers} layers x {n_time} pixels, "
      f"layer_df={wdm_set.layer_df * 1e3:.3f} mHz) ---")
print(f"  whitened chi2/dof (full 3x3) : {chi2_dof:.4f} "
      f"(expect 1 within ~{np.sqrt(2.0 / npix):.0e})")
print(f"  chi2/dof excluding lowest layer: {chi2_pix[1:].mean() / 3:.4f}")
print(f"  worst layers: " + ", ".join(
    f"{layer_f[i] * 1e3:.2f} mHz -> {chi2_layer[i]:.2f}"
    for i in np.argsort(np.abs(chi2_layer - 1))[-3:][::-1]))

# --------------------------------------------------------------------- plot
from matplotlib.ticker import NullFormatter

COL = {"X2": "#2a78d6", "Y2": "#1baf7a", "Z2": "#eda100"}  # categorical 1-3
GRAY = "#6b7280"
edges = np.geomspace(FMIN, FMAX, 41)
mid = np.sqrt(edges[:-1] * edges[1:])
binned = lambda y: np.array([np.median(y[(f_act >= lo) & (f_act < hi)])
                             for lo, hi in zip(edges[:-1], edges[1:])])
fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
for ax in axes:
    ax.grid(True, color="#e5e7eb", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

ax = axes[0]
for i, ch in enumerate(CHANNELS):
    ax.loglog(mid, binned(D_fd[i, i].real), color=COL[ch], lw=1.8, label=ch)
ax.loglog(mid, binned(S_fd[0, 0]), color="#111827", lw=1.8, ls="--",
          label="model (Soms_d=15e-12, Sa_a=3e-15)")
ax.set(xlabel="f [Hz]", ylabel="one-sided PSD [1/Hz]",
       title="FD: Welch PSD vs model (binned median)")
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
for i, ch in enumerate(CHANNELS):
    ax.semilogx(mid, binned(D_fd[i, i].real / S_fd[i, i]),
                color=COL[ch], lw=1.8, label=ch)
ax.axhline(1.0, color=GRAY, lw=1.0, ls=":")
ax.set(xlabel="f [Hz]", ylabel="data / model", title="FD: PSD ratio (binned median)")
ax.legend(frameon=False, fontsize=9)

ax = axes[2]
for i, ch in enumerate(CHANNELS):
    ax.semilogx(layer_f, layer_var[i], color=COL[ch], lw=1.8, label=ch)
ax.semilogx(layer_f, chi2_layer, color="#111827", lw=1.8, ls="--",
            label="full 3x3 chi2/dof")
ax.axhline(1.0, color=GRAY, lw=1.0, ls=":")
ax.set(xlabel="layer frequency [Hz]", ylabel="whitened variance",
       title="WDM: per-layer whitened variance")
ax.text(0.98, 0.72, f"chi2/dof = {chi2_dof:.4f}", transform=ax.transAxes,
        ha="right", fontsize=10, color="#111827")
ax.legend(frameon=False, fontsize=9)

for ax in axes:
    ax.set_xlim(FMIN, FMAX)
    ax.xaxis.set_minor_formatter(NullFormatter())

fig.savefig("check_noise_model_fit.png", dpi=150)
print("\nplot -> check_noise_model_fit.png")
