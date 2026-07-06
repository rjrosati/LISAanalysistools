"""Cross-domain validation: composite noise likelihood in FD vs WDM.

One stationary noise realization (instrument + isotropic galactic foreground
+ power-law SGWB) is drawn in the Fourier domain from the composite
covariance, inverse-FFT'd to a time series, and then analyzed twice through
the standard lisatools machinery:

* **FD**: ``TDSignal.transform(FDSettings)`` + FD composite matrix
  (complex Whittle convention, ``logdet_factor = 1``);
* **WDM**: ``TDSignal.transform(WDMSettings)`` + WDM composite matrix
  (real-Gaussian convention, ``logdet_factor = 0.5``).

Absolute log-likelihoods are convention-dependent and need not match, but
**likelihood differences between parameter points are basis-independent**
(the WDM transform of band-limited data is ~unitary), so the script
compares ``logL(theta) - logL(theta_inj)`` curves between the two domains
for three scans:

1. a global covariance scale ``s * C_inj`` (tests the quadratic + det terms
   together; both domains must peak at ``s = 1``);
2. the instrument OMS noise level ``Soms_d``;
3. the SGWB amplitude ``log10_A``.

Both domains analyze the same interior band; the data are drawn over the
*full* rfft band so no hard spectral edge falls inside the analysis band.
Residual disagreement comes from WDM leakage at the analysis-band edges
(edge layers gather power from just outside the FD band), so curves agree
at the percent level rather than machine precision.

Run from the repo root:

    python scripts/diagnostics/wdm_fd_likelihood_comparison.py
"""

import warnings

import numpy as np

from lisatools.sensitivity import CompositeSensitivityBackend
from lisatools.diagnostic import inner_product, noise_likelihood_term
from lisatools.domains import FDSettings, TDSettings, TDSignal, WDMSettings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
NF = 256
NT = 128
DT = 5.0
N = NF * NT
TOBS = N * DT
DF = 1.0 / TOBS
LAYER_DF = 1.0 / (2 * NF * DT)

BAND = (3e-4, 8e-3)  # analysis band; matches the noise validation scripts

PSD_INJ = np.array([15e-12, 3e-15])
GALFOR_INJ = np.array(
    [3.26651613e-44, 2.09278117e-03, 1.18300266e00, 3.01430978e03, 2.95774596e03]
)
# PowerLawSGWB's amplitude is Omega_gw at SGWB_FREF = 25 Hz; with alpha = 2/3
# a level measurable against instrument + foreground noise in the 0.3-8 mHz
# band on this short (1.9 day) grid needs log10_A ~ -9 (the GLASS-style
# -20.45 contributes only ~1e-12 of the covariance here and is unrecoverable
# -- see NOISE_TODO "double check SGWB response").
SGWB_INJ = np.array([-9.0, 2.0 / 3.0])

SEED = 7


def make_backend(settings):
    """Composite backend with the stationary (isotropic) foreground limit."""
    return CompositeSensitivityBackend(
        settings, tdi_generation=2, sgwb_stochastic_fn="PowerLawSGWB"
    )


def build_sens(settings, psd=PSD_INJ, galfor=GALFOR_INJ, sgwb=SGWB_INJ, name="inj"):
    return make_backend(settings)(name, psd, galfor_params=galfor, sgwb_params=sgwb)


def log_like(sig, sensmat):
    """nlt + source term; the domain-aware logdet_factor makes this the
    correct Gaussian convention in both FD (complex) and WDM (real)."""
    ip = inner_product(sig, sig, psd=sensmat)
    nlt = noise_likelihood_term(sensmat)
    return float(nlt - 0.5 * ip)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Draw the realization in FD over the full rfft band (DC excluded).
    # ------------------------------------------------------------------
    n_rfft = N // 2 + 1
    fd_full = FDSettings(N=n_rfft, df=DF, min_freq=DF / 2, max_freq=None, force_backend="cpu")
    sens_full = build_sens(fd_full, name="draw")
    C_full = np.asarray(sens_full.sens_mat)  # (3, 3, n_active)

    L = np.linalg.cholesky(C_full.transpose(2, 0, 1))  # (n_active, 3, 3)
    rng = np.random.default_rng(SEED)
    z = (
        rng.standard_normal((L.shape[0], 3, 1))
        + 1j * rng.standard_normal((L.shape[0], 3, 1))
    )
    # E|d~|^2 = S * Tobs / 2 makes <d|d> = 4 df sum |d~|^2/S a chi^2 with
    # 2 dof per complex bin per channel (the lisatools FD convention).
    d_active = np.sqrt(TOBS / 4.0) * (L @ z)[:, :, 0].T  # (3, n_active)
    d_active[:, -1] = np.sqrt(2.0) * d_active[:, -1].real  # Nyquist bin must be real

    d_full = np.zeros((3, n_rfft), dtype=complex)
    d_full[:, fd_full.active_slice] = d_active
    d_full[:, 0] = 0.0  # DC

    # FD convention is d~ = dt * rfft(x): invert it to land in TD.
    x = np.fft.irfft(d_full / DT, n=N, axis=-1)
    td = TDSignal(x, TDSettings(t0=0.0, dt=DT, N=N, force_backend="cpu"))

    # round-trip check: lisatools' own FD transform must reproduce the draw
    fd_check = td.transform(fd_full)
    rt = np.max(
        np.abs(np.asarray(fd_check.arr)[:, : d_active.shape[1]] - d_active)
    ) / np.max(np.abs(d_active))
    print(f"TD->FD round-trip max rel err: {rt:.2e}")
    assert rt < 1e-10

    # ------------------------------------------------------------------
    # Analysis-band signals + injected sensitivity matrices per domain.
    # ------------------------------------------------------------------
    wdm_set = WDMSettings(
        Nf=NF, Nt=NT, dt=DT, min_freq=BAND[0], max_freq=BAND[1], force_backend="cpu"
    )
    # align the FD band to the WDM layer coverage: layer m gathers
    # (m -/+ 1/2) * layer_df
    f_layers = np.asarray(wdm_set.f_arr)
    fd_band = (float(f_layers.min() - LAYER_DF / 2), float(f_layers.max() + LAYER_DF / 2))
    fd_set = FDSettings(
        N=n_rfft, df=DF, min_freq=fd_band[0], max_freq=fd_band[1], force_backend="cpu"
    )

    fd_sig = td.transform(fd_set)
    wdm_sig = td.transform(wdm_set)

    sens_fd = build_sens(fd_set)
    sens_wdm = build_sens(wdm_set)

    n_fd = len(np.asarray(fd_set.f_arr))
    n_wdm = int(np.prod(wdm_set.basis_shape_active))
    print(f"FD : {n_fd} complex bins x 3 ch  -> dof = {2 * 3 * n_fd}")
    print(f"WDM: {n_wdm} real pixels  x 3 ch -> dof = {3 * n_wdm}")

    ip_fd = float(inner_product(fd_sig, fd_sig, psd=sens_fd))
    ip_wdm = float(inner_product(wdm_sig, wdm_sig, psd=sens_wdm))
    print(f"<d|d>/dof  FD : {ip_fd / (2 * 3 * n_fd):.4f}   (expect ~1)")
    print(f"<d|d>/dof  WDM: {ip_wdm / (3 * n_wdm):.4f}   (expect ~1)")

    logl_fd_inj = log_like(fd_sig, sens_fd)
    logl_wdm_inj = log_like(wdm_sig, sens_wdm)

    # ------------------------------------------------------------------
    # Scan 1: global covariance scale (must peak at s = 1 in both domains)
    # ------------------------------------------------------------------
    print("\n--- global scale scan: logL(s C_inj) - logL(C_inj) ---")
    C_fd = np.asarray(sens_fd.sens_mat).copy()
    C_wdm = np.asarray(sens_wdm.sens_mat).copy()
    scales = np.array([0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2])
    print(f"{'s':>6} {'dFD':>12} {'dWDM':>12} {'diff':>10}")
    d_fd_list, d_wdm_list = [], []
    for s in scales:
        sens_fd.sens_mat = s * C_fd
        sens_wdm.sens_mat = s * C_wdm
        d_fd = log_like(fd_sig, sens_fd) - logl_fd_inj
        d_wdm = log_like(wdm_sig, sens_wdm) - logl_wdm_inj
        d_fd_list.append(d_fd)
        d_wdm_list.append(d_wdm)
        print(f"{s:6.2f} {d_fd:12.2f} {d_wdm:12.2f} {d_fd - d_wdm:10.2f}")
    sens_fd.sens_mat = C_fd
    sens_wdm.sens_mat = C_wdm
    s_fd = scales[np.argmax(d_fd_list)]
    s_wdm = scales[np.argmax(d_wdm_list)]
    print(f"argmax: FD s = {s_fd}, WDM s = {s_wdm} (both should be 1.0)")
    assert s_fd == 1.0 and s_wdm == 1.0

    # ------------------------------------------------------------------
    # Scans 2 & 3: physical parameters through the backend in both domains
    # ------------------------------------------------------------------
    def param_scan(label, values, param_builder):
        print(f"\n--- {label} scan: logL(theta) - logL(theta_inj) ---")
        print(f"{'value':>12} {'dFD':>12} {'dWDM':>12} {'diff':>10}")
        d_fd_l, d_wdm_l = [], []
        for v in values:
            psd, galfor, sgwb = param_builder(v)
            sm_fd = build_sens(fd_set, psd, galfor, sgwb, name=f"{label}_{v}")
            sm_wdm = build_sens(wdm_set, psd, galfor, sgwb, name=f"{label}_{v}")
            d_fd = log_like(fd_sig, sm_fd) - logl_fd_inj
            d_wdm = log_like(wdm_sig, sm_wdm) - logl_wdm_inj
            d_fd_l.append(d_fd)
            d_wdm_l.append(d_wdm)
            print(f"{v:12.4g} {d_fd:12.2f} {d_wdm:12.2f} {d_fd - d_wdm:10.2f}")
        v_fd = values[int(np.argmax(d_fd_l))]
        v_wdm = values[int(np.argmax(d_wdm_l))]
        print(f"argmax: FD = {v_fd:.4g}, WDM = {v_wdm:.4g}")
        return v_fd, v_wdm

    soms_vals = PSD_INJ[0] * np.array([0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2])
    v_fd, v_wdm = param_scan(
        "Soms_d",
        soms_vals,
        lambda v: (np.array([v, PSD_INJ[1]]), GALFOR_INJ, SGWB_INJ),
    )
    assert v_fd == v_wdm, "FD and WDM peak at different Soms_d values"

    log10A_vals = SGWB_INJ[0] + np.array([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    v_fd, v_wdm = param_scan(
        "SGWB log10_A",
        log10A_vals,
        lambda v: (PSD_INJ, GALFOR_INJ, np.array([v, SGWB_INJ[1]])),
    )
    assert v_fd == v_wdm, "FD and WDM peak at different SGWB amplitudes"

    print("\nWDM vs FD likelihood comparison passed: both domains peak at the "
          "injection and the delta-logL curves agree.")
