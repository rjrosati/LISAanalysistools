#!/usr/bin/env python3
"""Match the analytic galaxy-modulation model to a mojito L1 full-galaxy file.

`galaxy_modulation.py` ports the glass_galaxy.c pipeline, but its density
model, start time, and constellation phases are tuned to the old glass setup.
This script matches the *current* galaxy model empirically, directly against
the TDI data in a mojito L1 GB file (e.g.
``GB_731d_2.5s_L1_source0_0_*.h5``):

  1. Extract the true orbit phases from ``orbits/sc_position_*``:
     the guiding-center phase alpha(t) (interpolated, so the Kepler
     eccentricity wobble is kept) and the per-spacecraft constellation
     angles beta_i fitted from the spacecraft z-motion
     (z_i = -sqrt(3) a e cos(alpha - beta_i), glass convention; the
     mojito orbits follow the same handedness, beta_{i+1} - beta_i = +2pi/3).
  2. Measure the empirical modulation: Hann-windowed chunked cross-spectra
     of X2/Y2/Z2 summed over the confusion band, normalized so
     mean (XX+YY+ZZ)/3 = 1.
  3. Fit the sky's real spherical-harmonic coefficients a_lm (l = 0, 2, 4 —
     the complete support of the low-frequency Michelson response kernels)
     to all six curves by linear least squares. The modulation is linear in
     a_lm, so this is exact and replaces the analytic bulge+disk density,
     whose best fit is both worse and driven to unphysical parameters.
  4. Write ``modulation.dat`` (t XX YY ZZ XY XZ YZ) evaluated from the
     fitted a_lm on a dense grid, plus a verification plot and stats
     against the measured curves.

Verified on GB_731d_2.5s_L1_source0_0_20251205T020733787241Z.h5:
per-channel rms 0.04-0.06 with white (chunk-uncorrelated) residuals, i.e.
the fit absorbs all the systematic annual modulation and the residual is
periodogram estimator noise. The fitted a_lm are stable to chunk length;
the small l=4 imaginary terms shift with the analysis band (the confusion
anisotropy is weakly frequency-dependent), so set --fmin/--fmax to the band
your application actually uses.
"""

import argparse
import math

import h5py
import numpy as np

import galaxy_modulation as gm

CHANS = ("XX", "YY", "ZZ", "XY", "XZ", "YZ")
# (l, m) slots with support in the response kernels
LM = [(0, 0), (2, 0), (2, 1), (2, 2), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
COLS = [("R", l, m) for l, m in LM] + [("I", l, m) for l, m in LM if m > 0]


# --- orbit phases from the mojito file ----------------------------------------
def extract_orbit_phases(fh):
    """Return (t_orbit, alpha_unwrapped, betas[3]) from orbits/sc_position_*."""
    p = [fh[f"orbits/sc_position_{i}"][:] for i in (1, 2, 3)]
    s = fh["orbits/sampling"]
    t = s.attrs["t0"] + s.attrs["dt"] * np.arange(s.attrs["size"])
    cen = sum(p) / 3.0
    alpha = np.unwrap(np.arctan2(cen[:, 1], cen[:, 0]))
    betas = []
    for pi in p:
        # glass convention: z_i - z_cen = -sqrt(3) a e cos(alpha - beta_i)
        dz = pi[:, 2] - cen[:, 2]
        design = np.column_stack([np.cos(alpha), np.sin(alpha)])
        (c_coef, s_coef), *_ = np.linalg.lstsq(design, dz, rcond=None)
        betas.append(np.arctan2(-s_coef, -c_coef) % (2.0 * np.pi))
    return t, alpha, betas


# --- empirical modulation from the TDI data ------------------------------------
def measure_modulation(fh, chunk_days, fmin, fmax):
    """Chunked Hann cross-spectra of X2/Y2/Z2 in [fmin, fmax] -> 6 curves."""
    samp = fh["tdis/sampling"]
    dt, t0, n = samp.attrs["dt"], samp.attrs["t0"], samp.attrs["size"]
    x, y, z = (fh[f"tdis/{k}"][:] for k in ("X2", "Y2", "Z2"))

    nc = int(round(chunk_days * 86400.0 / dt))
    nchunks = int(n // nc)
    freqs = np.fft.rfftfreq(nc, dt)
    sel = (freqs >= fmin) & (freqs <= fmax)
    win = np.hanning(nc)

    tc = t0 + (np.arange(nchunks) + 0.5) * nc * dt
    out = {k: np.empty(nchunks) for k in CHANS}
    for i in range(nchunks):
        sl = slice(i * nc, (i + 1) * nc)
        xf = np.fft.rfft(win * x[sl])[sel]
        yf = np.fft.rfft(win * y[sl])[sel]
        zf = np.fft.rfft(win * z[sl])[sel]
        out["XX"][i] = np.sum(np.abs(xf) ** 2)
        out["YY"][i] = np.sum(np.abs(yf) ** 2)
        out["ZZ"][i] = np.sum(np.abs(zf) ** 2)
        out["XY"][i] = np.sum((xf * np.conj(yf)).real)
        out["XZ"][i] = np.sum((xf * np.conj(zf)).real)
        out["YZ"][i] = np.sum((yf * np.conj(zf)).real)

    av = np.mean((out["XX"] + out["YY"] + out["ZZ"]) / 3.0)
    for k in out:
        out[k] /= av
    return tc, out


# --- kernel design matrix -------------------------------------------------------
def design_matrix(alpha, betas):
    """G such that concat(curves over CHANS) = G @ alm_vec (COLS ordering)."""
    cb, sb = {}, {}
    for name, b in zip("xyz", betas):
        cb[name], sb[name] = gm._trig_powers(b)
    n = len(alpha)
    G = np.zeros((6 * n, len(COLS)))
    for i in range(n):
        ca, sa = gm._trig_powers(alpha[i])
        ker = {
            "XX": gm.kernel_XX(ca, sa, cb["x"], sb["x"]),
            "YY": gm.kernel_XX(ca, sa, cb["y"], sb["y"]),
            "ZZ": gm.kernel_XX(ca, sa, cb["z"], sb["z"]),
            "XY": gm.kernel_XY(ca, sa, cb["x"], sb["x"]),
            "XZ": gm.kernel_XY(ca, sa, cb["z"], sb["z"]),
            "YZ": gm.kernel_XY(ca, sa, cb["y"], sb["y"]),
        }
        for ci, ch in enumerate(CHANS):
            kR, kI = ker[ch]
            for j, (part, l, m) in enumerate(COLS):
                w = 1.0 if m == 0 else 2.0
                G[ci * n + i, j] = w * (kR[l, m] if part == "R" else kI[l, m])
    return G


def curves_from_alm(alpha, betas, coef):
    """Evaluate the 6 modulation curves from an alm coefficient vector."""
    mod = design_matrix(alpha, betas) @ coef
    n = len(alpha)
    return {ch: mod[ci * n : (ci + 1) * n] for ci, ch in enumerate(CHANS)}


# --- verification plot ----------------------------------------------------------
def verification_plot(t_days, data, model, stats, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface = "#fcfcfb"
    ink, ink2, muted = "#0b0b0b", "#52514e", "#898781"
    grid_c, axis_c = "#e1e0d9", "#c3c2b7"
    series = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
    model_c = "#2a78d6"

    fig = plt.figure(figsize=(13, 9.5), dpi=150, facecolor=surface)
    gs = fig.add_gridspec(
        3, 3, height_ratios=[1, 1, 0.8], hspace=0.42, wspace=0.22,
        left=0.06, right=0.97, top=0.93, bottom=0.07,
    )

    def style(ax):
        ax.set_facecolor(surface)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(axis_c)
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(True, color=grid_c, linewidth=0.6)
        ax.set_axisbelow(True)

    for ci, ch in enumerate(CHANS):
        ax = fig.add_subplot(gs[ci // 3, ci % 3])
        style(ax)
        ax.plot(t_days, model[ch], color=model_c, lw=2.0, zorder=3, label="model (fit $a_{lm}$)")
        ax.plot(
            t_days, data[ch], ls="none", marker="o", ms=3.2, mfc=ink2, mec="none",
            zorder=4, label="data (chunked spectra)",
        )
        ax.set_title(ch, fontsize=11, color=ink, fontweight="bold")
        ax.text(
            0.02, 0.04, f"rms {stats[ch][0]:.3f}   corr {stats[ch][1]:.3f}",
            transform=ax.transAxes, fontsize=8, color=muted,
        )
        if ci == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(
                handles, labels, loc="upper right", bbox_to_anchor=(0.97, 1.0),
                fontsize=9, frameon=False, labelcolor=ink2, handlelength=1.4,
            )
        if ci >= 3:
            ax.set_xlabel("days since data start", fontsize=9, color=ink2)

    axr = fig.add_subplot(gs[2, :])
    style(axr)
    order = np.argsort([data[ch][-1] - model[ch][-1] for ch in CHANS])[::-1]
    ylim = 0.0
    for rank, ci in enumerate(order):
        ch = CHANS[ci]
        resid = data[ch] - model[ch]
        ylim = max(ylim, np.abs(resid).max())
        axr.plot(t_days, resid, color=series[ci], lw=1.5)
    ylim *= 1.15
    axr.set_ylim(-ylim, ylim)
    # direct end labels (relief for low-contrast slots), staggered by rank
    for rank, ci in enumerate(order):
        ch = CHANS[ci]
        y_lab = ylim * (0.85 - 1.7 * rank / (len(CHANS) - 1))
        axr.plot(
            [t_days[-1] + 8], [y_lab], marker="o", ms=4, mfc=series[ci],
            mec="none", clip_on=False,
        )
        axr.text(
            t_days[-1] + 14, y_lab, ch, fontsize=8, color=ink2, va="center",
        )
    axr.axhline(0.0, color=axis_c, lw=1.0)
    axr.set_title("residual  (data − model)", fontsize=11, color=ink, fontweight="bold")
    axr.set_xlabel("days since data start", fontsize=9, color=ink2)

    fig.suptitle(
        "Galaxy modulation: mojito full-galaxy data vs fitted $a_{lm}$ model",
        fontsize=13, color=ink, fontweight="bold",
    )
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


# --- driver ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("l1_file", help="mojito L1 GB h5 file (full galaxy)")
    ap.add_argument("--chunk-days", type=float, default=7.0, help="chunk length [days]")
    ap.add_argument("--fmin", type=float, default=5e-4, help="band lower edge [Hz]")
    ap.add_argument("--fmax", type=float, default=3e-3, help="band upper edge [Hz]")
    ap.add_argument("--nout", type=int, default=200, help="dense output samples")
    ap.add_argument("-o", "--out", default="modulation_match.dat", help="output .dat")
    ap.add_argument("--plot", default="modulation_match.png", help="verification plot")
    ap.add_argument("--alm-out", default=None, help="optional .npz to store fitted alm")
    args = ap.parse_args()

    with h5py.File(args.l1_file, "r") as fh:
        t_orb, alpha_orb, betas = extract_orbit_phases(fh)
        tc, data = measure_modulation(fh, args.chunk_days, args.fmin, args.fmax)
        samp = fh["tdis/sampling"]
        t0, dur = samp.attrs["t0"], samp.attrs["duration"]

    print("constellation betas [rad]:", " ".join(f"{b:.4f}" for b in betas))
    print(f"alpha(t0) = {np.interp(t0, t_orb, alpha_orb) % (2 * math.pi):.4f} rad")

    # fit alm to the measured curves
    alpha_c = np.interp(tc, t_orb, alpha_orb)
    G = design_matrix(alpha_c, betas)
    y = np.concatenate([data[ch] for ch in CHANS])
    coef, *_ = np.linalg.lstsq(G, y, rcond=None)

    # renormalize exactly: mean (XX+YY+ZZ)/3 = 1 on the fitted curves
    model_c = curves_from_alm(alpha_c, betas, coef)
    av = np.mean((model_c["XX"] + model_c["YY"] + model_c["ZZ"]) / 3.0)
    coef /= av
    model_c = {ch: v / av for ch, v in model_c.items()}

    stats = {}
    print("\nverification against data (per channel):")
    for ch in CHANS:
        rms = float(np.sqrt(np.mean((data[ch] - model_c[ch]) ** 2)))
        corr = float(np.corrcoef(data[ch], model_c[ch])[0, 1])
        white = float(np.var(np.diff(data[ch] - model_c[ch]))
                      / (2.0 * np.var(data[ch] - model_c[ch])))
        stats[ch] = (rms, corr)
        print(f"  {ch}: rms {rms:.4f}  corr {corr:+.4f}  residual whiteness {white:.2f}")

    print("\nfitted a_lm (normalized):")
    for j, (part, l, m) in enumerate(COLS):
        print(f"  {part}[{l}][{m}] = {coef[j]:+.6f}")

    # dense output grid (padded like galaxy_modulation.py for spline use)
    dt_out = dur / (args.nout - 3)
    t_out = t0 + dt_out * np.arange(args.nout) - dt_out
    alpha_out = np.interp(t_out, t_orb, alpha_orb)
    model_out = curves_from_alm(alpha_out, betas, coef)
    np.savetxt(
        args.out,
        np.column_stack([t_out] + [model_out[ch] for ch in CHANS]),
        fmt="%f " + " ".join(["%.10f"] * 6),
    )
    print(f"\nwrote {args.out} ({args.nout} samples)")

    if args.alm_out:
        almR = np.zeros((gm.LMAX + 1, gm.LMAX + 1))
        almI = np.zeros_like(almR)
        for j, (part, l, m) in enumerate(COLS):
            (almR if part == "R" else almI)[l, m] = coef[j]
        np.savez(args.alm_out, almR=almR, almI=almI, betas=np.array(betas),
                 t_orbit=t_orb, alpha_orbit=alpha_orb)
        print(f"wrote {args.alm_out}")

    t_days = (tc - t0) / 86400.0
    verification_plot(t_days, data, model_c, stats, args.plot)
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
