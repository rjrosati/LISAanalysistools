"""PE run for the noise + galfor + SGWB global-fit settings.

Single-rank driver (no MPI ranks needed, same pattern as
noise_sgwb_gf_smoke.py): loads the noise_sgwb settings, draws the synthetic
data, starts the whole ensemble at the *injected* parameters (plus a small
relative scatter so the stretch move is non-degenerate), and loops the joint
psd+galfor+sgwb ``PSDMove`` until the requested number of cold-chain samples
have been collected. Saves the chain and makes a corner plot with the
injection truths overlaid.

Run from the repo root:

    python scripts/diagnostics/noise_sgwb_gf_pe.py                 # full 10k run
    python scripts/diagnostics/noise_sgwb_gf_pe.py --samples 200   # quick check

Note: the settings file's ``nwalkers = 4`` is a smoke-test value; a stretch
move in the 9-dim joint space needs more walkers (>= 2*ndim), so this driver
overrides it (``--nwalkers``, default 24) before building the state/ACs.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "global_fit_input"))

from mpi4py import MPI

from noise_sgwb_global_fit_settings import (
    get_global_fit_settings,
    PSD_INJECTION,
    GALFOR_INJECTION,
    SGWB_INJECTION,
)

from eryn.state import BranchSupplemental
from eryn.moves.tempering import TemperatureControl

from lisatools.globalfit.run import GlobalFit
from lisatools.globalfit.engine import GlobalFitInfo
from lisatools.globalfit.moves import PSDMove

BRANCHES = ("psd", "galfor", "sgwb")
INJECTIONS = {
    "psd": PSD_INJECTION,
    "galfor": GALFOR_INJECTION,
    "sgwb": SGWB_INJECTION,
}
LABELS = [
    r"$S_{\rm oms}$",
    r"$S_{\rm tm}$",
    r"gal amp",
    r"gal $f_k$",
    r"gal $\alpha$",
    r"gal sl1",
    r"gal sl2",
    r"$\log_{10} A_{\rm gw}$",
    r"$\alpha_{\rm gw}$",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=10_000, help="cold-chain samples to collect")
    p.add_argument("--nwalkers", type=int, default=24, help="override settings nwalkers")
    p.add_argument(
        "--thin", type=int, default=5, help="stretch-move repeats between recorded samples"
    )
    p.add_argument("--scatter", type=float, default=1e-3, help="relative start scatter")
    p.add_argument("--outdir", type=str, default="./gf_output", help="output directory")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    curr = get_global_fit_settings()
    # smoke-test settings use nwalkers=4; bump for a usable stretch-move ensemble
    curr.general_info.nwalkers = args.nwalkers
    gf = GlobalFit(curr, MPI.COMM_WORLD)
    ntemps, nwalkers = gf.ntemps, gf.nwalkers

    priors = {}
    for name in curr.branch_names:
        priors.update(curr.source_info[name].priors)

    state = gf.load_info(priors)

    # ------------------------------------------------------------------
    # Start at the injection (small relative scatter keeps the ensemble
    # non-degenerate for the stretch move). PSDMove.propose recomputes
    # log_like/log_prior from the coords on entry, so no need to set them.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    for name in BRANCHES:
        inj = np.asarray(INJECTIONS[name], dtype=float)
        coords = state.branches[name].coords  # (ntemps, nwalkers, 1, ndim)
        coords[:] = inj[None, None, None, :] * (
            1.0 + args.scatter * rng.standard_normal(coords.shape)
        )

    supps = BranchSupplemental(
        {"walker_inds": np.tile(np.arange(nwalkers), (ntemps, 1))},
        base_shape=(ntemps, nwalkers),
        copy=True,
    )
    state.supplemental = supps

    acs = gf.setup_acs(state)
    general_info = curr.general_info

    # joint psd+galfor+sgwb PE move, as setup_recipe builds it
    effective_ndim = sum(curr.ndims[key] for key in BRANCHES)
    temperature_control = TemperatureControl(
        effective_ndim, nwalkers, ntemps=ntemps, Tmax=1e6, permute=False
    )
    move = PSDMove(
        acs,
        priors,
        num_repeats=args.thin,
        max_logl_mode=False,
        live_dangerously=True,
        temperature_control=temperature_control,
        sensitivity_backend=general_info.sensitivity_backend,
        psd_transform_fn=curr.source_info["psd"].transform_fn,
        name="psd+galfor+sgwb pe move",
    )
    move.accepted = np.zeros((ntemps, nwalkers))

    model = GlobalFitInfo(acs, map, np.random.RandomState(2026))

    n_iter = int(np.ceil(args.samples / nwalkers))
    chain = np.empty((n_iter, nwalkers, effective_ndim))
    logl_chain = np.empty((n_iter, nwalkers))

    print(
        f"collecting {n_iter} iterations x {nwalkers} walkers = "
        f"{n_iter * nwalkers} cold-chain samples (thin={args.thin})"
    )
    t_start = time.perf_counter()
    for it in range(n_iter):
        state, accepted = move.propose(model, state)
        chain[it] = np.concatenate(
            [state.branches_coords[name][0, :, 0, :] for name in BRANCHES], axis=-1
        )
        logl_chain[it] = state.log_like[0]
        if it == 0:
            per_it = time.perf_counter() - t_start
            print(f"first iteration: {per_it:.1f} s -> ETA ~{per_it * n_iter / 60:.1f} min")
        if (it + 1) % 25 == 0 or it == n_iter - 1:
            elapsed = time.perf_counter() - t_start
            print(
                f"iter {it + 1}/{n_iter}  max cold logL = {state.log_like[0].max():.2f}  "
                f"[{elapsed / 60:.1f} min elapsed]"
            )
            # checkpoint so a partial run is still usable
            np.save(outdir / "noise_sgwb_pe_chain.npy", chain[: it + 1])
            np.save(outdir / "noise_sgwb_pe_logl.npy", logl_chain[: it + 1])

    samples = chain.reshape(-1, effective_ndim)
    print(f"done: {samples.shape[0]} samples saved to {outdir / 'noise_sgwb_pe_chain.npy'}")

    # ------------------------------------------------------------------
    # Corner plot with the injection truths.
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import corner

    truths = np.concatenate([np.asarray(INJECTIONS[name], dtype=float) for name in BRANCHES])
    fig = corner.corner(
        samples,
        labels=LABELS,
        truths=truths,
        show_titles=True,
        title_fmt=".3g",
        quantiles=[0.16, 0.5, 0.84],
    )
    fig_path = outdir / "noise_sgwb_pe_corner.png"
    fig.savefig(fig_path, dpi=150)
    print(f"corner plot saved to {fig_path}")


if __name__ == "__main__":
    main()
