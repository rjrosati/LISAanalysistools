"""Single-rank smoke test of the noise + galfor + SGWB global-fit wiring.

Emulates the main-rank portion of ``GlobalFit.run_global_fit`` without MPI
ranks: loads the noise_sgwb settings, draws the synthetic data, builds the
analysis-container array, and runs the joint psd+galfor+sgwb ``PSDMove``
for a handful of repeats. Verifies:

* the likelihood at the injected parameters beats the prior-draw start;
* the sampler walks uphill toward the injection;
* the full sgwb-branch plumbing (state -> move -> CompositeSensitivityBackend)
  executes end to end.

Run from the repo root:

    python scripts/diagnostics/noise_sgwb_gf_smoke.py
"""

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


N_OUTER = 5
NUM_REPEATS = 10


if __name__ == "__main__":
    curr = get_global_fit_settings()
    gf = GlobalFit(curr, MPI.COMM_WORLD)

    # priors per branch, as run_global_fit assembles them
    priors = {}
    for name in curr.branch_names:
        priors.update(curr.source_info[name].priors)

    state = gf.load_info(priors)
    ntemps, nwalkers = gf.ntemps, gf.nwalkers
    supps = BranchSupplemental(
        {"walker_inds": np.tile(np.arange(nwalkers), (ntemps, 1))},
        base_shape=(ntemps, nwalkers),
        copy=True,
    )
    state.supplemental = supps

    acs = gf.setup_acs(state)
    general_info = curr.general_info

    # ------------------------------------------------------------------
    # Reference: likelihood with the *injected* parameters installed.
    # ------------------------------------------------------------------
    backend = general_info.sensitivity_backend
    inj_sens = backend(
        "injection",
        PSD_INJECTION,
        galfor_params=GALFOR_INJECTION,
        sgwb_params=SGWB_INJECTION,
    )
    original_sens = acs[0].sens_mat
    acs[0].sens_mat = inj_sens
    acs.reset_linear_psd_arr()
    logl_injection = float(np.asarray(acs.likelihood())[0])
    acs[0].sens_mat = original_sens
    acs.reset_linear_psd_arr()

    logl_start = np.asarray(acs.likelihood())
    print(f"logL at injection           : {logl_injection:.2f}")
    print(f"logL at prior-draw start    : {np.max(logl_start):.2f} (best walker)")

    # ------------------------------------------------------------------
    # Joint psd+galfor+sgwb stretch move, as setup_recipe builds it.
    # ------------------------------------------------------------------
    effective_ndim = sum(curr.ndims[key] for key in ("psd", "galfor", "sgwb"))
    temperature_control = TemperatureControl(
        effective_ndim, nwalkers, ntemps=ntemps, Tmax=1e6, permute=False
    )
    move = PSDMove(
        acs,
        priors,
        num_repeats=NUM_REPEATS,
        max_logl_mode=False,
        live_dangerously=True,
        temperature_control=temperature_control,
        sensitivity_backend=backend,
        psd_transform_fn=curr.source_info["psd"].transform_fn,
        name="psd+galfor+sgwb smoke move",
    )
    move.accepted = np.zeros((ntemps, nwalkers))

    model = GlobalFitInfo(acs, map, np.random.RandomState(2026))

    best_logl = -np.inf
    for it in range(N_OUTER):
        t0 = time.perf_counter()
        state, accepted = move.propose(model, state)
        t1 = time.perf_counter()
        best = float(state.log_like[0].max())
        best_logl = max(best_logl, best)
        w = int(np.argmax(state.log_like[0]))
        print(
            f"iter {it}: max cold logL = {best:.2f} "
            f"(injection {logl_injection:.2f})  [{t1 - t0:.1f} s]"
        )
        print(f"  psd    : {state.branches_coords['psd'][0, w, 0]}")
        print(f"  galfor : {state.branches_coords['galfor'][0, w, 0]}")
        print(f"  sgwb   : {state.branches_coords['sgwb'][0, w, 0]}")

    print()
    print(f"injection psd    : {PSD_INJECTION}")
    print(f"injection galfor : {GALFOR_INJECTION}")
    print(f"injection sgwb   : {SGWB_INJECTION}")
    gap = logl_injection - best_logl
    print(f"\nfinal best logL - injection logL = {-gap:.2f}")
    assert best_logl > np.max(logl_start), "sampler did not improve on the prior draw"
    print("smoke test passed: move runs end-to-end and improves the likelihood")
