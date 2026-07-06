"""Scratch profiler: break down one CompositeSensitivityBackend call by phase."""
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "global_fit_input"))

from noise_sgwb_global_fit_settings import (
    get_global_fit_settings,
    PSD_INJECTION,
    GALFOR_INJECTION,
    SGWB_INJECTION,
)


def timeit(fn, n=5):
    fn()  # warm
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e3  # ms


curr = get_global_fit_settings()
backend = curr.general_info.sensitivity_backend
settings = backend.basis_settings

# ---- full backend call (what PSDMove does per walker per proposal) ----
def full_call():
    return backend(
        "prof", PSD_INJECTION, galfor_params=GALFOR_INJECTION, sgwb_params=SGWB_INJECTION
    )

print(f"full backend call:                  {timeit(full_call):8.1f} ms")

sm = full_call()
inst, galfor, sgwb = sm.components

# ---- exactness of the linear instrument basis vs the direct fold ----
from lisatools import detector as lisa_models
from lisatools.sensitivity import InstrumentNoise

direct = InstrumentNoise(
    tdi_generation=backend.tdi_generation,
    model=lisa_models.LISAModel(
        float(PSD_INJECTION[0]) ** 2, float(PSD_INJECTION[1]) ** 2,
        lisa_models.DefaultOrbits(), "check",
    ),
    fill_nans=backend.instrument_fill_nans,
).covariance(settings)
linear = inst.covariance(settings)
denom = np.abs(direct)
denom[denom == 0] = 1.0
print(f"  linear-basis max rel err vs fold: {np.max(np.abs(linear - direct) / denom):8.2e}")

print(f"  InstrumentNoise.covariance:       {timeit(lambda: inst.covariance(settings)):8.1f} ms")
print(f"  GalFor.base_covariance:           {timeit(lambda: galfor.base_covariance(settings)):8.1f} ms")
print(f"  GalFor.time_modulation:           {timeit(lambda: galfor.time_modulation(settings)):8.1f} ms")
print(f"  GalFor.covariance (total):        {timeit(lambda: galfor.covariance(settings)):8.1f} ms")
print(f"  SGWB.covariance:                  {timeit(lambda: sgwb.covariance(settings)):8.1f} ms")

# det/inv triggered by sens_mat assignment, lazily on first read
C = sm.sens_mat.copy()
def detinv():
    sm.sens_mat = C
    _ = sm.invC

print(f"  det/inv (assign + first read):    {timeit(detinv):8.1f} ms")

# likelihood on top of a built matrix
from lisatools.diagnostic import inner_product, noise_likelihood_term

nf, nt = C.shape[2], C.shape[3]
rng = np.random.default_rng(0)
from lisatools.domains import WDMSignal
data = WDMSignal(rng.standard_normal((3, nf, nt)) * 1e-22, settings)
print(f"  inner_product <d|d>:              {timeit(lambda: inner_product(data, data, psd=sm)):8.1f} ms")
print(f"  noise_likelihood_term:            {timeit(lambda: noise_likelihood_term(sm)):8.1f} ms")
