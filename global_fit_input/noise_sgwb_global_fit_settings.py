"""Global-fit settings: WDM-domain noise + galactic foreground + SGWB run.

Samples the instrument PSD (Soms_d, Sa_a), the 5-parameter hyperbolic-tangent
galactic foreground (with a fixed GLASS per-element time modulation), and a
2-parameter power-law SGWB, all through the WDM-domain
:class:`CompositeSensitivityBackend`.

The instrument-noise model is selectable (``INSTRUMENT_NOISE_MODEL`` below):
either the sampled 2-parameter model above, or the *fixed* noise estimates
stored in a Mojito L1 file's ``noise_estimates`` group (no psd branch is then
sampled). The confusion-noise (galfor) and SGWB components are separately
enablable via ``INCLUDE_GALFOR`` / ``INCLUDE_SGWB`` in either mode.

The data are a synthetic noise realization Cholesky-drawn from the *injected*
composite covariance (instrument + modulated foreground + SGWB), so the run
is a self-consistent end-to-end recovery test. Swap
``SyntheticCompositeNoiseProcessor`` for a Sangria/Mojito/CD1L-style loader
to run on real datasets.
"""

import logging
import shutil
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d


# ============================================================
# *** Backend selection ***
# ============================================================
try:
    import cupy as cp

    GPU_BACKEND = "cuda12x"  # change to "cuda11x" / "cuda13x" if needed
    gpu_available = True
except (ModuleNotFoundError, ImportError):
    import numpy as cp

    GPU_BACKEND = "cpu"
    gpu_available = False
# ============================================================

logger = logging.getLogger(__name__)

from eryn.moves.tempering import TemperatureControl

from lisatools.detector import EqualArmlengthOrbits
from lisatools.globalfit.run import CurrentInfoGlobalFit, GlobalFit
from lisatools.globalfit.moves import PSDMove
from lisatools.globalfit.engine import (
    GlobalFitSettings, GeneralSetup, GeneralSettings, RankInfo,
)
from lisatools.globalfit.recipe import Recipe, RecipeStep
from lisatools.globalfit.stock.erebor import (
    PSDSetup, PSDSettings,
    GalForSetup, GalForSettings,
    SGWBSetup, SGWBSettings,
)
from lisatools.sensitivity import CompositeSensitivityBackend, TabulatedNoise
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal
from lisatools.utils.utility import asnumpy

from eryn.prior import uniform_dist, ProbDistContainer


# ============================================================
# *** Injection truths ***
# ============================================================
# These parameterize the covariance the synthetic data are drawn from AND sit
# inside the sampled priors, so the run should recover them.
PSD_INJECTION = np.array([15e-12, 3e-15])  # (Soms_d, Sa_a), sqrt units
GALFOR_INJECTION = np.array(
    [
        3.26651613e-44,  # amp
        2.09278117e-03,  # fk (knee)
        1.18300266e00,   # alpha
        3.01430978e03,   # slope 1
        2.95774596e03,   # slope 2
    ]
)
# NB: PowerLawSGWB's amplitude is Omega_gw at SGWB_FREF = 25 Hz. Chosen to be detectable in short datastreams
SGWB_INJECTION = np.array([-9.5, 2.0 / 3.0])  # (log10_A @ 25 Hz, alpha)
SGWB_STOCHASTIC_FN = "PowerLawSGWB"
NOISE_SEED = 0

# GLASS modulation file (tuned to match Sangria's galaxy orientation).
MODULATION_FILE = str(Path(__file__).resolve().parent.parent / "modulation.dat")


# ============================================================
# *** Noise / component model selection ***
# ============================================================
# Instrument-noise model:
#   "sampled" — 2-parameter (Soms_d, Sa_a) instrument PSD sampled by the
#               "psd" branch (the original behavior).
#   "file"    — fixed noise estimates read from the ``noise_estimates`` group
#               of NOISE_ESTIMATE_FILE (Mojito L1 layout). No "psd" branch is
#               sampled; the synthetic data draw uses the same fixed
#               covariance, so the run recovers galfor/SGWB on top of the
#               file noise.
INSTRUMENT_NOISE_MODEL = "sampled"
NOISE_ESTIMATE_FILE = str(
    Path(__file__).resolve().parent.parent.parent
    / "NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5"
)
# Use the file estimates' daily time dependence (WDM domain; FD falls back to
# the time average automatically). "layer_constant" keeps the per-column WDM
# evaluation cheap; switch to "fold" for the exact (much slower) fold.
NOISE_ESTIMATE_TIME_DEPENDENT = True
NOISE_ESTIMATE_WDM_PSD_METHOD = "layer_constant"

# Confusion noise (galactic foreground) and SGWB are separately enablable
# regardless of the instrument-noise choice: each toggle controls both the
# injection into the synthetic data and the sampled branch.
INCLUDE_GALFOR = True
INCLUDE_SGWB = True


class FileNoiseEstimates:
    """Lazy, picklable :class:`TabulatedNoise` from a Mojito ``noise_estimates`` group.

    Duck-types the :class:`NoiseComponent` interface (``name``, ``nchannels``,
    ``covariance``) so it can be handed directly to
    ``CompositeSensitivityBackend(instrument_component=...)`` and to the
    synthetic-data processor. The HDF5 payload (~50 MB) is loaded on first
    use, so instances pickle cheaply across MPI ranks (same pattern as
    :class:`GlassModulation`).

    The stored covariance accompanies the file's ``xyz_doppler`` data, which
    mojito normalizes by the laser frequency — so the estimates are scaled by
    ``1 / laser_frequency**2`` here, putting them in the same
    fractional-frequency convention as the lisatools galfor/SGWB components.
    The estimate times are shifted onto the analysis time coordinate
    (``t = 0`` at the TDI data start).
    """

    name = "instrument"
    nchannels = 3

    def __init__(
        self,
        path: str,
        dataset: str = "XYZ",
        time_dependent: bool = True,
        wdm_psd_method: str = "layer_constant",
    ):
        self.path = path
        self.dataset = dataset
        self.time_dependent = time_dependent
        self.wdm_psd_method = wdm_psd_method
        self._component = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_component"] = None  # reload lazily on the receiving rank
        return state

    def component(self) -> TabulatedNoise:
        if self._component is None:
            import h5py

            with h5py.File(self.path, "r") as f:
                grp = f["noise_estimates"]
                cov = grp[self.dataset][:]  # (nT, nF, nch, nch), complex
                fa = grp["log_frequency_sampling"].attrs
                f_tab = np.logspace(
                    np.log10(float(fa["fmin"])), np.log10(float(fa["fmax"])), int(fa["size"])
                )
                sa = grp["sampling"].attrs
                t_tab = float(sa["t0"]) + np.arange(int(sa["size"])) * float(sa["dt"])
                # analysis domains use t = 0 at the data start
                t_tab = t_tab - float(f["tdis"]["sampling"].attrs["t0"])
                scale = 1.0 / float(f.attrs["laser_frequency"]) ** 2
            if not self.time_dependent:
                cov = cov.mean(axis=0)
                t_tab = None
            self._component = TabulatedNoise(
                f_tab,
                cov,
                t_tab=t_tab,
                scale=scale,
                wdm_psd_method=self.wdm_psd_method,
            )
        return self._component

    def covariance(self, settings):
        return self.component().covariance(settings)


def get_instrument_component():
    """Resolve ``INSTRUMENT_NOISE_MODEL`` into a fixed component (or ``None``).

    ``None`` means the sampled 2-parameter instrument model (a "psd" branch is
    created); a component means the instrument noise is fixed and only the
    enabled galfor/SGWB branches are sampled.
    """
    if INSTRUMENT_NOISE_MODEL == "sampled":
        return None
    if INSTRUMENT_NOISE_MODEL == "file":
        return FileNoiseEstimates(
            NOISE_ESTIMATE_FILE,
            time_dependent=NOISE_ESTIMATE_TIME_DEPENDENT,
            wdm_psd_method=NOISE_ESTIMATE_WDM_PSD_METHOD,
        )
    raise ValueError(
        f"INSTRUMENT_NOISE_MODEL must be 'sampled' or 'file', got {INSTRUMENT_NOISE_MODEL!r}."
    )


class GlassModulation:
    """Picklable callable ``t_arr -> (3, 3, Ntime)`` from a GLASS modulation file.

    The file columns are ``t, XX, YY, ZZ, XY, XZ, YZ``; the symmetric
    per-element matrix is interpolated onto the requested time grid. Loaded
    lazily on each call so the object pickles cleanly across MPI ranks.
    """

    def __init__(self, path: str):
        self.path = path

    def __call__(self, t_arr):
        glass = np.loadtxt(self.path)
        mod = np.array(
            [
                [glass[:, 1], glass[:, 4], glass[:, 5]],
                [glass[:, 4], glass[:, 2], glass[:, 6]],
                [glass[:, 5], glass[:, 6], glass[:, 3]],
            ]
        )
        return interp1d(glass[:, 0], mod)(np.asarray(asnumpy(t_arr)))


GALFOR_MODULATION = GlassModulation(MODULATION_FILE)


# ============================================================
# *** Synthetic data processor ***
# ============================================================


class SyntheticCompositeNoiseProcessor:
    """Duck-typed data "processor" that synthesizes WDM-domain noise.

    Implements the subset of the :class:`BaseProcessingStep` interface the
    engine consumes (``process``, ``pour``, ``td_signal``, ``catalogue``).
    ``process`` only establishes the time grid; ``pour`` builds the injected
    composite covariance on the active WDM grid and Cholesky-draws one noise
    realization per pixel (the same construction as noise_mcmc_validate.py).
    The TD window is ignored — the draw lives directly in the WDM basis.

    The instrument part of the injected covariance is either the sampled
    2-parameter model (``psd_injection``) or a fixed component such as
    :class:`FileNoiseEstimates` (``instrument_component``) — exactly one of
    the two must be given, mirroring the sensitivity backend used in the fit.
    """

    def __init__(
        self,
        Tobs: float,
        dt: float,
        psd_injection=None,
        galfor_injection=None,
        sgwb_injection=None,
        galfor_modulation=None,
        sgwb_stochastic_fn="PowerLawSGWB",
        instrument_component=None,
        tdi_generation: int = 2,
        seed: int = 0,
    ):
        self.Tobs = Tobs
        self.dt = dt
        # exactly one instrument-noise source: the 2-parameter injection or a
        # fixed component (e.g. FileNoiseEstimates)
        if (psd_injection is None) == (instrument_component is None):
            raise ValueError(
                "Provide exactly one of psd_injection (sampled instrument model) "
                "or instrument_component (fixed instrument noise)."
            )
        self.psd_injection = (
            None if psd_injection is None else np.asarray(psd_injection, dtype=float)
        )
        self.galfor_injection = (
            None if galfor_injection is None else np.asarray(galfor_injection, dtype=float)
        )
        self.sgwb_injection = (
            None if sgwb_injection is None else np.asarray(sgwb_injection, dtype=float)
        )
        self.galfor_modulation = galfor_modulation
        self.sgwb_stochastic_fn = sgwb_stochastic_fn
        self.instrument_component = instrument_component
        self.tdi_generation = tdi_generation
        self.seed = seed

        self.orbits = None  # engine falls back to the settings' orbits
        self.catalogue = dict(
            psd_injection=self.psd_injection,
            galfor_injection=self.galfor_injection,
            sgwb_injection=self.sgwb_injection,
            instrument_component=(
                None if instrument_component is None else repr(instrument_component.__dict__)
            ),
            seed=seed,
        )

    def process(self, **kwargs):
        """Establish the time grid; all filter/trim kwargs are ignored."""
        N = int(round(self.Tobs / self.dt))
        times = np.arange(N) * self.dt
        self.times = times
        self.td_signal = TDSignal(
            np.zeros((3, N)),
            TDSettings(t0=0.0, dt=self.dt, N=N, force_backend="cpu"),
        )
        return times, self.td_signal.arr

    def pour(self, settings, window=None, return_orbits=False):
        """Draw the WDM noise realization from the injected composite covariance."""
        if not isinstance(settings, WDMSettings):
            raise NotImplementedError(
                "SyntheticCompositeNoiseProcessor only pours into WDM domains."
            )
        backend = CompositeSensitivityBackend(
            settings,
            tdi_generation=self.tdi_generation,
            galfor_modulation=self.galfor_modulation,
            sgwb_stochastic_fn=self.sgwb_stochastic_fn,
            instrument_component=self.instrument_component,
        )
        sensmat = backend(
            "injection",
            self.psd_injection,
            galfor_params=self.galfor_injection,
            sgwb_params=self.sgwb_injection,
        )
        # sens_mat is on the active grid (3, 3, Nf_active, Nt_active) — the
        # same shape TDSignal.transform produces for poured data.
        C = asnumpy(sensmat.sens_mat)
        nf, nt = C.shape[2], C.shape[3]
        Cp = C.transpose(2, 3, 0, 1).reshape(-1, 3, 3)  # (Npix, 3, 3), pixel = f*nt + t
        L = np.linalg.cholesky(Cp)
        rng = np.random.default_rng(self.seed)
        z = rng.standard_normal((Cp.shape[0], 3, 1))
        draw = (L @ z)[:, :, 0].reshape(nf, nt, 3).transpose(2, 0, 1)  # (3, nf, nt)

        data_signal = WDMSignal(settings.xp.asarray(draw), settings)

        if return_orbits:
            return data_signal, self.orbits
        return data_signal


################

### DEFINE RECIPE

#############


class PSDSearchRecipeStep(RecipeStep):
    def setup_run(self, iteration, last_sample, sampler):
        # making sure
        sampler.moves = self.moves
        sampler.weights = self.weights

    def stopping_function(self, iteration, last_sample, sampler):
        # this will already be converged to max logl
        return True


class PSDPERecipeStep(RecipeStep):
    def setup_run(self, iteration, last_sample, sampler):
        # making sure
        sampler.moves = self.moves
        sampler.weights = self.weights

    def stopping_function(self, iteration, last_sample, sampler):
        # this will already be converged to max logl
        return False


def setup_recipe(recipe, engine_info, curr, acs, priors, state):

    general_info = curr.general_info
    nwalkers = curr.general_info.nwalkers
    ntemps = curr.general_info.ntemps

    gpus = curr.general_info.gpus
    if gpus is not None:
        cp.cuda.runtime.setDevice(gpus[0])

    # The PSDMove samples the psd + galfor + sgwb branches jointly, so the
    # tempering ladder dimension is the sum over the branches present.
    effective_ndim = sum(
        engine_info.ndims[key] for key in ("psd", "galfor", "sgwb") if key in engine_info.ndims
    )
    Tmax = 1e6
    temperature_control = TemperatureControl(
        effective_ndim, nwalkers, ntemps=ntemps, Tmax=Tmax, permute=False
    )

    psd_move_args = (acs, priors)

    # Smoke test: 2 repeats per outer iteration so the WDM Composite path
    # cycles in seconds rather than ~20 min on CPU. Bump back up to ~60
    # for a real run.
    psd_move_kwargs = dict(
        num_repeats=2,
        live_dangerously=True,
        temperature_control=temperature_control,
        sensitivity_backend=general_info.sensitivity_backend,
        # no psd branch when the instrument noise is fixed from file
        psd_transform_fn=(
            curr.source_info["psd"].transform_fn if "psd" in curr.source_info else None
        ),
    )

    psd_search_move = PSDMove(
        *psd_move_args,
        max_logl_mode=True,
        name="psd search move",
        **psd_move_kwargs,
    )

    psd_pe_move = PSDMove(
        *psd_move_args,
        max_logl_mode=False,
        name="psd pe move",
        **psd_move_kwargs,
    )
    # TODO: put this under the hood
    psd_search_move.accepted = np.zeros((ntemps, nwalkers))
    psd_pe_move.accepted = np.zeros((ntemps, nwalkers))

    recipe.add_recipe_component(PSDSearchRecipeStep(moves=[psd_search_move]), name="psd search")
    recipe.add_recipe_component(PSDPERecipeStep(moves=[psd_pe_move]), name="psd pe")


#######################
##### SETTINGS ###########
###############


def get_psd_erebor_settings(general_set: GeneralSetup) -> PSDSetup:

    initialize_kwargs_psd = dict()

    # 2-parameter (Soms_d, Sa_a) parameterisation consumed by
    # CompositeSensitivityBackend. Priors contain the injection.
    priors_psd = {
        r"$S_{\rm oms}$": uniform_dist(6.0e-12, 20.0e-11),
        r"$S_{\rm tm}$": uniform_dist(1.0e-15, 20.0e-14),
    }
    priors = {"psd": ProbDistContainer(priors_psd)}

    psd_settings = PSDSettings(
        ndim=2,
        injection=PSD_INJECTION,
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=initialize_kwargs_psd,
        priors=priors,
        log_dir=general_set.file_store_dir,
    )

    return PSDSetup(psd_settings)


def get_galfor_erebor_settings(general_set: GeneralSetup) -> GalForSetup:
    # Stock priors (amp, knee, alpha, slope1, slope2) contain the injection.
    galfor_settings = GalForSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs={},
        log_dir=general_set.file_store_dir,
    )

    return GalForSetup(galfor_settings)


def get_sgwb_erebor_settings(general_set: GeneralSetup) -> SGWBSetup:
    # Priors centered on the measurable amplitude range (see the
    # SGWB_INJECTION note above); the stock default (-22, -18) targets the
    # GLASS-style convention and would not contain this injection.
    priors_sgwb = {
        0: uniform_dist(-16.0, -9.0),  # log10_A (Omega_gw at 25 Hz)
        1: uniform_dist(-1.0, 2.0),  # alpha
    }
    priors = {"sgwb": ProbDistContainer(priors_sgwb)}

    sgwb_settings = SGWBSettings(
        injection=SGWB_INJECTION,
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs={},
        priors=priors,
        log_dir=general_set.file_store_dir,
    )

    return SGWBSetup(sgwb_settings)


# ============================================================
# *** Domain selection ***
# ============================================================
#
# WDM smoke-test grid. ``Tobs = NF * NT * DT`` keeps the data length an
# exact multiple of ``Nf * Nt`` so the WDM transform fits without padding.
# 768 * 1024 * 5 = 3,932,160 s ≈ 45.5 days. The active band matches the
# standalone noise validation scripts (3e-4 – 8e-3 Hz).
NF = 768
NT = 1024
DT = 5.0
TOBS = NF * NT * DT
DOMAIN_CHOICE = WDMSettings.make_factory(
    Nf=NF,
    Nt=NT,
    min_freq=3e-4,
    max_freq=8e-3,
)
# ============================================================


def get_general_erebor_settings() -> GeneralSetup:
    Tobs = TOBS
    dt = DT

    base_file_name = "noise_sgwb_smoke_test"
    file_store_dir = "./gf_output/"

    gpus = [0] if gpu_available else None
    if gpus is not None:
        cp.cuda.runtime.setDevice(gpus[0])
    # Small smoke-test config — CPU run, lightweight memory.
    nwalkers = 4
    ntemps = 2

    # The synthetic draw lives directly in the WDM basis; the TD window is
    # never applied, so use a rectangular (alpha = 0) window for consistency.
    window_taper_duration = 0.0

    orbits = EqualArmlengthOrbits()
    gpu_orbits = EqualArmlengthOrbits(force_backend=GPU_BACKEND)

    domain_settings = DOMAIN_CHOICE

    # None -> sampled 2-parameter instrument model; component -> fixed noise
    # estimates from the file (no psd branch). Galfor/SGWB injections follow
    # their own toggles independently.
    instrument_component = get_instrument_component()

    processor_init_kwargs = dict(
        Tobs=Tobs,
        dt=dt,
        psd_injection=PSD_INJECTION if instrument_component is None else None,
        galfor_injection=GALFOR_INJECTION if INCLUDE_GALFOR else None,
        sgwb_injection=SGWB_INJECTION if INCLUDE_SGWB else None,
        galfor_modulation=GALFOR_MODULATION,
        sgwb_stochastic_fn=SGWB_STOCHASTIC_FN,
        instrument_component=instrument_component,
        tdi_generation=2,
        seed=NOISE_SEED,
    )

    # The synthetic processor ignores filtering/trimming; explicit Nones keep
    # the engine's defaults from re-introducing a highpass + edge trim.
    preprocess_kwargs = dict(
        highpass_kwargs=None,
        trim_kwargs=None,
        Tobs=None,
        normalize=False,
    )

    # CompositeSensitivityBackend consumes these directly. The galfor
    # modulation is the same fixed GLASS modulation used in the injection;
    # the sampled foreground parameters scale the modulated template. A fixed
    # instrument component (file mode) replaces the sampled instrument model.
    sensitivity_init_kwargs = dict(
        tdi_generation=2,
        galfor_modulation=GALFOR_MODULATION,
        sgwb_stochastic_fn=SGWB_STOCHASTIC_FN,
        instrument_component=instrument_component,
    )

    # With a fixed instrument component there is no psd branch; these kwargs
    # are what run.py hands the sensitivity backend in that case (the sampled
    # galfor/sgwb branch coordinates are merged in per walker).
    fixed_psd_kwargs = (
        dict(psd_params=None) if instrument_component is not None else None
    )

    general_settings = GeneralSettings(
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        orbits=orbits,
        gpu_orbits=gpu_orbits,
        domain_settings=domain_settings,
        random_seed=103209,
        backup_iter=5,
        nwalkers=nwalkers,
        ntemps=ntemps,
        window_type="tukey",
        window_taper_duration=window_taper_duration,
        gpu_backend=GPU_BACKEND,
        gpus=gpus,
        data_processor=SyntheticCompositeNoiseProcessor,
        processor_init_kwargs=processor_init_kwargs,
        preprocess_kwargs=preprocess_kwargs,
        sensitivity_init_kwargs=sensitivity_init_kwargs,
        fixed_psd_kwargs=fixed_psd_kwargs,
    )

    general_setup = GeneralSetup(general_settings)
    return general_setup


def get_global_fit_settings(copy_settings_file=False):

    general_setup = get_general_erebor_settings()

    if copy_settings_file:
        shutil.copy(
            __file__,
            general_setup.file_store_dir
            + general_setup.base_file_name
            + "_"
            + __file__.split("/")[-1],
        )

    ###############################
    ######    Rank/GPU setup  #####
    ###############################

    head_rank = 1
    main_rank = 0

    rank_info = RankInfo(head_rank=head_rank, main_rank=main_rank)

    ##################################
    ###  Branch settings  ############
    ##################################

    # Branch lineup follows the model toggles: no psd branch when the
    # instrument noise is fixed from file; galfor / sgwb independently
    # enablable in either mode.
    source_info = {}
    if INSTRUMENT_NOISE_MODEL == "sampled":
        source_info["psd"] = get_psd_erebor_settings(general_setup)
    if INCLUDE_GALFOR:
        source_info["galfor"] = get_galfor_erebor_settings(general_setup)
    if INCLUDE_SGWB:
        source_info["sgwb"] = get_sgwb_erebor_settings(general_setup)
    if not source_info:
        raise ValueError(
            "No sampled branches: enable at least one of the sampled instrument "
            "model (INSTRUMENT_NOISE_MODEL='sampled'), INCLUDE_GALFOR, or INCLUDE_SGWB."
        )

    ##############
    ## READ OUT ##
    ##############

    global_settings = GlobalFitSettings(
        source_info=source_info,
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
    )

    curr_info = CurrentInfoGlobalFit(global_settings)

    return curr_info


if __name__ == "__main__":
    settings = get_global_fit_settings()
    breakpoint()
