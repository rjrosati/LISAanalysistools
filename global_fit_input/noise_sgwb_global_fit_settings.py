"""Global-fit settings: WDM-domain noise + galactic foreground + SGWB run.

Samples the instrument PSD (Soms_d, Sa_a), the 5-parameter hyperbolic-tangent
galactic foreground (with a fixed GLASS per-element time modulation), and a
2-parameter power-law SGWB, all through the WDM-domain
:class:`CompositeSensitivityBackend`.

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
from lisatools.sensitivity import CompositeSensitivityBackend
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
    """

    def __init__(
        self,
        Tobs: float,
        dt: float,
        psd_injection,
        galfor_injection=None,
        sgwb_injection=None,
        galfor_modulation=None,
        sgwb_stochastic_fn="PowerLawSGWB",
        tdi_generation: int = 2,
        seed: int = 0,
    ):
        self.Tobs = Tobs
        self.dt = dt
        self.psd_injection = np.asarray(psd_injection, dtype=float)
        self.galfor_injection = (
            None if galfor_injection is None else np.asarray(galfor_injection, dtype=float)
        )
        self.sgwb_injection = (
            None if sgwb_injection is None else np.asarray(sgwb_injection, dtype=float)
        )
        self.galfor_modulation = galfor_modulation
        self.sgwb_stochastic_fn = sgwb_stochastic_fn
        self.tdi_generation = tdi_generation
        self.seed = seed

        self.orbits = None  # engine falls back to the settings' orbits
        self.catalogue = dict(
            psd_injection=self.psd_injection,
            galfor_injection=self.galfor_injection,
            sgwb_injection=self.sgwb_injection,
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
        psd_transform_fn=curr.source_info["psd"].transform_fn,
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

    processor_init_kwargs = dict(
        Tobs=Tobs,
        dt=dt,
        psd_injection=PSD_INJECTION,
        galfor_injection=GALFOR_INJECTION,
        sgwb_injection=SGWB_INJECTION,
        galfor_modulation=GALFOR_MODULATION,
        sgwb_stochastic_fn=SGWB_STOCHASTIC_FN,
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
    # the sampled foreground parameters scale the modulated template.
    sensitivity_init_kwargs = dict(
        tdi_generation=2,
        galfor_modulation=GALFOR_MODULATION,
        sgwb_stochastic_fn=SGWB_STOCHASTIC_FN,
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

    psd_setup = get_psd_erebor_settings(general_setup)
    galfor_setup = get_galfor_erebor_settings(general_setup)
    sgwb_setup = get_sgwb_erebor_settings(general_setup)

    ##############
    ## READ OUT ##
    ##############

    global_settings = GlobalFitSettings(
        source_info={
            "psd": psd_setup,
            "galfor": galfor_setup,
            "sgwb": sgwb_setup,
        },
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
    )

    curr_info = CurrentInfoGlobalFit(global_settings)

    return curr_info


if __name__ == "__main__":
    settings = get_global_fit_settings()
    breakpoint()
