"""
run_task_complexity.py

C6: Task Complexity / KC Threshold Scaling (Reviewer 3, Point 8).
Retrain models with different numbers of odors (7, 14, 28) and a larger
synthetic odor set (56) to test whether KC thresholds scale with task demands.

Key question: ~47% of KC thresholds hit the upper boundary (-30 mV) in the
canonical 28-odor model. Does this fraction scale with the number of odors?
If so, the connectome has capacity for more complex discrimination tasks.

Each condition: 2-phase training (teacher 300ep -> student 300ep), 3 seeds (42-44).
After training, extract per-KC v_th distributions and compute boundary stats.

Results saved to: results/task_complexity_r6/
  results_r6_n{7,14,56}_s{42-44}.json
  canonical_thresholds.json   (n=28 baseline, seeds 42-44)
  task_complexity_summary.json

Notebook section: Section B -- C6 (task complexity table and KC bound fraction figure).

Pipeline context
----------------
This driver exercises the full connectome-constrained olfactory model:
    OR responses (Kreher 2008, 21 receptor channels)
      -> ORN (LIF) -> LN (LIF) -> PN (LIF)        [antennal lobe, AL]
      -> KC (two-compartment) <- APL (graded divisive inhibition)   [mushroom body, MB]
      -> linear decoder over the odor classes.
Connectivity is fixed by the Winding 2023 connectome; only ~449 biological
parameters (synaptic strengths, per-neuron thresholds, time constants, the KC
soma conductance, etc.) are learned. The experiment here probes how the
learned per-KC spike thresholds (v_th) respond to changing the decoding load
(number of odors), which is a proxy for task complexity.

The threshold (v_th) "upper bound" in this file means the *most excitable*
allowed setting, V_TH_MAX = -30 mV (a depolarized threshold is easiest to
reach), while the "lower bound" V_TH_MIN = -55 mV is the least excitable. A KC
sitting at the upper bound is one that gradient descent has pushed as far
toward "always fire" as the biological clamp permits -- i.e. the network wants
more KC activity than the bound allows, which is the saturation signature the
reviewer asked about.

Note on runtime constants: this driver mirrors the canonical pipeline and pins
N_STEPS = 30 and the sparsity loss offset/target (0.02 / 0.05) locally, which
match the canonical in-class defaults (n_steps=30 and the
class-level compute_loss). The canonical values below are the source of truth.
"""
import sys
from pathlib import Path
import argparse

# --- sys.path bootstrap ---------------------------------------------------
# This script may be launched from anywhere, so make the package's *parent*
# code/ on sys.path so direct subpackage imports (from core., from analysis.) resolve
# this repo). Path(__file__).parent.parent.parent climbs scripts/ -> repo ->
# repo's parent directory.
_pkg_parent = str(Path(__file__).parent.parent)
if _pkg_parent not in sys.path:
    sys.path.insert(0, _pkg_parent)

# Force UTF-8 + line buffering so progress logs (which contain unicode such as
# the degree/arrow glyphs) stream correctly when piped to a file or notebook.
if hasattr(sys.stdout, 'reconfigure'):  # absent under Jupyter's OutStream
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):  # absent under Jupyter's OutStream
    sys.stderr.reconfigure(encoding='utf-8')

import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Spiking student model + its parameter/noise dataclass.
from core.model import SpikingConnectomeConstrainedModel
from core.layers import SpikingParams
# Shared analysis metrics (representational decorrelation, Mancini divisive-norm
# test, and centroid/template-matching accuracy) reused across all experiments.
from analysis.compute import (
    compute_per_pair_decorrelation as _compute_per_pair_decorrelation,
    compute_mean_sim_decorrelation,
    run_mancini_test as _run_mancini_test,
    centroid_accuracy as _centroid_accuracy,
)
# Rate-based (non-spiking) teacher model used for ANN->SNN transfer.
from core.rate_model import ConnectomeConstrainedModel

# ============================================================================
# CONFIG  (mirrors run_ablation.py — C6 uses same training pipeline)
# ============================================================================
DATA_DIR = None  # resolved at runtime  # set in main() once the Kreher data dir is found
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'results' / 'task_complexity_r6'  # all C6 outputs land here
TEACHER_DIR = OUTPUT_DIR / 'teachers'  # cached teacher checkpoints (one per n_odors x seed)

TEACHER_EPOCHS = 300        # Phase-1 rate-teacher training length (epochs)
STUDENT_EPOCHS = 300        # Phase-2 spiking-student training length (epochs)
N_STEPS = 30                # CANONICAL simulation steps per forward pass (matches model.py's default)
MAX_SP_WEIGHT = 15.0        # Peak weight of the KC sparsity penalty after ramp-up
ENERGY_RAMP_EPOCHS = 60     # Epochs over which the sparsity weight ramps 0 -> MAX_SP_WEIGHT
BASE_LR = 1e-3              # Base Adam learning rate; per-group multipliers scale it (see get_param_groups)
GRAD_CLIP = 5.0             # Global grad-norm clip for the spiking student (stabilizes surrogate-grad training)

APL_BOOST = 4.0             # Multiply the teacher's APL gain when seeding the student (spiking APL needs stronger drive)
LN_VTH_INIT = -0.0475       # Initial LN spike threshold (-47.5 mV) used for LN neurons
LN_PN_SCALE = 1.2           # Scale-up factor for the inhibitory LN->PN synapse strength at init
ORN_PN_SCALE = 0.7          # Scale-down factor for the excitatory ORN->PN synapse strength at init
NONAD_INIT = np.log(1e-13)  # Log-space init (~0.1 pA) for "non-adapting" extra synapse pathways (near-zero start)

G_SOMA_MIN, G_SOMA_MAX_BIO = 1e-9, 20e-9  # KC soma-coupling conductance clamp (1–20 nS), the canonical g_soma range
KCKC_LOG_MIN = np.log(1e-15)  # Log-space floor (~1 fA) for KC<->KC synaptic strengths (effectively "off")
LOG_STRENGTH_MAX = np.log(1e-8)  # Log-space ceiling (10 nA) for synaptic strengths used in clamp_biological

NOISE_STD = 0.3             # Std of OR-response augmentation noise used to build the datasets/metrics
NOISE_TYPE = 'multiplicative'  # 'multiplicative' => noise scales the pattern; else additive


def _to_native(obj):
    """Recursively convert numpy/torch types to JSON-serializable Python builtins.

    C6 fix: json.dump(default=...) is bypassed by the C-extension for types
    like np.bool_ (post-NumPy 1.24). Recursively converting the whole dict
    before dump is the reliable alternative.

    Args:
        obj: Any value — typically a nested dict/list of numpy scalars,
            numpy arrays, or torch scalars produced by the metric functions.

    Returns:
        The same structure with every numpy/torch leaf replaced by a native
        Python ``bool``/``int``/``float``/``list``, so ``json.dump`` can
        serialize it without a custom encoder.
    """
    if isinstance(obj, dict):
        # Recurse into dict values, preserving keys.
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        # Tuples are flattened to lists (JSON has no tuple type).
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.bool_):
        # The exact type the C-extension mishandles; convert explicitly.
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        # Arrays -> nested lists of native floats/ints.
        return obj.tolist()
    if hasattr(obj, 'item'):  # torch scalars
        # 0-d torch tensors expose .item() -> a Python scalar.
        return obj.item()
    return obj  # already a native/serializable type

# Learning rate multipliers (same as run_ablation.py)
# Each multiplier scales BASE_LR for a group of parameters (see get_param_groups).
# Different subsystems need very different effective step sizes for stable training.
AL_LR = 0.2        # Antennal-lobe synapses/thresholds (slow — AL dynamics are sensitive)
NONAD_LR = 0.05    # Non-adapting auxiliary synapse pathways (very slow, start near zero)
KC_LR = 4.0        # KC/APL bulk params (fast — drive the sparse readout)
KCKC_LR = 0.1      # KC<->KC recurrent synapse strengths
GSOMA_LR = 0.1     # KC two-compartment soma conductance + STD time constants
APL_TAU_LR = 0.05  # APL inhibition time constant
KC_VTH_LR = 0.01   # Per-KC spike threshold (the C6 quantity of interest — learned slowly)
LN_VTH_LR = 0.01   # Per-LN spike threshold
ORN_VTH_LR = 0.01  # Per-ORN spike threshold
PN_VTH_LR = 0.01   # Per-PN spike threshold

# V_TH bounds (from layers.py — used for threshold analysis)
# These mirror the clamp range applied inside layers.py. Note: -30 mV is the
# UPPER bound = most excitable (easiest to spike); -55 mV is the LOWER bound =
# least excitable. The C6 measurement counts KCs pinned at these clamps.
V_TH_MIN = -0.055   # -55 mV
V_TH_MAX = -0.030   # -30 mV

# Noise configuration for the spiking student: enables all six biological noise
# sources used in the paper (membrane voltage, background current, synaptic
# release, threshold jitter, ORN receptor binding, plus the master switch).
# Values here are the "realistic" preset (somewhat above SpikingParams defaults).
REALISTIC_PARAMS = SpikingParams(
    v_noise_std=1.0e-3, i_noise_std=15e-12, syn_noise_std=0.25,
    threshold_jitter_std=1.0e-3, orn_receptor_noise_std=0.10,
    circuit_noise_enabled=True,
)

# R6 conditions: number of odors to train with
# 7/14 = subsets of the real Kreher odors, 28 = the canonical full set,
# 56 = 28 real + 28 synthetic odors (see generate_synthetic_odors).
ODOR_COUNTS = [7, 14, 28, 56]
SEEDS = [42, 43, 44]  # three RNG seeds for replication / error bars


# ============================================================================
# SPIKE ACCUMULATOR (same as run_ablation.py)
# ============================================================================
class SpikeAccumulator:
    """Captures ORN and LN spikes via forward hooks.

    Registers PyTorch forward hooks on the ORN and LN LIF layers so that, over
    a multi-step simulation, the per-timestep spike tensors are summed into
    running spike-count buffers without modifying the model's forward code.

    Attributes:
        orn_spikes: Running sum of ORN spikes, shape (batch, n_orn), or None
            before the first forward pass / after reset().
        ln_spikes: Running sum of LN spikes, shape (batch, n_ln), or None.
        _hooks: Handles for the registered hooks (so they can be removed).
    """
    def __init__(self):
        # Lazily initialized on the first hook firing; None means "nothing yet".
        self.orn_spikes = None
        self.ln_spikes = None
        self._hooks = []

    def register(self, model):
        """Attach forward hooks to the model's ORN and LN neuron layers.

        Args:
            model: A SpikingConnectomeConstrainedModel instance.

        Returns:
            self, so callers can write ``acc = SpikeAccumulator().register(m)``.

        Side effects:
            Appends two hook handles to ``self._hooks``.
        """
        def orn_hook(module, input, output):
            # The LIF layer returns (membrane_state, spikes); output[1] is the
            # spike tensor for this timestep. Accumulate across timesteps.
            spk = output[1]
            self.orn_spikes = spk if self.orn_spikes is None else self.orn_spikes + spk
        def ln_hook(module, input, output):
            # Same convention for the LN layer.
            spk = output[1]
            self.ln_spikes = spk if self.ln_spikes is None else self.ln_spikes + spk
        self._hooks.append(model.antennal_lobe.orn_neurons.register_forward_hook(orn_hook))
        self._hooks.append(model.antennal_lobe.ln_neurons.register_forward_hook(ln_hook))
        return self

    def reset(self):
        """Clear accumulated spike counts before a new (multi-step) forward pass."""
        self.orn_spikes = None
        self.ln_spikes = None

    def remove(self):
        """Detach all hooks (call once training/eval for this model is done)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ============================================================================
# DATASET: Repeated odor presentations with noise
# ============================================================================
class SubsetOdorDataset(Dataset):
    """OR responses for a subset of odors, with noise augmentation.

    C6: allows training with fewer (7, 14) or more (synthetic 56) odors
    while keeping the same training infrastructure.

    Each item is one noisy presentation of one odor's 21-channel OR response
    vector. With ``repeats_per_odor`` copies per odor, every __getitem__ draws
    fresh noise, so the dataset acts as an online augmentation generator.

    Args:
        or_responses: Tensor of clean OR responses, shape (n_odors, 21),
            values normalized to roughly [0, 1].
        repeats_per_odor: Number of (noisy) presentations per odor in one epoch.
        noise_std: Std of the augmentation noise.
        noise_type: 'multiplicative' (noise scales the pattern) or otherwise
            additive.
    """
    def __init__(self, or_responses, repeats_per_odor=10,
                 noise_std=0.3, noise_type='multiplicative'):
        self.or_responses = or_responses  # (n_odors, 21)
        self.n_odors = or_responses.shape[0]
        self.repeats = repeats_per_odor
        self.noise_std = noise_std
        self.noise_type = noise_type

    def __len__(self):
        # Total samples = odors x repeats; ordering is grouped by odor.
        return self.n_odors * self.repeats

    def __getitem__(self, idx):
        # Map the flat index back to an odor (block of `repeats` consecutive idx).
        odor_idx = idx // self.repeats
        pattern = self.or_responses[odor_idx]
        # Draw fresh Gaussian noise matching the pattern's shape/dtype/device.
        noise = torch.randn_like(pattern) * self.noise_std
        if self.noise_type == 'multiplicative':
            # Scale each channel by (1 + noise): models proportional receptor variability.
            noisy = pattern * (1.0 + noise)
        else:
            # Additive noise: a fixed-magnitude perturbation regardless of level.
            noisy = pattern + noise
        # OR responses are non-negative firing-rate proxies; clip away negatives.
        noisy = noisy.clamp(min=0)
        return noisy, odor_idx  # (features, integer odor label)


# ============================================================================
# SYNTHETIC ODOR GENERATION
# ============================================================================
def generate_synthetic_odors(real_or_responses, n_synthetic, seed):
    """Generate synthetic OR response patterns for larger odor sets.

    C6: Creates plausible odor patterns by sampling from the empirical
    distribution of each OR type (per-receptor mean/std from Kreher 2008),
    then ensuring patterns are distinct from each other and from real odors.

    The goal is to enlarge the discrimination task beyond the 28 measured odors
    (e.g. to 56) so KC-threshold scaling can be probed under heavier load,
    while keeping each synthetic odor statistically and structurally similar to
    a real one (matched per-receptor statistics + sparse activation).

    Args:
        real_or_responses: Tensor (n_real, n_or) of measured OR responses.
        n_synthetic: How many new odor patterns to generate.
        seed: Base RNG seed (offset internally so it does not collide with
            other seed-derived RNGs in this script).

    Returns:
        FloatTensor (n_synthetic, n_or) of synthetic OR patterns in [0, 1].
    """
    # Offset the seed so synthetic-odor RNG is independent of subset/train RNGs.
    rng = np.random.RandomState(seed + 5000)
    real_np = real_or_responses.numpy()
    n_real, n_or = real_np.shape

    # Per-OR-type statistics from Kreher 2008
    # Compute per-receptor mean/std across the real odors to match the marginal
    # distribution of each of the n_or receptor channels.
    or_means = real_np.mean(axis=0)
    or_stds = real_np.std(axis=0)
    or_stds = np.maximum(or_stds, 0.05)  # floor to avoid degenerate types  # keep variance for near-silent receptors

    # Sample from per-receptor Gaussian, clamp to [0, 1]
    synthetic = np.zeros((n_synthetic, n_or))
    for i in range(n_synthetic):
        # Draw a candidate pattern from the matched per-receptor Gaussian.
        pattern = rng.normal(or_means, or_stds)
        # Add structured sparsity: each odor activates 3-8 OR types strongly
        # (real odors drive only a handful of receptors strongly).
        n_active = rng.randint(3, 9)  # number of strongly-driven receptors (3..8 inclusive)
        active_mask = np.zeros(n_or, dtype=bool)
        # Pick which receptors are "active" without replacement.
        active_mask[rng.choice(n_or, n_active, replace=False)] = True
        # Suppress non-active receptors (reduce to background level)
        pattern[~active_mask] *= 0.2
        pattern = np.clip(pattern, 0, 1)  # valid OR-response range
        synthetic[i] = pattern

    return torch.from_numpy(synthetic).float()


# ============================================================================
# HELPERS (mirrored from run_ablation.py)
# ============================================================================
def clamp_biological(model):
    """Project all learnable params back into their biological ranges in-place.

    Called after every optimizer step so gradient descent cannot push
    parameters outside physiologically/numerically valid bounds. Because the
    relevant synaptic strengths and conductances are stored in LOG space (so
    they stay strictly positive and span orders of magnitude), the clamps here
    are applied to the log-parameters.

    Args:
        model: The spiking student model being trained.

    Side effects:
        Mutates the model's parameters under ``torch.no_grad()``.
    """
    # Layer-level clamps (per-neuron v_th, AL synapses, etc.) defined in layers.py.
    model.clamp_to_biological_bounds()
    with torch.no_grad():
        # KC<->KC axon-axon recurrent synapse strength (log nA): floor ~1 fA, ceil 10 nA.
        model.kc_layer.kc_kc_aa.log_strength.clamp_(KCKC_LOG_MIN, LOG_STRENGTH_MAX)
        # KC two-compartment soma coupling conductance (log S): 1–20 nS biological range.
        model.kc_layer.kc_neurons.log_g_soma.clamp_(np.log(G_SOMA_MIN), np.log(G_SOMA_MAX_BIO))
        if model.kc_layer.kc_kc_ad is not None:
            # KC<->KC axon-dendrite recurrent synapse (only if this pathway exists).
            model.kc_layer.kc_kc_ad.log_strength.clamp_(KCKC_LOG_MIN, LOG_STRENGTH_MAX)
        if model.kc_layer.pn_kc_nonad is not None:
            # PN->KC non-adapting (no short-term depression) synapse, if present.
            model.kc_layer.pn_kc_nonad.log_strength.clamp_(KCKC_LOG_MIN, LOG_STRENGTH_MAX)


def get_param_groups(model):
    """Build optimizer param groups (same LR schedule as run_ablation.py).

    Assigns each trainable parameter its own Adam param-group with a learning
    rate of ``BASE_LR * mult``, where ``mult`` is selected by matching the
    parameter's (dotted) name against the *_LR multiplier constants above. This
    lets fast-moving readout params (KC/APL) and slow, sensitive params (AL,
    per-neuron thresholds, auxiliary synapses) train at appropriate rates.

    Args:
        model: The spiking student model.

    Returns:
        list[dict]: One ``{'params': [param], 'lr': ...}`` dict per trainable
        parameter, ready to pass to ``torch.optim.Adam``.
    """
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen buffers / non-trainable params
        # The branches below are matched in order; the FIRST match wins, so the
        # ordering encodes precedence (e.g. STD time constants before generic AL).
        if 'log_tau_rec' in name or 'logit_U' in name:
            # Tsodyks-Markram short-term depression params (recovery tau, utilization U).
            mult = GSOMA_LR
        elif 'nonad' in name:
            mult = NONAD_LR
        elif 'kc_kc_aa' in name or 'kc_kc_ad' in name:
            mult = KCKC_LR
        elif 'kc_kc_dd_gain' in name or 'kc_kc_da_gain' in name:
            mult = KCKC_LR
        elif 'log_g_soma' in name:
            mult = GSOMA_LR
        elif 'log_g_gap' in name:
            # Electrical gap-junction conductances (LN-LN, PN-PN sister, eLN-PN).
            mult = AL_LR
        elif 'ln_pn_excit' in name:
            mult = AL_LR
        elif 'ln_ln' in name or 'pn_ln' in name:
            mult = AL_LR
        elif 'v_th' in name:
            # Per-neuron spike thresholds, learned slowly and sub-typed by neuron class.
            if 'kc' in name: mult = KC_VTH_LR
            elif 'ln' in name: mult = LN_VTH_LR
            elif 'orn' in name: mult = ORN_VTH_LR
            elif 'pn' in name: mult = PN_VTH_LR
            else: mult = 0.01  # fallback v_th rate
        elif 'or_to_orn' in name or 'or_gains' in name:
            # OR->ORN receptor gains.
            mult = 0.5
        elif 'apl_gain' in name:
            # APL inhibition gain — part of the fast KC/MB readout subsystem.
            mult = KC_LR
        elif 'orn_neurons' in name or 'ln_neurons' in name or 'pn_neurons' in name or 'antennal_lobe' in name:
            mult = AL_LR
        elif 'log_tau_apl' in name:
            mult = APL_TAU_LR
        elif 'kc_layer' in name or 'kc_neurons' in name or 'apl' in name:
            mult = KC_LR
        else:
            mult = 1.0  # default: full BASE_LR for anything unmatched (e.g. decoder)
        param_groups.append({'params': [param], 'lr': BASE_LR * mult})
    return param_groups


# ============================================================================
# KC THRESHOLD ANALYSIS  (C6-specific: the key measurement)
# ============================================================================
def analyze_kc_thresholds(model):
    """Extract KC threshold distribution and boundary statistics.

    C6: Measures how many KC thresholds hit V_TH_MAX (-30 mV), which
    indicates that gradient descent pushed them as excitable as possible.

    A large ``frac_at_upper_bound`` means the network is "starved" of KC
    activity at the imposed clamp — the central C6 observable for whether
    threshold demand scales with the number of odors.

    Args:
        model: A trained spiking model (canonical or C6 student).

    Returns:
        dict of summary stats (all in mV) plus the full per-KC threshold list:
            n_kc, v_th_{mean,std,min,max,median}_mV,
            frac/n_at_upper_bound, frac/n_at_lower_bound, v_th_values_mV.
    """
    # Per-KC learned thresholds are stored in volts; convert to mV for reporting.
    v_th = model.kc_layer.kc_neurons.v_th.detach().cpu().numpy() * 1e3  # convert to mV
    n_kc = len(v_th)

    # Threshold at boundary = within 0.5 mV of V_TH_MAX
    # Upper bound = most-excitable clamp (-30 mV); a 0.5 mV tolerance counts
    # near-saturated KCs (i.e. v_th >= -30.5 mV).
    at_upper = np.sum(v_th >= (V_TH_MAX * 1e3 - 0.5))  # -30.5 mV
    # Lower bound = least-excitable clamp (-55 mV); count v_th <= -54.5 mV.
    at_lower = np.sum(v_th <= (V_TH_MIN * 1e3 + 0.5))  # -54.5 mV

    return {
        'n_kc': int(n_kc),
        'v_th_mean_mV': float(np.mean(v_th)),
        'v_th_std_mV': float(np.std(v_th)),
        'v_th_min_mV': float(np.min(v_th)),
        'v_th_max_mV': float(np.max(v_th)),
        'v_th_median_mV': float(np.median(v_th)),
        'frac_at_upper_bound': float(at_upper / n_kc),   # key C6 metric: fraction pinned at -30 mV
        'n_at_upper_bound': int(at_upper),
        'frac_at_lower_bound': float(at_lower / n_kc),
        'n_at_lower_bound': int(at_lower),
        'v_th_values_mV': v_th.tolist(),  # full distribution for plotting
    }


# ============================================================================
# TRAINING
# ============================================================================
def train_teacher(seed, data_dir, n_odors, train_loader, test_loader):
    """Train rate-based teacher model (Phase 1).

    Phase 1 of the ANN->SNN transfer: train the non-spiking
    ConnectomeConstrainedModel (a rate model over the same connectome) to
    classify the odors. Its learned weights later seed the spiking student.
    Teachers are cached on disk and reloaded if present.

    Args:
        seed: RNG seed (controls weight init + data shuffling).
        data_dir: Path to the connectome/Kreher data directory.
        n_odors: Number of output classes for this condition.
        train_loader / test_loader: DataLoaders over noisy OR responses.

    Returns:
        The teacher's CPU ``state_dict`` (a dict of tensors), either freshly
        trained or loaded from cache.

    Side effects:
        Writes ``teacher_n{n_odors}_seed{seed}.pt`` under TEACHER_DIR.
    """
    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    teacher_path = TEACHER_DIR / f'teacher_n{n_odors}_seed{seed}.pt'

    if teacher_path.exists():
        # Cache hit: avoid re-running 300 epochs of teacher training.
        print(f"  Teacher n_odors={n_odors} seed={seed} exists, loading...")
        return torch.load(teacher_path, weights_only=False)

    print(f"--- Training teacher (n_odors={n_odors}, seed={seed}, {TEACHER_EPOCHS} ep) ---")
    # Seed both torch and numpy RNGs for reproducible teacher init/shuffling.
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Build the rate teacher from the connectome data; target_sparsity drives
    # the KC sparsity regularizer toward ~10% active KCs.
    teacher = ConnectomeConstrainedModel.from_data_dir(
        data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10)
    opt = torch.optim.Adam(teacher.parameters(), lr=1e-2)  # teacher trains at a flat higher LR
    for ep in range(TEACHER_EPOCHS):
        teacher.train()
        for bx, by in train_loader:
            opt.zero_grad()
            # Teacher's own combined loss (CE + sparsity); sparsity_weight is fixed here.
            loss, _ = teacher.compute_loss(bx, by, sparsity_weight=2.0)
            loss.backward()
            opt.step()
        if (ep + 1) % 100 == 0:
            # Periodic eval: report top-1 accuracy on the (noisy) test loader.
            teacher.eval()
            c, t = 0, 0
            with torch.no_grad():
                for bx, by in test_loader:
                    c += (teacher(bx).argmax(-1) == by).sum().item()
                    t += len(by)
            print(f"  Teacher ep {ep+1}: {c/t:.1%}")

    # Detach + move to CPU before caching so the file is device-independent.
    teacher_state = {k: v.cpu().clone() for k, v in teacher.state_dict().items()}
    torch.save(teacher_state, teacher_path)
    print(f"  Teacher saved: {teacher_path}")
    return teacher_state


def train_student(seed, data_dir, n_odors, or_responses,
                  train_loader, test_loader, teacher_state):
    """Train spiking student model (Phase 2) and analyze KC thresholds.

    C6: If a trained model file already exists, skips training and runs
    evaluation only. This avoids re-running 300 epochs when the model was
    saved but the JSON write failed (e.g. the np.bool_ serialization bug).

    Phase 2 of ANN->SNN transfer: initialize the spiking student from the
    rate teacher (decoder, OR gains, APL gain x APL_BOOST), apply a set of
    hand-tuned biological inits for the spiking-only machinery (thresholds,
    KC<->KC synapses, soma conductance, AL gap/inhibition, non-adapting
    pathways), then train with cross-entropy + a ramped KC sparsity penalty.
    Finally extract the per-KC threshold distribution (the C6 measurement) and
    a battery of accuracy/decorrelation/divisive-norm metrics.

    Args:
        seed: RNG seed.
        data_dir: Connectome/Kreher data directory.
        n_odors: Number of odor classes for this condition.
        or_responses: Clean OR responses for this condition, shape (n_odors, 21);
            used by the post-hoc metric functions.
        train_loader / test_loader: DataLoaders over noisy OR responses.
        teacher_state: The Phase-1 teacher's state_dict used to seed the student.

    Returns:
        dict of results (label, n_odors, seed, accuracy, centroid_accuracy,
        kc_sparsity, decorrelation, mancini, threshold_stats).

    Side effects:
        Writes ``model_{label}.pt`` (the trained student) and
        ``results_{label}.json`` under OUTPUT_DIR.
    """
    label = f'r6_n{n_odors}_s{seed}'
    model_path = OUTPUT_DIR / f'model_{label}.pt'
    results_path = OUTPUT_DIR / f'results_{label}.json'

    print(f"\n{'='*70}")
    print(f"STUDENT TRAINING: {label}")
    print(f"{'='*70}")

    # Reproducible student init/data order.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build student model (always needed, even for eval-only path)
    # include_nonad=True enables the extra non-adapting synapse pathways.
    student = SpikingConnectomeConstrainedModel.from_data_dir(
        data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10,
        params=REALISTIC_PARAMS, include_nonad=True)
    # Pin both AL and KC simulation lengths to the canonical N_STEPS (30 steps).
    student.n_steps_al = N_STEPS
    student.n_steps_kc = N_STEPS

    if model_path.exists():
        # C6: trained model exists — load it and skip the 300-epoch training loop
        print(f"  Trained model found for {label}, loading (skipping training)...")
        state = torch.load(model_path, weights_only=False, map_location='cpu')
        student.load_state_dict(state)
    else:
        # Full training path: initialize from teacher then run training loop
        teacher = ConnectomeConstrainedModel.from_data_dir(
            data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10)
        teacher.load_state_dict(teacher_state)

        # ---- Seed the spiking student from the teacher + biological inits ----
        with torch.no_grad():
            # Initialize every per-neuron spike threshold: LNs to LN_VTH_INIT
            # (-47.5 mV), all others to -42.5 mV (a moderately excitable start).
            for name, param in student.named_parameters():
                if 'v_th' in name:
                    param.fill_(LN_VTH_INIT if 'ln' in name else -0.0425)

            # Transfer the rate teacher's trained linear readout + receptor gains.
            student.decoder.weight.copy_(teacher.decoder.weight)
            student.decoder.bias.copy_(teacher.decoder.bias)
            student.or_to_orn.or_gains.copy_(teacher.or_to_orn.or_gains)
            # Spiking APL needs stronger inhibition than the rate model: x APL_BOOST.
            student.kc_layer.apl.apl_gain.data = teacher.kc_layer.apl.apl_gain.data.clone() * APL_BOOST

            # Nudge KC thresholds up 5 mV (less excitable) to keep early KC firing sparse.
            student.kc_layer.kc_neurons.v_th.data += 0.005
            # KC<->KC axon-axon synapse: small log-space init (~10 pA).
            student.kc_layer.kc_kc_aa.log_strength.fill_(np.log(1e-11))
            if student.kc_layer.kc_kc_ad is not None:
                # KC<->KC axon-dendrite: near-off init (~0.1 pA).
                student.kc_layer.kc_kc_ad.log_strength.fill_(np.log(1e-13))
            # KC soma coupling conductance: start at 10 nS (mid biological range).
            student.kc_layer.kc_neurons.log_g_soma.fill_(np.log(10e-9))

            # AL recurrent/feedback chemical synapses start essentially off (~0.1 pA),
            # so the AL begins near a feedforward regime and learns its recurrence.
            if student.antennal_lobe.ln_ln is not None:
                student.antennal_lobe.ln_ln.log_strength.fill_(np.log(1e-13))
            if student.antennal_lobe.pn_ln is not None:
                student.antennal_lobe.pn_ln.log_strength.fill_(np.log(1e-13))
            if student.antennal_lobe.ln_orn is not None:
                student.antennal_lobe.ln_orn.log_strength.fill_(np.log(1e-13))
            # Strengthen the inhibitory LN->PN synapse by LN_PN_SCALE (in log space => add log).
            ln_pn_orig = student.antennal_lobe.ln_pn.log_strength.item()
            student.antennal_lobe.ln_pn.log_strength.fill_(ln_pn_orig + np.log(LN_PN_SCALE))
            # Set the excitatory LN->PN component to 10% of the (boosted) inhibitory strength.
            ln_pn_inhib_strength = np.exp(student.antennal_lobe.ln_pn.log_strength.item())
            student.antennal_lobe.ln_pn_excit.log_strength.fill_(np.log(ln_pn_inhib_strength * 0.1))
            # Scale the excitatory ORN->PN synapse down by ORN_PN_SCALE (0.7).
            orn_pn_orig = student.antennal_lobe.orn_pn.log_strength.item()
            student.antennal_lobe.orn_pn.log_strength.fill_(orn_pn_orig + np.log(ORN_PN_SCALE))

            # All AL non-adapting (no short-term depression) pathways start near zero.
            for attr in ['orn_ln_nonad', 'ln_pn_nonad', 'ln_pn_excit_nonad',
                          'ln_ln_nonad', 'pn_ln_nonad', 'ln_orn_nonad']:
                layer = getattr(student.antennal_lobe, attr, None)
                if layer is not None:
                    layer.log_strength.fill_(NONAD_INIT)
            if student.kc_layer.pn_kc_nonad is not None:
                student.kc_layer.pn_kc_nonad.log_strength.fill_(NONAD_INIT)
        del teacher  # teacher no longer needed; free its memory

        # Hooks to track ORN/LN spiking during training (used in periodic eval).
        accumulator = SpikeAccumulator().register(student)
        param_groups = get_param_groups(student)
        optimizer = torch.optim.Adam(param_groups)

        # ---- TRAINING LOOP ----
        for epoch in range(STUDENT_EPOCHS):
            # Ramp the sparsity-penalty weight linearly from 0 to MAX_SP_WEIGHT
            # over ENERGY_RAMP_EPOCHS, so the net first learns to classify and
            # only later is pressured toward sparse, energy-efficient KC codes.
            progress = min(1.0, epoch / ENERGY_RAMP_EPOCHS)
            sp_w = progress * MAX_SP_WEIGHT

            student.train()
            train_correct, train_total = 0, 0
            for bx, by in train_loader:
                optimizer.zero_grad()
                accumulator.reset()
                # Forward pass through the full spiking pipeline; return_all gives
                # intermediate info (e.g. summed KC spike counts) needed for loss/metrics.
                logits, info = student(bx, return_all=True)
                ce_loss = F.cross_entropy(logits, by)

                # KC sparsity loss (same as canonical)
                # Per-KC mean firing rate over the N_STEPS window (spikes/step).
                kc_rates = info['kc_spikes'] / N_STEPS
                # Soft-count the fraction of KCs above ~0.02 rate (sigmoid gate at
                # offset 0.02, slope 50), penalize squared deviation from target 0.05.
                # These offset/target constants are the CANONICAL values.
                sp_loss = (torch.sigmoid((kc_rates - 0.02) * 50).mean() - 0.05) ** 2
                total_loss = ce_loss + sp_w * sp_loss

                total_loss.backward()
                # Clip global grad norm to GRAD_CLIP for stable surrogate-grad training.
                torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
                optimizer.step()
                clamp_biological(student)  # re-project params into biological bounds after each step
                train_correct += (logits.argmax(-1) == by).sum().item()
                train_total += len(by)

            if (epoch + 1) % 50 == 0:
                # Periodic eval: train/test accuracy, KC activity, AL/MB decorrelation, Mancini.
                # (KC-threshold stats are computed once in the final evaluation and written
                # to the results JSON; the live line uses the shared 6-field format.)
                student.eval()
                tc, tt = 0, 0
                all_kc = []
                with torch.no_grad():
                    for bx, by in test_loader:
                        accumulator.reset()
                        logits, info = student(bx, return_all=True)
                        tc += (logits.argmax(-1) == by).sum().item()
                        tt += len(by)
                        # Fraction of KCs that spiked at least once this trial.
                        all_kc.append((info['kc_spikes'] > 0).float().mean().item())
                decorr = compute_mean_sim_decorrelation(student, or_responses)
                manc = _run_mancini_test(student)
                # Sp = KC active fraction (the KC sparsity proxy used across all runs).
                print(f"  Ep {epoch+1}: Train={train_correct/train_total:.1%}, Test={tc/tt:.1%}, "
                      f"Sp={np.mean(all_kc):.1%}, AL={decorr['al_decorr']:.1f}%, MB={decorr['mb_decorr']:.1f}%, "
                      f"Manc={manc['ratio']:.2f}")

        accumulator.remove()  # detach hooks once training is finished
        # Save model after training completes (not needed in eval-only path)
        torch.save(student.state_dict(), model_path)

    # ---- FINAL EVALUATION ---- (runs for both trained and loaded models)
    student.eval()

    # KC threshold analysis (C6 key measurement)
    threshold_stats = analyze_kc_thresholds(student)

    # Classification accuracy
    # Linear-decoder top-1 accuracy + mean KC activity fraction on the test set.
    tc, tt = 0, 0
    all_kc = []
    with torch.no_grad():
        for bx, by in test_loader:
            logits, info = student(bx, return_all=True)
            tc += (logits.argmax(-1) == by).sum().item()
            tt += len(by)
            all_kc.append((info['kc_spikes'] > 0).float().mean().item())
    test_acc = tc / tt
    kc_sparsity = float(np.mean(all_kc))

    # Decorrelation
    # Per-odor-pair representational decorrelation at the AL and MB stages
    # (how much the circuit separates odor representations vs. their inputs).
    pp = _compute_per_pair_decorrelation(student, or_responses, 10, NOISE_STD)

    # Centroid accuracy
    # Template/centroid-matching accuracy: assign each noisy trial to the nearest
    # per-odor mean representation (a decoder-free readout sanity check).
    cent_acc = _centroid_accuracy(student, or_responses, 20, NOISE_STD)

    # Mancini test
    # Tests whether the APL inhibition behaves like divisive (shunting)
    # normalization; returns a ratio and a boolean pass flag.
    manc = _run_mancini_test(student)

    print(f"\n  --- FINAL: {label} ---")
    print(f"  Acc:  linear={test_acc:.1%}, centroid={cent_acc:.1%}")
    print(f"  KC:   sparsity={kc_sparsity:.1%}")
    print(f"  Dec:  AL={pp['al_decorr']:.1f}%, MB={pp['mb_decorr']:.1f}%")
    print(f"  Manc: {manc['ratio']:.2f}")
    print(f"  v_th: mean={threshold_stats['v_th_mean_mV']:.1f}mV, "
          f"at_upper={threshold_stats['frac_at_upper_bound']:.0%} "
          f"({threshold_stats['n_at_upper_bound']}/{threshold_stats['n_kc']})")

    # Assemble the per-run result record (threshold_stats holds the C6 payload).
    results = {
        'label': label,
        'n_odors': n_odors,
        'seed': seed,
        'accuracy': test_acc,
        'centroid_accuracy': cent_acc,
        'kc_sparsity': kc_sparsity,
        'decorrelation': {
            'al': pp['al_decorr'], 'mb': pp['mb_decorr'], 'total': pp['total_decorr'],
        },
        'mancini': {'ratio': float(manc['ratio']), 'passes': bool(manc['passes'])},
        'threshold_stats': threshold_stats,
    }

    # R6 fix: use _to_native() to pre-convert the whole dict before json.dump.
    # json's C-extension bypasses default= for np.bool_ (post-NumPy 1.24),
    # so recursive pre-conversion is the only reliable approach.
    with open(results_path, 'w') as f:
        json.dump(_to_native(results), f, indent=2)
    print(f"  Saved: {results_path}")
    return results


# ============================================================================
# CANONICAL THRESHOLD EXTRACTION (for the 28-odor baseline)
# ============================================================================
def extract_canonical_thresholds(data_dir, seeds):
    """Extract KC thresholds from already-trained canonical models.

    Loads the paper's canonical 28-odor spiking models (one per seed) from
    results/all_connections_nonad_canonical/ and runs analyze_kc_thresholds on
    each, providing the n=28 baseline that the C6 conditions are compared
    against. No training happens here.

    Args:
        data_dir: Connectome/Kreher data directory (needed to rebuild the model
            skeleton before loading the checkpoint).
        seeds: Iterable of seeds whose canonical checkpoints to load.

    Returns:
        list[dict]: One threshold-stats dict per available seed (with an added
        'seed' key). Missing checkpoints are skipped.
    """
    canonical_dir = Path(__file__).parent.parent.parent / 'results' / 'all_connections_nonad_canonical'
    results = []
    for seed in seeds:
        model_path = canonical_dir / f'model_seed{seed}.pt'
        if not model_path.exists():
            print(f"  Canonical model seed {seed} not found, skipping")
            continue
        # Rebuild the canonical architecture (28 odors) then load its weights.
        model = SpikingConnectomeConstrainedModel.from_data_dir(
            data_dir, n_odors=28, n_or_types=21, target_sparsity=0.10,
            params=REALISTIC_PARAMS, include_nonad=True)
        state = torch.load(model_path, weights_only=False, map_location='cpu')
        model.load_state_dict(state)
        model.eval()
        th = analyze_kc_thresholds(model)
        th['seed'] = seed
        results.append(th)
        del model  # free memory before the next seed's model
        print(f"  Canonical seed {seed}: vth_mean={th['v_th_mean_mV']:.1f}mV, "
              f"at_upper={th['frac_at_upper_bound']:.0%} "
              f"({th['n_at_upper_bound']}/{th['n_kc']})")
    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Entry point: run the C6 task-complexity / KC-threshold-scaling experiment.

    Resolves the data directory, loads the Kreher 2008 OR responses, extracts
    the canonical 28-odor threshold baseline, then for each requested odor
    count and seed builds the dataset (real subset or real+synthetic), trains a
    teacher and a student, and records KC-threshold + accuracy metrics. Finally
    prints a summary table and writes the aggregate JSON.

    CLI flags:
        --n-odors: run a single odor count (default: all of ODOR_COUNTS).
        --seed: run a single seed (default: all of SEEDS).
        --canonical-only: only extract canonical thresholds, then return.
    """
    parser = argparse.ArgumentParser(description='C6: Task complexity / KC threshold scaling')
    parser.add_argument('--n-odors', type=int, default=None,
                        help='Train a single odor count (7, 14, 28, or 56)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Single seed (default: run all 3)')
    parser.add_argument('--canonical-only', action='store_true',
                        help='Only extract thresholds from canonical 28-odor models')
    args = parser.parse_args()

    # Resolve data directory
    # Search several likely locations for the connectome data; the canonical
    # marker is a 'kreher2008' subfolder. _parent is the grandparent of scripts/.
    _parent = Path(__file__).resolve().parent.parent.parent
    _data_candidates = [
        Path(__file__).resolve().parent.parent.parent / 'data',
    ]
    global DATA_DIR
    DATA_DIR = next((p for p in _data_candidates if (p / 'kreher2008').is_dir()), None)
    if DATA_DIR is None:
        raise FileNotFoundError('Cannot find connectome data.')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Kreher 2008 OR responses (all 28 odors)
    # Prefer the pre-saved tensor; fall back to the CSV (index column = odor name).
    kreher_dir = DATA_DIR / 'kreher2008'
    pt_path = kreher_dir / 'orn_responses_normalized.pt'
    if pt_path.exists():
        all_or_responses = torch.load(pt_path, weights_only=True)
    else:
        df = pd.read_csv(kreher_dir / 'orn_responses_normalized.csv', index_col=0)
        all_or_responses = torch.from_numpy(df.values).float()

    # Extract canonical model thresholds as baseline
    print("="*70)
    print("EXTRACTING CANONICAL (28-odor) KC THRESHOLDS")
    print("="*70)
    canonical_th = extract_canonical_thresholds(DATA_DIR, SEEDS)
    if canonical_th:
        # Persist the baseline and report the average upper-bound saturation.
        canon_path = OUTPUT_DIR / 'canonical_thresholds.json'
        with open(canon_path, 'w') as f:
            json.dump(canonical_th, f, indent=2)
        avg_upper = np.mean([t['frac_at_upper_bound'] for t in canonical_th])
        print(f"\n  Canonical avg at upper bound: {avg_upper:.1%}")

    if args.canonical_only:
        return  # baseline-only mode: stop before any training

    # Determine which conditions to run
    # If a CLI flag was given, restrict to that single value; otherwise sweep all.
    odor_counts = [args.n_odors] if args.n_odors else ODOR_COUNTS
    seeds = [args.seed] if args.seed else SEEDS

    all_results = []  # collects every per-(n_odors, seed) result dict

    for n_odors in odor_counts:
        print(f"\n{'='*70}")
        print(f"CONDITION: {n_odors} ODORS")
        print(f"{'='*70}")

        # Prepare OR responses for this condition
        if n_odors <= 28:
            # C6: Subset of real Kreher odors (deterministic selection per seed)
            for seed in seeds:
                # Seed-offset RNG so the subset choice is reproducible but
                # differs from the train/synthetic RNG streams.
                rng = np.random.RandomState(seed + 2000)
                if n_odors < 28:
                    # Pick n_odors distinct real-odor indices (sorted for stable logging).
                    indices = sorted(rng.choice(28, n_odors, replace=False))
                else:
                    # n_odors == 28: use the full canonical odor set.
                    indices = list(range(28))
                or_subset = all_or_responses[indices]
                print(f"\n  Seed {seed}: using odor indices {indices}")

                # Create dataloaders for this subset
                # 10 noisy repeats/odor for training, 5 for testing.
                train_ds = SubsetOdorDataset(or_subset, repeats_per_odor=10,
                                             noise_std=NOISE_STD, noise_type=NOISE_TYPE)
                test_ds = SubsetOdorDataset(or_subset, repeats_per_odor=5,
                                            noise_std=NOISE_STD, noise_type=NOISE_TYPE)
                train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
                test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

                # Train teacher
                teacher_state = train_teacher(seed, DATA_DIR, n_odors, train_loader, test_loader)

                # Train student and analyze thresholds
                result = train_student(seed, DATA_DIR, n_odors, or_subset,
                                       train_loader, test_loader, teacher_state)
                all_results.append(result)

        else:
            # C6: Synthetic odor set (larger than 28)
            for seed in seeds:
                # Top up the 28 real odors with (n_odors - 28) synthetic ones.
                n_synthetic = n_odors - 28
                synthetic = generate_synthetic_odors(all_or_responses, n_synthetic, seed)
                or_combined = torch.cat([all_or_responses, synthetic], dim=0)
                print(f"\n  Seed {seed}: 28 real + {n_synthetic} synthetic = {n_odors} odors")

                train_ds = SubsetOdorDataset(or_combined, repeats_per_odor=10,
                                             noise_std=NOISE_STD, noise_type=NOISE_TYPE)
                test_ds = SubsetOdorDataset(or_combined, repeats_per_odor=5,
                                            noise_std=NOISE_STD, noise_type=NOISE_TYPE)
                train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
                test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

                teacher_state = train_teacher(seed, DATA_DIR, n_odors, train_loader, test_loader)
                result = train_student(seed, DATA_DIR, n_odors, or_combined,
                                       train_loader, test_loader, teacher_state)
                all_results.append(result)

    # ---- SUMMARY ----
    # Print a per-run table then per-n_odors averages of the key metrics.
    print(f"\n{'='*80}")
    print("C6 TASK COMPLEXITY SUMMARY: KC THRESHOLD SCALING")
    print(f"{'='*80}")
    print(f"\n  {'n_odors':>8} {'seed':>5} {'Acc':>6} {'Cent':>6} {'KC%':>6} "
          f"{'MB dec':>7} {'vth_mean':>9} {'at_upper':>9}")

    for r in all_results:
        th = r['threshold_stats']
        print(f"  {r['n_odors']:>8} {r['seed']:>5} {r['accuracy']:>5.1%} "
              f"{r['centroid_accuracy']:>5.1%} {r['kc_sparsity']:>5.1%} "
              f"{r['decorrelation']['mb']:>+6.1f}% {th['v_th_mean_mV']:>8.1f}mV "
              f"{th['frac_at_upper_bound']:>8.0%}")

    # Aggregate by n_odors
    # The headline C6 trend: does at_upper (and v_th) shift with odor count?
    print(f"\n  --- Averages ---")
    for n in sorted(set(r['n_odors'] for r in all_results)):
        subset = [r for r in all_results if r['n_odors'] == n]
        avg_acc = np.mean([r['centroid_accuracy'] for r in subset])
        avg_kc = np.mean([r['kc_sparsity'] for r in subset])
        avg_mb = np.mean([r['decorrelation']['mb'] for r in subset])
        avg_upper = np.mean([r['threshold_stats']['frac_at_upper_bound'] for r in subset])
        avg_vth = np.mean([r['threshold_stats']['v_th_mean_mV'] for r in subset])
        print(f"  n={n:>3}: centroid={avg_acc:.1%}, KC={avg_kc:.1%}, "
              f"MB_dec={avg_mb:+.1f}%, vth={avg_vth:.1f}mV, at_upper={avg_upper:.0%}")

    # Save combined results (without v_th_values_mV for compactness)
    # Deep-copy each record via json round-trip, then drop the large per-KC
    # threshold list so the summary file stays small.
    summary_results = []
    for r in all_results:
        r_copy = json.loads(json.dumps(r))
        r_copy['threshold_stats'].pop('v_th_values_mV', None)
        summary_results.append(r_copy)
    summary_path = OUTPUT_DIR / 'task_complexity_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({'results': summary_results, 'canonical_thresholds': canonical_th}, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == '__main__':
    main()
