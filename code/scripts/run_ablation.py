"""
run_ablation.py

Unified ablation training script for CCN 2026 revisions.
Handles C1 (component ablations), C3 (LN threshold), C4 (sparsity ablations),
and APL-boosted energy experiments — all via command-line flags.

All conditions share the same training infrastructure. Teacher is trained once
per seed and cached to disk.

Examples:
    # Train teacher for a seed
    python scripts/run_ablation.py --train-teacher --seed 42

    # C1(i): No gap junctions
    python scripts/run_ablation.py --no-gap --kc-sparsity --seed 42 --label r3i_no_gap_s42

    # C1(ii) / C4(ii): No APL + KC sparsity
    python scripts/run_ablation.py --no-apl --kc-sparsity --seed 42 --label r3ii_no_apl_s42

    # C1(iii): Shuffled connectome
    python scripts/run_ablation.py --shuffle --kc-sparsity --seed 42 --label r3iii_shuffle_s42

    # R4: Different LN fan-out quantile
    python scripts/run_ablation.py --kc-sparsity --ln-quantile 0.50 --seed 42 --label r4_q050_s42

    # APL-boosted energy
    python scripts/run_ablation.py --energy-weight 15 --apl-boost 8 --seed 42 --label e15_apl8_s42

Results saved to: results/ablations_r1/

Notebook sections: Section B — C1 (retrained ablations),
                   Section B — C3 (LN threshold variants),
                   Section B — C4 (c4i_no_sp sparsity ablation).

------------------------------------------------------------------------------
READER'S OVERVIEW (added documentation — not part of the original behaviour):

This file is a *merged* driver. It contains THREE independent CLI entry points,
dispatched by ``main()`` based on the first positional argv token:

  * (no subcommand) -> ``_main_trained()``  — the canonical ablation trainer
        (ANN rate "teacher" distilled into a spiking "student"; ablations such
        as --no-gap / --no-apl / --shuffle / --ln-quantile are applied to the
        student before training).
  * ``posthoc``     -> ``_main_posthoc()``  — eval-only post-hoc ablations
        (load already-trained canonical checkpoints, zero a component, re-score;
        no retraining). Merged from the former run_posthoc_ablation.py.
  * ``std``         -> ``_main_std()``       — Tsodyks-Markram short-term
        depression (STD) ablation, both "train from scratch with STD off" and
        "load canonical and evaluate with STD off". Merged from the former
        run_std_ablation.py.

PIPELINE RECAP (where this fits): OR responses (Kreher 2008) -> ORN (LIF) ->
LN (LIF) -> PN (LIF) -> KC (two-compartment) <- APL (graded divisive inhibition)
-> linear decoder over 28 odors. Connectivity is FIXED by the Winding 2023
connectome; only ~449 biological parameters are learned. Biophysical parameters
that must stay positive (synapse strengths, gap-junction and soma conductances,
time constants) are stored in LOG space and recovered with exp()/softplus() so
gradient descent can move them freely while the recovered value stays > 0.

CANONICAL RUNTIME CONSTANTS (deliberately duplicated across the run_*.py
drivers): N_STEPS = 30 simulation steps; KC-sparsity loss offset 0.02 / target
0.05; g_soma clamp [1, 20] nS. model.py's in-class compute_loss / n_steps=20
defaults are LEGACY and are overridden here.

NOTE ON TWO UNDEFINED GLOBALS: ``SEEDS`` and ``APL_BOOST`` are *referenced* in
the merged ``posthoc`` and ``std`` code paths (and ``run_no_std_training``) but
are NOT assigned at module scope anywhere in this file — they were module-level
globals in the former standalone scripts that did not survive the merge. The
canonical ``_main_trained()`` path does not touch them, so it is unaffected;
the ``posthoc`` / ``std`` paths will raise NameError unless ``SEEDS`` is first
created (e.g. ``_main_std`` only assigns it via ``global SEEDS`` when
``--seed`` is passed). FIXED: ``SEEDS`` and ``APL_BOOST`` are now defined at
module scope below; a ``--seed`` override still rebinds ``SEEDS`` to one seed.
"""
import sys
from pathlib import Path
import argparse

# --- sys.path bootstrap so the sibling packages import regardless of CWD -----
# The package this script lives in is two levels up (.../<pkg>/scripts/this.py),
# so its *grandparent* directory is where the top-level packages
# code/ (which holds core/, analysis/, scripts/) is now on sys.path.
_pkg_parent = str(Path(__file__).parent.parent)
if _pkg_parent not in sys.path:
    sys.path.insert(0, _pkg_parent)

# Force UTF-8 + line-buffered stdout so the ✓/→ glyphs and live progress lines
# render correctly and flush immediately when piped to a log file.
if hasattr(sys.stdout, 'reconfigure'):  # absent under Jupyter's OutStream
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):  # absent under Jupyter's OutStream
    sys.stderr.reconfigure(encoding='utf-8')

import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

# Spiking "student" model + its biophysical noise parameter container.
from core.model import SpikingConnectomeConstrainedModel
from core.layers import SpikingParams
# Shared analysis routines (decorrelation, Mancini APL test, concentration
# invariance, centroid classifier). Imported with private aliases so the thin
# wrapper functions defined later in THIS file can reuse the names.
from analysis.compute import (
    compute_per_pair_decorrelation as _compute_per_pair_decorrelation,
    compute_mean_sim_decorrelation, run_mancini_test as _run_mancini_test,
    run_concentration_invariance as _run_concentration_invariance,
    centroid_accuracy as _centroid_accuracy,
)
# Rate-based ANN teacher.
from core.rate_model import ConnectomeConstrainedModel
from core.dataset import load_kreher2008_all_odors, create_dataloaders

# ============================================================================
# SPIKE ACCUMULATOR
# ============================================================================
class SpikeAccumulator:
    """Forward-hook helper that sums ORN and LN spikes across the simulation.

    The ORN and LN LIF layers return ``(membrane_state, spikes)`` tuples per
    time step, but the spiking model integrates over N_STEPS internally and only
    exposes PN/KC spike *counts* in its ``info`` dict. To recover ORN/LN
    population activity for the energy loss and the sparsity report, we register
    forward hooks on those two layers and accumulate ``output[1]`` (the binary
    spike tensor, shape [batch, n_neurons]) every time the layer fires.

    Side effects: holds live PyTorch hook handles; call :meth:`remove` to detach.
    """
    def __init__(self):
        # Running sums of spikes (None until the first forward pass populates
        # them). After a full forward these hold [batch, n_orn] / [batch, n_ln]
        # integer-valued spike counts summed over the N_STEPS inner time steps.
        self.orn_spikes = None
        self.ln_spikes = None
        self._hooks = []  # registered hook handles, removed in remove()

    def register(self, model):
        """Attach forward hooks to the model's ORN and LN neuron layers.

        Args:
            model: a SpikingConnectomeConstrainedModel whose antennal_lobe
                exposes ``orn_neurons`` and ``ln_neurons`` LIF modules.

        Returns:
            self, so callers can write ``acc = SpikeAccumulator().register(m)``.
        """
        def orn_hook(module, input, output):
            # output is (state, spikes); index [1] is the spike tensor. Sum it
            # into the running ORN total (initialise on first call).
            spk = output[1]
            self.orn_spikes = spk if self.orn_spikes is None else self.orn_spikes + spk
        def ln_hook(module, input, output):
            # Same accumulation for the LN layer.
            spk = output[1]
            self.ln_spikes = spk if self.ln_spikes is None else self.ln_spikes + spk
        self._hooks.append(model.antennal_lobe.orn_neurons.register_forward_hook(orn_hook))
        self._hooks.append(model.antennal_lobe.ln_neurons.register_forward_hook(ln_hook))
        return self

    def reset(self):
        """Clear the accumulated spike sums before each new forward pass."""
        self.orn_spikes = None
        self.ln_spikes = None

    def remove(self):
        """Detach all hooks (must be called to avoid leaking handles)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

# ============================================================================
# CONFIGURATION
# ============================================================================
# Biological clamp range for the KC soma->dendrite coupling conductance g_soma:
# 1 nS .. 20 nS (stored in SI Siemens). Keeps the learned two-compartment
# coupling within a physiologically plausible band.
G_SOMA_MIN, G_SOMA_MAX_BIO = 1e-9, 20e-9
# Lower clamp for KC<->KC synapse log-strength: log(1e-15) ~ effectively-off
# (an "almost zero" floor so the strength can shrink toward but never reach 0).
KCKC_LOG_MIN = np.log(1e-15)
# Upper clamp for chemical synapse log-strength: log(1e-8 S) = strongest allowed.
LOG_STRENGTH_MAX = np.log(1e-8)

APL_BOOST_DEFAULT = 4.0   # default multiplier applied to the teacher's APL gain
# Module-scope globals restored after the script merge (see file header). The
# posthoc/std/run_no_std_training paths iterate these internally; a --seed CLI
# override rebinds SEEDS to a single seed. Without these, `run_ablation.py posthoc`
# and `... std` (called with no --seed) raise NameError.
SEEDS = [42, 43, 44, 45, 46]    # canonical 5-seed ensemble
APL_BOOST = APL_BOOST_DEFAULT   # teacher->student APL-gain multiplier (run_no_std_training)
TEACHER_EPOCHS = 300      # ANN teacher training epochs (per seed)
STUDENT_EPOCHS = 300      # spiking student training epochs
MAX_SP_WEIGHT = 15.0      # peak weight of the KC sparsity loss after ramp-up
BASE_LR = 1e-3            # base Adam learning rate; per-group multipliers below
KC_VTH_LR = 0.01          # LR multiplier for KC firing thresholds
LN_VTH_LR = 0.01          # LR multiplier for LN firing thresholds
ORN_VTH_LR = 0.01         # LR multiplier for ORN firing thresholds
PN_VTH_LR = 0.01          # LR multiplier for PN firing thresholds
GRAD_CLIP = 5.0           # global gradient-norm clip (surrogate grads can spike)

LN_VTH_INIT = -0.0475     # initial LN spike threshold V_th (volts; ~-47.5 mV)
LN_PN_SCALE = 1.2         # multiply initial LN->PN inhibitory strength by this
ORN_PN_SCALE = 0.7        # multiply initial ORN->PN excitatory strength by this
N_STEPS = 30              # canonical sim steps (matches model.py's default)

# Per-parameter-group learning-rate multipliers (applied on top of BASE_LR).
# These reflect how sensitive each biological knob is: AL synapses move slowly,
# KC/APL knobs need a large LR to escape flat regions of the surrogate loss.
AL_LR = 0.2               # antennal-lobe synapses / gap junctions
NONAD_LR = 0.05           # non-adult ("nonad") connectome pathways
KC_LR = 4.0              # KC layer + APL gain (needs aggressive steps)
KCKC_LR = 0.1             # KC<->KC recurrent synapses
GSOMA_LR = 0.1            # g_soma, STD recovery tau, STD release-prob U
APL_TAU_LR = 0.05         # APL inhibition time constant

NOISE_TYPE = 'multiplicative'  # OR-response trial noise model (CV-style, 30%)
NOISE_STD = 0.3                # noise magnitude (coefficient of variation = 0.3)
NONAD_INIT = np.log(1e-13)     # init log-strength for nonad pathways (near-off)

# Six biological noise sources injected inside the LIF dynamics. Units are SI:
#   v_noise_std         membrane voltage noise        ~1 mV
#   i_noise_std         synaptic current noise        ~15 pA
#   syn_noise_std       multiplicative synaptic noise  0.25 (fractional)
#   threshold_jitter_std spike-threshold jitter        ~1 mV
#   orn_receptor_noise_std receptor-level noise        0.10 (fractional)
#   circuit_noise_enabled master toggle for the above
REALISTIC_PARAMS = SpikingParams(
    v_noise_std=1.0e-3, i_noise_std=15e-12, syn_noise_std=0.25,
    threshold_jitter_std=1.0e-3, orn_receptor_noise_std=0.10,
    circuit_noise_enabled=True,
)

# Concentration-invariance sweep: odor "concentrations" tested (arbitrary units
# relative to EC50). The Hill function maps concentration -> receptor occupancy.
CONCENTRATIONS = [0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
HILL_EC50 = 1.0          # half-max concentration of the Hill response curve
HILL_N = 1               # Hill coefficient (1 = non-cooperative binding)
N_CONC_TRIALS = 10       # noisy trials per (odor, concentration) point
ENERGY_RAMP_EPOCHS = 60  # epochs over which sparsity/energy loss ramps 0->1
DEVICE = torch.device('cpu')  # everything runs on CPU (model is small)

# Output locations, anchored to the package root (two levels up from scripts/).
_pkg_root = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = _pkg_root / 'results' / 'ablations_r1'   # trained-ablation results
TEACHER_DIR = OUTPUT_DIR / 'teachers'                 # cached teacher checkpoints

# ============================================================================
# HELPERS
# ============================================================================
def clamp_biological(model):
    """Project all learned biophysical parameters back into legal ranges.

    Called after every optimizer.step(). First defers to the model's own
    ``clamp_to_biological_bounds`` (handles thresholds, AL synapses, etc.), then
    additionally clamps the log-space KC recurrent / soma parameters here:

      * KC<->KC (aa = axon-axon, ad = axon-dendrite) and PN->KC nonad
        log-strengths into [log(1e-15), log(1e-8)].
      * KC soma conductance log_g_soma into [log(1 nS), log(20 nS)].

    Side effect: mutates ``model`` parameters in place (no_grad).
    """
    model.clamp_to_biological_bounds()
    with torch.no_grad():
        model.kc_layer.kc_kc_aa.log_strength.clamp_(KCKC_LOG_MIN, LOG_STRENGTH_MAX)
        model.kc_layer.kc_neurons.log_g_soma.clamp_(np.log(G_SOMA_MIN), np.log(G_SOMA_MAX_BIO))
        # axon-dendrite KC<->KC synapses only exist in some configs (may be None)
        if model.kc_layer.kc_kc_ad is not None:
            model.kc_layer.kc_kc_ad.log_strength.clamp_(KCKC_LOG_MIN, LOG_STRENGTH_MAX)
        # the "nonad" (developmental) PN->KC pathway is also optional
        if model.kc_layer.pn_kc_nonad is not None:
            model.kc_layer.pn_kc_nonad.log_strength.clamp_(KCKC_LOG_MIN, LOG_STRENGTH_MAX)


def get_param_groups(model, apl_lr_mult=1.0):
    """Build optimizer param groups. apl_lr_mult scales APL gain learning rate.

    Walks every trainable parameter and assigns it a per-parameter learning rate
    of ``BASE_LR * mult``, where ``mult`` is chosen by matching substrings of the
    parameter's dotted name to the LR-multiplier constants above. One Adam group
    per tensor (so each can carry its own LR). This is how the model gives slow,
    careful updates to AL synapses while letting KC/APL knobs move fast.

    Args:
        model: the spiking student.
        apl_lr_mult: extra multiplier on the APL-gain group's LR (lets the CLI
            tune how aggressively APL inhibition is learned).

    Returns:
        list[dict]: param groups consumable by ``torch.optim.Adam``.
    """
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # STD parameters: recovery time constant tau_rec and release-prob logit U.
        if 'log_tau_rec' in name or 'logit_U' in name:
            mult = GSOMA_LR
        elif 'nonad' in name:                       # developmental ("nonad") pathways
            mult = NONAD_LR
        elif 'kc_kc_aa' in name or 'kc_kc_ad' in name:   # KC<->KC chemical synapses
            mult = KCKC_LR
        elif 'kc_kc_dd_gain' in name or 'kc_kc_da_gain' in name:  # KC<->KC gap gains
            mult = KCKC_LR
        elif 'log_g_soma' in name:                  # two-compartment soma coupling
            mult = GSOMA_LR
        elif 'log_g_gap' in name:                   # gap-junction conductances
            mult = AL_LR
        elif 'ln_pn_excit' in name:                 # excitatory LN->PN synapses
            mult = AL_LR
        elif 'ln_ln' in name or 'pn_ln' in name:    # LN<->LN, PN->LN synapses
            mult = AL_LR
        elif 'v_th' in name:                        # spike thresholds, per layer
            if 'kc' in name: mult = KC_VTH_LR
            elif 'ln' in name: mult = LN_VTH_LR
            elif 'orn' in name: mult = ORN_VTH_LR
            elif 'pn' in name: mult = PN_VTH_LR
            else: mult = 0.01
        elif 'or_to_orn' in name or 'or_gains' in name:  # receptor->ORN gains
            mult = 0.5
        elif 'apl_gain' in name:                    # APL inhibitory gain (key knob)
            mult = KC_LR * apl_lr_mult
        elif 'orn_neurons' in name or 'ln_neurons' in name or 'pn_neurons' in name or 'antennal_lobe' in name:
            mult = AL_LR
        elif 'log_tau_apl' in name:                 # APL inhibition time constant
            mult = APL_TAU_LR
        elif 'kc_layer' in name or 'kc_neurons' in name or 'apl' in name:
            mult = KC_LR
        else:
            mult = 1.0                              # default: plain BASE_LR
        param_groups.append({'params': [param], 'lr': BASE_LR * mult})
    return param_groups


# ============================================================================
# CONNECTOME SHUFFLING (C1iii: preserve degree distribution per neuron)
# ============================================================================
def shuffle_connectome(model, seed):
    """Shuffle all connectome weight matrices while preserving per-neuron degree.

    C1(iii): Reviewer 2 — tests whether specific wiring matters or just
    degree distribution. For each connectivity matrix, independently permute
    rows (presynaptic) and columns (postsynaptic), preserving marginal sums.

    Mechanically: for a weight/mask matrix W[pre, post], draw independent
    permutations of the row and column indices and apply W = W[row_perm][:, col_perm].
    Because each row is moved as a unit and each column as a unit, every neuron
    keeps its in-degree and out-degree (the marginal connection counts), but
    *which* specific partners it talks to is randomised. Square symmetric
    matrices (gap-junction masks) use the SAME permutation on rows and columns to
    stay symmetric. A seed offset (+1000) decorrelates this RNG from the training
    seed.

    Args:
        model: the spiking student (mutated in place).
        seed: base random seed; the shuffle uses ``seed + 1000``.

    Side effects: overwrites connectome buffers in place; prints a confirmation.
    """
    rng = np.random.RandomState(seed + 1000)  # Offset to avoid seed collision

    def shuffle_synapse_layer(layer):
        # Degree-preserving partner shuffle of one SpikingConnectomeLinear pathway.
        # SpikingConnectomeLinear stores its FIXED connectome as the registered
        # buffers synapse_counts / norm_counts / mask (each shape (n_pre, n_post));
        # there is NO 'weight_matrix' attribute. (The previous version gated on
        # hasattr(layer, 'weight_matrix'), which is always False, so it silently
        # skipped every chemical pathway and left the main connectome unshuffled --
        # this is the bug being fixed.) We draw ONE independent (row, col)
        # permutation for the pathway and apply it identically to all three buffers
        # so they remain mutually consistent (the forward pass uses
        # norm_counts * mask). Permuting rows (presynaptic) and columns
        # (postsynaptic) as whole units preserves every neuron's in-/out-degree
        # (the marginal synapse counts) while randomising WHICH specific partners
        # it connects to -- exactly the "preserve per-neuron counts, randomise
        # partners" null model the paper describes.
        if layer is None or getattr(layer, 'mask', None) is None:
            return
        row_perm = rng.permutation(layer.mask.shape[0])
        col_perm = rng.permutation(layer.mask.shape[1])
        for attr in ('synapse_counts', 'norm_counts', 'mask'):
            buf = getattr(layer, attr, None)
            if buf is None:
                continue
            buf_np = buf.cpu().numpy().copy()
            buf.copy_(torch.from_numpy(buf_np[row_perm][:, col_perm]).to(buf.dtype))

    def shuffle_raw_buffer(obj, name, symmetric=False):
        # Permute a raw 2-D connectivity/weight buffer in place (used for the APL
        # KC<->APL weight matrices, the graded KC-KC coupling weights, and the
        # gap-junction masks, none of which are SpikingConnectomeLinear layers).
        # `symmetric=True` reuses one permutation for rows and cols so that square,
        # bidirectional gap-junction masks stay symmetric (coupling i<->j implies
        # j<->i).
        buf = getattr(obj, name, None)
        if buf is None:
            return
        buf_np = buf.cpu().numpy().copy()
        row_perm = rng.permutation(buf_np.shape[0])
        if symmetric and buf_np.shape[0] == buf_np.shape[1]:
            col_perm = row_perm
        else:
            col_perm = rng.permutation(buf_np.shape[1])
        buf.copy_(torch.from_numpy(buf_np[row_perm][:, col_perm]).to(buf.dtype))

    # --- Chemical synaptic pathways (the main connectome) ---
    # AL: ORN/LN/PN feedforward (orn_pn, orn_ln, ln_pn, ln_pn_excit) plus the
    # LN-LN / PN-LN / LN-ORN recurrent pathways, AD and non-AD. ln_orn is None in
    # the canonical model (0 AD synapses) and is skipped by the None guard.
    al = model.antennal_lobe
    for layer_name in ['orn_pn', 'orn_ln', 'ln_pn', 'ln_pn_excit', 'ln_ln', 'pn_ln', 'ln_orn',
                       'orn_ln_nonad', 'ln_pn_nonad', 'ln_pn_excit_nonad',
                       'ln_ln_nonad', 'pn_ln_nonad', 'ln_orn_nonad']:
        shuffle_synapse_layer(getattr(al, layer_name, None))

    # KC: PN->KC feedforward (dominant) and spiking KC-KC recurrence (aa / ad).
    kc = model.kc_layer
    for layer_name in ['pn_kc', 'pn_kc_nonad', 'kc_kc_aa', 'kc_kc_ad']:
        shuffle_synapse_layer(getattr(kc, layer_name, None))

    # Graded KC-KC coupling weight buffers (dendrite->dendrite, dendrite->axon).
    for buf_name in ['kc_kc_dd_weights', 'kc_kc_da_weights']:
        shuffle_raw_buffer(kc, buf_name)

    # --- APL pathways — KC->APL and APL->KC weight buffers (raw tensors). ---
    apl = kc.apl
    for buf_name in ['kc_apl_weights', 'apl_kc_weights', 'kc_apl_da_weights']:
        shuffle_raw_buffer(apl, buf_name)

    # --- Gap junction masks — binary [pre, post] adjacency for electrical coupling. ---
    # Square LN-LN / PN-PN masks are symmetric (reuse one permutation); the
    # rectangular eLN-PN mask uses independent row/col perms.
    for mask_name in ['gap_ln_ln_mask', 'gap_pn_pn_mask', 'gap_eln_pn_mask']:
        shuffle_raw_buffer(al, mask_name, symmetric=True)

    print("  Connectome shuffled (degree-preserving partner permutation; all chemical + gap + APL pathways)")


def make_shuffled_connectome(data_dir, seed):
    """Materialize a degree-preserving shuffled COPY of the connectome for one seed.

    This is the data-level shuffle used by the *fair* retrained shuffled-connectome
    ablation (C1iii): instead of permuting a built student's buffers, we permute the
    raw synapse-count matrices on disk and then build BOTH the rate teacher AND the
    spiking student from this same shuffled directory via ``from_data_dir``. That
    guarantees the teacher and student are trained on the IDENTICAL shuffled wiring
    ("shuffle the teacher the same way as the student").

    Every 2-D synapse-count matrix under winding2023/ and winding2023_compartments/
    has its rows (presynaptic) and columns (postsynaptic) independently permuted, so
    each neuron keeps its in-/out-degree while its specific partners are randomized.
    The RNG is seeded with ``seed + 1000``, so DIFFERENT seeds get DIFFERENT shuffles,
    while a given seed's teacher and student share one cached directory (same shuffle).
    The kreher2008 OR-response data and any non-matrix files are copied verbatim.

    Args:
        data_dir: the real connectome root (contains winding2023/, kreher2008/, ...).
        seed: experiment seed; selects the per-seed permutation (via seed + 1000).

    Returns:
        Path to the cached shuffled connectome directory
        (results/_shuffled_connectomes/seed{seed}/), built once and reused.
    """
    import shutil
    out = OUTPUT_DIR.parent / '_shuffled_connectomes' / f'seed{seed}'
    done_flag = out / '.shuffled_ok'
    if done_flag.exists():
        return out  # already built for this seed; reuse so teacher+student match
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed + 1000)  # per-seed => distinct shuffle per seed
    for sub in ['winding2023', 'winding2023_compartments', 'kreher2008']:
        src = data_dir / sub
        if not src.is_dir():
            continue
        dst = out / sub
        dst.mkdir(parents=True, exist_ok=True)
        shuffle_matrices = sub.startswith('winding2023')  # only the connectome, not OR data
        # sorted() => deterministic file order => reproducible RNG draws for this seed.
        for f in sorted(src.iterdir()):
            if shuffle_matrices and f.suffix == '.pt':
                W = torch.load(f, weights_only=True)
                if torch.is_tensor(W) and W.dim() == 2:
                    rp = rng.permutation(W.shape[0])   # presynaptic relabel
                    cp = rng.permutation(W.shape[1])   # postsynaptic relabel
                    W = W[rp][:, cp].contiguous()       # degree-preserving partner shuffle
                torch.save(W, dst / f.name)
            elif f.is_file():
                shutil.copy2(f, dst / f.name)           # OR data / json copied unchanged
    done_flag.write_text('ok')
    print(f"  Built shuffled connectome (degree-preserving, seed {seed}) at {out}")
    return out


# ============================================================================
# LN THRESHOLD OVERRIDE (C3: vary Picky vs Broad/Choosy classification)
# ============================================================================
def override_ln_threshold(model, data_dir, quantile):
    """Reclassify LN subtypes with a different fan-out quantile.

    C3: Reviewer 3, Point 2 — sensitivity to LN classification threshold.
    Default is 0.33 (33rd percentile). Berck et al. 2016 proportional split
    would be ~0.417 (5 picky / 12 connected = 5:7 split).

    BUG FIX: Must rebuild ln_pn and ln_pn_excit SpikingConnectomeLinear layers,
    not just update the is_excitatory_ln buffer. The old code only changed the
    buffer, which had no effect because the weight matrices were already baked
    into separate inhibitory/excitatory layers at init time.

    Biology: a local neuron's *fan-out* (how many PNs it connects to) is used as
    a proxy for its subtype. "Picky" LNs (low fan-out, at/below the quantile) are
    modelled as EXCITATORY onto PNs; "Broad/Choosy" LNs (high fan-out) are
    INHIBITORY. Because the model stores excitatory and inhibitory LN->PN paths
    as two separate sign-locked linear layers, changing the classification means
    *rebuilding* those layers from the raw connectome with the new sign mask.

    Args:
        model: the spiking student (mutated in place).
        data_dir: dataset root; the winding2023 connectome tensors live under it.
        quantile: fan-out quantile in [0, 1] defining the picky/broad cutoff.

    Side effects: replaces several layers on ``model.antennal_lobe`` and prints
    the new excit/inhib/silent counts.
    """
    from core.layers import SpikingConnectomeLinear

    # Raw LN->PN connectivity matrix [n_ln, n_pn] from the Winding 2023 connectome.
    winding_dir = data_dir / 'winding2023'
    ln_to_pn = torch.load(winding_dir / 'ln_to_pn.pt', weights_only=True)

    # Fan-out = number of PNs each LN connects to (count of positive entries/row).
    fan_out = (ln_to_pn > 0).sum(dim=1)
    has_pn = fan_out > 0                 # LNs that connect to at least one PN
    n_connected = has_pn.sum().item()

    if n_connected > 0:
        # Quantile threshold over connected LNs' fan-out; LNs at/below it are
        # the low-fan-out "picky" (excitatory) class.
        connected_fan_out = fan_out[has_pn].float()
        threshold = torch.quantile(connected_fan_out, quantile)
        is_excitatory_ln = has_pn & (fan_out <= threshold)
    else:
        is_excitatory_ln = torch.zeros(ln_to_pn.shape[0], dtype=torch.bool)

    n_excit = is_excitatory_ln.sum().item()
    n_inhib = (has_pn & ~is_excitatory_ln).sum().item()
    n_silent = (~has_pn).sum().item()    # LNs with no PN target ("silent")

    # Update buffer
    model.antennal_lobe.is_excitatory_ln.copy_(is_excitatory_ln)

    # Rebuild AD LN→PN layers with new classification  [C3 FIX]
    # Split the single ln_to_pn matrix into a purely-inhibitory copy (excitatory
    # LN rows zeroed) and a purely-excitatory copy (inhibitory LN rows zeroed).
    al = model.antennal_lobe
    ln_pn_inhib = ln_to_pn.clone()
    ln_pn_inhib[is_excitatory_ln] = 0
    ln_pn_excit = ln_to_pn.clone()
    ln_pn_excit[~is_excitatory_ln] = 0

    # Save log_strength values before replacing layers — initialize_student sets
    # these before calling override_ln_threshold, and new layers reset them.
    old_ln_pn_log_strength = al.ln_pn.log_strength.item()
    old_ln_pn_excit_log_strength = al.ln_pn_excit.log_strength.item()

    # Construct fresh sign-locked linear layers from the re-split matrices.
    al.ln_pn = SpikingConnectomeLinear(ln_pn_inhib, al.ln_pn.params, sign="inhibitory")
    al.ln_pn_excit = SpikingConnectomeLinear(ln_pn_excit, al.ln_pn_excit.params, sign="excitatory")

    # Restore log_strength values that were set by initialize_student
    # (the new layers initialise their own log_strength; overwrite with the
    # carefully-tuned values from initialize_student so the warm start survives).
    with torch.no_grad():
        al.ln_pn.log_strength.fill_(old_ln_pn_log_strength)
        al.ln_pn_excit.log_strength.fill_(old_ln_pn_excit_log_strength)
    print(f"  log_strength restored: ln_pn={old_ln_pn_log_strength:.4f}, "
          f"ln_pn_excit={old_ln_pn_excit_log_strength:.4f}")

    # Rebuild non-AD LN→PN layers if they exist
    # (the developmental "nonad" connectome has its own LN->PN matrix that must
    # be re-split the same way for consistency).
    ln_to_pn_nonad = None
    nonad_path = winding_dir / 'ln_to_pn_nonad.pt'
    if nonad_path.exists():
        ln_to_pn_nonad = torch.load(nonad_path, weights_only=True)
    if ln_to_pn_nonad is not None and al.ln_pn_nonad is not None:
        ln_pn_nonad_inhib = ln_to_pn_nonad.clone()
        ln_pn_nonad_inhib[is_excitatory_ln] = 0
        ln_pn_nonad_excit = ln_to_pn_nonad.clone()
        ln_pn_nonad_excit[~is_excitatory_ln] = 0
        # NOTE: this LOCAL NONAD_INIT (1e-12, a linear strength) shadows the
        # module-level NONAD_INIT (= log(1e-13)); here it is the raw init_strength
        # passed to the new layers, not a log value.
        NONAD_INIT = 1e-12
        al.ln_pn_nonad = SpikingConnectomeLinear(
            ln_pn_nonad_inhib, al.ln_pn_nonad.params,
            init_strength=NONAD_INIT, sign="inhibitory")
        al.ln_pn_excit_nonad = SpikingConnectomeLinear(
            ln_pn_nonad_excit, al.ln_pn_excit_nonad.params,
            init_strength=NONAD_INIT, sign="excitatory")

    # Rebuild eLN-PN gap junction mask
    # The excitatory-LN -> PN electrical coupling mask must follow the new
    # classification: keep gap entries only for the now-excitatory LNs, binarised.
    gap_eln_pn = ln_to_pn.clone().float()
    gap_eln_pn[~is_excitatory_ln] = 0
    gap_eln_pn = (gap_eln_pn > 0).float()
    al.gap_eln_pn_mask.copy_(gap_eln_pn)

    print(f"  LN reclassified (quantile={quantile:.2f}): "
          f"{n_excit} excit (Picky), {n_inhib} inhib (Broad/Choosy), {n_silent} silent")
    print(f"  Rebuilt ln_pn ({int((ln_pn_inhib > 0).sum())} inhib), "
          f"ln_pn_excit ({int((ln_pn_excit > 0).sum())} excit), "
          f"eLN-PN gap ({int(gap_eln_pn.sum())} conns)")


# ============================================================================
# STUDENT INITIALIZATION WITH ABLATIONS
# ============================================================================
def initialize_student(data_dir, n_odors, teacher_state, args):
    """Create student model with optional ablations applied.

    Builds a fresh spiking student, warm-starts a subset of its parameters from
    the trained rate teacher (decoder weights, receptor gains, APL gain) and from
    hand-tuned biophysical defaults, then applies whichever ablation the CLI
    requested (--no-gap / --no-apl / --shuffle / --ln-quantile). Returns the
    model moved to DEVICE.

    Args:
        data_dir: dataset / connectome root.
        n_odors: number of odor classes (decoder output dim).
        teacher_state: state_dict of the trained ConnectomeConstrainedModel.
        args: parsed CLI namespace (carries apl_boost, no_gap, no_apl, ...).

    Returns:
        SpikingConnectomeConstrainedModel on DEVICE, ready to train.
    """
    apl_boost = args.apl_boost

    # Build the spiking student from the connectome; include the developmental
    # ("nonad") pathways and the realistic biological noise sources.
    student = SpikingConnectomeConstrainedModel.from_data_dir(
        data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10,
        params=REALISTIC_PARAMS, include_nonad=True)
    # Override the model's legacy step counts with the canonical N_STEPS=30 for
    # both the antennal-lobe and KC/MB simulation loops.
    student.n_steps_al = N_STEPS
    student.n_steps_kc = N_STEPS

    # Build the rate teacher and load its trained weights (source of warm start).
    teacher = ConnectomeConstrainedModel.from_data_dir(
        data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10)
    teacher.load_state_dict(teacher_state)

    with torch.no_grad():
        # Initialise all spike thresholds: LNs to LN_VTH_INIT (-47.5 mV),
        # everything else to -42.5 mV. (Volts.)
        for name, param in student.named_parameters():
            if 'v_th' in name:
                param.fill_(LN_VTH_INIT if 'ln' in name else -0.0425)

        # Transfer the linear decoder and receptor gains directly from teacher
        # (ANN->SNN transfer: the readout and front-end gains carry over).
        student.decoder.weight.copy_(teacher.decoder.weight)
        student.decoder.bias.copy_(teacher.decoder.bias)
        student.or_to_orn.or_gains.copy_(teacher.or_to_orn.or_gains)
        # APL gain warm-started from teacher and scaled up by apl_boost so the
        # spiking student starts with strong sparsifying inhibition.
        student.kc_layer.apl.apl_gain.data = teacher.kc_layer.apl.apl_gain.data.clone() * apl_boost

        # Nudge KC thresholds up slightly (+5 mV) to keep KCs sparse at init.
        student.kc_layer.kc_neurons.v_th.data += 0.005
        # KC<->KC recurrent synapse strengths (log space): aa ~1e-11 S, ad near-off.
        student.kc_layer.kc_kc_aa.log_strength.fill_(np.log(1e-11))
        if student.kc_layer.kc_kc_ad is not None:
            student.kc_layer.kc_kc_ad.log_strength.fill_(np.log(1e-13))
        # Two-compartment KC soma coupling conductance: init at 10 nS.
        student.kc_layer.kc_neurons.log_g_soma.fill_(np.log(10e-9))

        # AL recurrent inhibition paths start near-off (1e-13 S) so they grow in.
        if student.antennal_lobe.ln_ln is not None:
            student.antennal_lobe.ln_ln.log_strength.fill_(np.log(1e-13))
        if student.antennal_lobe.pn_ln is not None:
            student.antennal_lobe.pn_ln.log_strength.fill_(np.log(1e-13))
        if student.antennal_lobe.ln_orn is not None:
            student.antennal_lobe.ln_orn.log_strength.fill_(np.log(1e-13))
        # Scale up inhibitory LN->PN by LN_PN_SCALE (in log space: + log(scale)).
        ln_pn_orig = student.antennal_lobe.ln_pn.log_strength.item()
        student.antennal_lobe.ln_pn.log_strength.fill_(ln_pn_orig + np.log(LN_PN_SCALE))
        # Set excitatory LN->PN to 10% of the inhibitory strength.
        ln_pn_inhib_strength = np.exp(student.antennal_lobe.ln_pn.log_strength.item())
        student.antennal_lobe.ln_pn_excit.log_strength.fill_(np.log(ln_pn_inhib_strength * 0.1))
        # Scale ORN->PN excitation down by ORN_PN_SCALE (0.7).
        orn_pn_orig = student.antennal_lobe.orn_pn.log_strength.item()
        student.antennal_lobe.orn_pn.log_strength.fill_(orn_pn_orig + np.log(ORN_PN_SCALE))

        # All developmental ("nonad") AL pathways start at NONAD_INIT (~1e-13 S).
        for attr in ['orn_ln_nonad', 'ln_pn_nonad', 'ln_pn_excit_nonad',
                      'ln_ln_nonad', 'pn_ln_nonad', 'ln_orn_nonad']:
            layer = getattr(student.antennal_lobe, attr, None)
            if layer is not None:
                layer.log_strength.fill_(NONAD_INIT)
        if student.kc_layer.pn_kc_nonad is not None:
            student.kc_layer.pn_kc_nonad.log_strength.fill_(NONAD_INIT)

    # ---- APPLY ABLATIONS ----

    # C1(i): Disable gap junctions
    # Freeze every gap-junction log-conductance at log(1e-30) (~0 S) and turn off
    # its gradient, so electrical coupling I = g*(V_pre - V_post) -> ~0 for all
    # LN-LN, PN-PN and eLN-PN gaps. Conductances are stored in log space so a
    # huge-negative log == effectively zero conductance.
    if args.no_gap:
        with torch.no_grad():
            if hasattr(student.antennal_lobe, 'log_g_gap_ln'):
                student.antennal_lobe.log_g_gap_ln.fill_(np.log(1e-30))
                student.antennal_lobe.log_g_gap_ln.requires_grad = False
            if hasattr(student.antennal_lobe, 'log_g_gap_pn'):
                student.antennal_lobe.log_g_gap_pn.fill_(np.log(1e-30))
                student.antennal_lobe.log_g_gap_pn.requires_grad = False
            if hasattr(student.antennal_lobe, 'log_g_gap_eln_pn'):
                student.antennal_lobe.log_g_gap_eln_pn.fill_(np.log(1e-30))
                student.antennal_lobe.log_g_gap_eln_pn.requires_grad = False
        print("  Gap junctions DISABLED (all conductances frozen at ~0)")

    # C1(ii) / C4(ii): Disable APL
    # BUG FIX: apl_gain=0 gives softplus(0)=0.693, so APL is still ~53% active.
    # Use -100 so softplus(-100)≈0, truly zeroing APL output.  [Reviewer 2, Pt 7]
    # (The APL effective gain is softplus(apl_gain) to keep it non-negative; only
    # a large-negative raw value drives the effective gain to ~0.) All APL params
    # are then frozen so training cannot re-grow the inhibition.
    if args.no_apl:
        with torch.no_grad():
            student.kc_layer.apl.apl_gain.fill_(-100.0)
        for p in student.kc_layer.apl.parameters():
            p.requires_grad = False
        print("  APL DISABLED (gain=-100 → softplus≈0, all APL params frozen)")

    # C1(iii): the shuffle is applied at the connectome-DATA level (see
    # make_shuffled_connectome / _main_trained) so the teacher and student share
    # identical shuffled wiring; `data_dir` here is already the shuffled directory,
    # so NO in-model shuffle is applied (that would double-shuffle).

    # R4: Override LN threshold / reclassify picky-vs-broad LNs.
    if args.ln_quantile is not None:
        override_ln_threshold(student, data_dir, args.ln_quantile)

    return student.to(DEVICE)


# ============================================================================
# TRAIN TEACHER
# ============================================================================
def train_teacher(seed, data_dir, n_odors, train_loader, test_loader, cache_tag=''):
    """Train (or load from cache) the rate-based ANN teacher for one seed.

    The teacher is a non-spiking ConnectomeConstrainedModel trained with plain
    cross-entropy + a sparsity penalty. Its decoder, receptor gains and APL gain
    later seed the spiking student (ANN->SNN transfer). Teachers are cached per
    seed under TEACHER_DIR so repeated student runs reuse one teacher.

    Args:
        seed: RNG seed (also names the cache file).
        data_dir: dataset / connectome root.
        n_odors: number of odor classes.
        train_loader, test_loader: DataLoaders over the Kreher 2008 dataset.

    Returns:
        dict: the teacher's CPU state_dict (loaded from cache if present).

    Side effects: writes ``teacher_seed{seed}.pt`` if not already cached; prints
    accuracy every 100 epochs.
    """
    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    # cache_tag separates teachers trained on different connectomes (e.g. '_shuffle'
    # for the shuffled-connectome ablation), so the shuffled teacher never reuses
    # the cached real-connectome teacher.
    teacher_path = TEACHER_DIR / f'teacher{cache_tag}_seed{seed}.pt'

    # Cache hit: skip training entirely and return the saved state_dict.
    if teacher_path.exists():
        print(f"  Teacher for seed {seed} already exists, loading...")
        return torch.load(teacher_path, weights_only=False)

    print(f"--- Training teacher (seed {seed}, {TEACHER_EPOCHS} epochs) ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    teacher = ConnectomeConstrainedModel.from_data_dir(
        data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10)
    opt = torch.optim.Adam(teacher.parameters(), lr=1e-2)
    for ep in range(TEACHER_EPOCHS):
        teacher.train()
        for bx, by in train_loader:
            opt.zero_grad()
            # sparsity_weight=2.0 weights the teacher's KC-sparsity penalty.
            loss, _ = teacher.compute_loss(bx, by, sparsity_weight=2.0)
            loss.backward()
            opt.step()
        if (ep + 1) % 100 == 0:
            # periodic held-out accuracy report
            teacher.eval()
            c, t = 0, 0
            with torch.no_grad():
                for bx, by in test_loader:
                    c += (teacher(bx).argmax(-1) == by).sum().item()
                    t += len(by)
            print(f"  Teacher epoch {ep+1}: {c/t:.1%}")

    # Snapshot to CPU and persist for reuse across student runs.
    teacher_state = {k: v.cpu().clone() for k, v in teacher.state_dict().items()}
    torch.save(teacher_state, teacher_path)
    print(f"  Teacher saved to {teacher_path}")
    return teacher_state


# ============================================================================
# TRAIN STUDENT
# ============================================================================
def train_student(args, data_dir, train_loader, test_loader,
                  n_odors, or_responses, teacher_state):
    """Train (or recompute-eval) the spiking student under the chosen ablation.

    Builds the loss description string, initialises the student (with ablations),
    runs ``STUDENT_EPOCHS`` of surrogate-gradient training with a ramped
    sparsity/energy penalty, then runs the full evaluation suite (accuracy,
    per-layer sparsity, AL/MB decorrelation, Mancini APL test, concentration
    invariance) and writes both a checkpoint and a results JSON.

    Loss = cross-entropy [ + sparsity penalty ] [ + energy penalty ].

    Args:
        args: parsed CLI namespace (controls loss terms, ablations, recompute).
        data_dir: dataset / connectome root.
        train_loader, test_loader: DataLoaders.
        n_odors: number of odor classes.
        or_responses: [n_odors, n_or_types] receptor response matrix (for the
            decorrelation / invariance analyses).
        teacher_state: teacher state_dict for the warm start.

    Side effects: writes ``model_{label}.pt`` and ``results_{label}.json`` under
    OUTPUT_DIR; prints progress and a final report.
    """
    energy_weight = args.energy_weight
    kc_sparsity = args.kc_sparsity
    kc_energy_only = args.kc_energy_only
    label = args.label
    seed = args.seed

    # Assemble a human-readable description of the active loss terms + ablations
    # (printed and stored in the results JSON for provenance).
    loss_parts = ["CE"]
    if kc_sparsity: loss_parts.append("KC_sp")
    if energy_weight > 0:
        etype = "KC_energy" if kc_energy_only else "all_energy"
        loss_parts.append(f"{energy_weight}*{etype}")
    ablations = []
    if args.no_gap: ablations.append("no_gap")
    if args.no_apl: ablations.append("no_apl")
    if args.shuffle: ablations.append("shuffle")
    if args.ln_quantile is not None: ablations.append(f"ln_q{args.ln_quantile:.2f}")
    if args.apl_boost != APL_BOOST_DEFAULT: ablations.append(f"apl_boost={args.apl_boost}")

    loss_desc = " + ".join(loss_parts)
    if ablations: loss_desc += f" [{', '.join(ablations)}]"

    print(f"\n{'='*70}")
    print(f"TRAINING: {label} (seed {seed})")
    print(f"Loss = {loss_desc}")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build the ablated, warm-started student and attach the ORN/LN spike hooks.
    student = initialize_student(data_dir, n_odors, teacher_state, args)
    accumulator = SpikeAccumulator().register(student)

    param_groups = get_param_groups(student, apl_lr_mult=args.apl_lr_mult)
    optimizer = torch.optim.Adam(param_groups)

    # --recompute: eval-only mode. Load the existing checkpoint, skip training,
    # and recompute metrics live (the epoch loop range becomes 0).
    if getattr(args, 'recompute', False):
        ckpt = OUTPUT_DIR / f'model_{label}.pt'
        student.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False))
        student.to(DEVICE)
        print(f"  [recompute] loaded {ckpt.name}; skipping training, recomputing metrics live from checkpoint")

    for epoch in range(0 if getattr(args, 'recompute', False) else STUDENT_EPOCHS):
        # Linear 0->1 ramp of the auxiliary-loss weight over ENERGY_RAMP_EPOCHS,
        # so the model first learns to classify before being pushed toward
        # sparsity / low energy.
        progress = min(1.0, epoch / ENERGY_RAMP_EPOCHS)
        sp_w = progress * MAX_SP_WEIGHT if kc_sparsity else 0.0
        e_w = progress * energy_weight

        student.train()
        train_correct, train_total = 0, 0

        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            accumulator.reset()

            # Full spiking forward; info carries PN/KC spike counts (summed over
            # N_STEPS) used by the auxiliary losses.
            logits, info = student(bx, return_all=True)
            ce_loss = F.cross_entropy(logits, by)
            total_loss = ce_loss

            # KC firing RATE = spike count / N_STEPS (fraction of steps active).
            kc_rates = info['kc_spikes'] / N_STEPS
            if kc_sparsity:
                # Canonical KC-sparsity penalty: push the mean of a soft step
                # (sigmoid centred at rate=0.02, slope 50) toward target 0.05,
                # i.e. ~5% of KCs active. Squared error around that target.
                sp_loss = (torch.sigmoid((kc_rates - 0.02) * 50).mean() - 0.05) ** 2
                total_loss = total_loss + sp_w * sp_loss

            if energy_weight > 0:
                if kc_energy_only:
                    # Energy = mean KC firing rate only.
                    energy_loss = kc_rates.mean()
                else:
                    # Whole-pathway energy = mean firing rate averaged across
                    # ORN, LN, PN, KC populations (each = spikes / N_STEPS).
                    orn_rate = accumulator.orn_spikes.mean() / N_STEPS
                    ln_rate = accumulator.ln_spikes.mean() / N_STEPS
                    pn_rate = info['pn_spikes'].mean() / N_STEPS
                    kc_rate = kc_rates.mean()
                    energy_loss = (orn_rate + ln_rate + pn_rate + kc_rate) / 4.0
                total_loss = total_loss + e_w * energy_loss

            total_loss.backward()
            # Clip the global gradient norm (surrogate gradients can be spiky).
            torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
            optimizer.step()
            # Project parameters back into biological bounds after every step.
            clamp_biological(student)

            train_correct += (logits.argmax(-1) == by).sum().item()
            train_total += len(by)

        # Periodic (every 50 epochs) held-out diagnostics: accuracy, per-layer
        # activity fractions, AL/MB decorrelation and the Mancini ratio.
        if (epoch + 1) % 50 == 0:
            student.eval()
            tc, tt = 0, 0
            all_orn, all_ln, all_pn, all_kc = [], [], [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    accumulator.reset()
                    logits, info = student(bx, return_all=True)
                    tc += (logits.argmax(-1) == by).sum().item()
                    tt += len(by)
                    # Fraction of neurons that fired at all (>0 spikes) this batch.
                    all_orn.append((accumulator.orn_spikes > 0).float().mean().item())
                    all_ln.append((accumulator.ln_spikes > 0).float().mean().item())
                    all_pn.append((info['pn_spikes'] > 0).float().mean().item())
                    all_kc.append((info['kc_spikes'] > 0).float().mean().item())

            # Decorrelation / Mancini analyses run on CPU; move there and back.
            student.cpu()
            decorr = compute_mean_sim_decorrelation(student, or_responses)
            manc = _run_mancini_test(student)
            student.to(DEVICE)

            # Sp = KC active fraction (the KC sparsity proxy used across all runs).
            print(f"  Ep {epoch+1}: Train={train_correct/train_total:.1%}, Test={tc/tt:.1%}, "
                  f"Sp={np.mean(all_kc):.1%}, AL={decorr['al_decorr']:.1f}%, MB={decorr['mb_decorr']:.1f}%, "
                  f"Manc={manc['ratio']:.2f}")

    # ---- FINAL EVALUATION ----
    student.eval()
    print(f"\n  --- FINAL EVALUATION: {label} ---")

    # Re-measure held-out accuracy + per-layer activity fractions one last time.
    tc, tt = 0, 0
    all_orn, all_ln, all_pn, all_kc = [], [], [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            accumulator.reset()
            logits, info = student(bx, return_all=True)
            tc += (logits.argmax(-1) == by).sum().item()
            tt += len(by)
            all_orn.append((accumulator.orn_spikes > 0).float().mean().item())
            all_ln.append((accumulator.ln_spikes > 0).float().mean().item())
            all_pn.append((info['pn_spikes'] > 0).float().mean().item())
            all_kc.append((info['kc_spikes'] > 0).float().mean().item())

    accumulator.remove()  # detach hooks before the CPU analysis phase
    test_acc = tc / tt
    per_type_sp = {
        'orn': float(np.mean(all_orn)), 'ln': float(np.mean(all_ln)),
        'pn': float(np.mean(all_pn)), 'kc': float(np.mean(all_kc)),
    }

    student.cpu()
    # Per-pair decorrelation: how much each stage (AL, MB) decorrelates odor
    # representations relative to the OR input, averaged over noisy trials.
    pp = _compute_per_pair_decorrelation(student, or_responses, 10, NOISE_STD)
    # Skip Mancini test when APL is disabled — the test is meaningless without APL
    if args.no_apl:
        manc = {'ratio': float('nan'), 'passes': False,
                'baseline_spikes': float('nan'), 'boosted_spikes': float('nan'),
                'skipped': True}
        print("  Mancini: SKIPPED (APL disabled)")
    else:
        manc = _run_mancini_test(student)
    # Centroid (nearest-prototype) accuracy and concentration-invariance suite.
    cent_acc = _centroid_accuracy(student, or_responses, 20, NOISE_STD)
    conc_results, conc_tests = _run_concentration_invariance(
        student, or_responses, seed, CONCENTRATIONS, HILL_EC50, HILL_N,
        N_CONC_TRIALS, NOISE_STD)

    # Recover the soma conductance (S -> nS) and the effective APL gain
    # (softplus of the raw learned parameter) for reporting.
    g_soma = np.exp(student.kc_layer.kc_neurons.log_g_soma.item()) * 1e9
    apl_gain = F.softplus(torch.tensor(student.kc_layer.apl.apl_gain.item())).item()

    # Gap junction conductances — recover each log_g_gap_* (S) to nS for the JSON.
    gap_conductances = {}
    al = student.antennal_lobe
    for gname in ['log_g_gap_ln', 'log_g_gap_pn', 'log_g_gap_eln_pn']:
        if hasattr(al, gname):
            gap_conductances[gname.replace('log_g_gap_', '') + '_nS'] = float(
                np.exp(getattr(al, gname).item()) * 1e9)

    print(f"  Acc:     linear={test_acc:.1%}, centroid={cent_acc:.1%}")
    print(f"  Sparse:  ORN={per_type_sp['orn']:.1%}, LN={per_type_sp['ln']:.1%}, "
          f"PN={per_type_sp['pn']:.1%}, KC={per_type_sp['kc']:.1%}")
    print(f"  Decorr:  AL={pp['al_decorr']:.1f}%, MB={pp['mb_decorr']:.1f}%")
    print(f"  Mancini: {manc['ratio']:.2f} ({'PASS' if manc['passes'] else 'FAIL'})")
    print(f"  Gain:    OR={conc_tests['or_range']:.2f}x, "
          f"PN={conc_tests['pn_range']:.2f}x, KC={conc_tests['kc_range']:.2f}x")
    print(f"  FlatKC:  {conc_tests['flat_kc_activity']}, SubPN: {conc_tests['sublinear_pn_gain']}")

    # Persist the trained student (unless in eval-only --recompute mode).
    model_path = OUTPUT_DIR / f'model_{label}.pt'
    if not getattr(args, 'recompute', False):
        torch.save(student.state_dict(), model_path)

    # Assemble the results record written to disk.
    results = {
        'label': label, 'seed': seed, 'loss': loss_desc,
        'energy_weight': energy_weight, 'apl_boost': args.apl_boost,
        'accuracy': test_acc, 'centroid_accuracy': cent_acc,
        'per_type_sparsity': per_type_sp,
        'decorrelation': {
            'al': pp['al_decorr'], 'mb': pp['mb_decorr'], 'total': pp['total_decorr'],
        },
        'mancini': {'ratio': manc['ratio'], 'passes': manc['passes'],
                    'baseline': manc['baseline_spikes'], 'boosted': manc['boosted_spikes']},
        'concentration_invariance': {
            'or_range': conc_tests['or_range'], 'pn_range': conc_tests['pn_range'],
            'kc_range': conc_tests['kc_range'],
            'sublinear_pn': conc_tests['sublinear_pn_gain'],
            'flat_kc': conc_tests['flat_kc_activity'],
            'robust_class': conc_tests['robust_classification'],
            'odor_identity': conc_tests['odor_identity_preservation'],
        },
        'g_soma_nS': g_soma, 'apl_gain': apl_gain,
        'gap_conductances_nS': gap_conductances,
    }

    results_path = OUTPUT_DIR / f'results_{label}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  Saved: {results_path}")
    print(f"  DONE: {label}")


# ============================================================================
# MAIN
# ============================================================================
def _main_trained():
    """CLI entry point for the canonical trained-ablation path (no subcommand).

    Parses flags, locates the dataset, builds the Kreher-2008 loaders, trains (or
    loads) the teacher, then trains the student under the requested ablation. If
    ``--train-teacher`` is given, only the teacher is trained/cached and the
    function returns early.
    """
    parser = argparse.ArgumentParser(description='Ablation training for CCN revisions')
    parser.add_argument('--train-teacher', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--label', type=str, required=False)
    # Loss components
    parser.add_argument('--energy-weight', type=float, default=0.0)
    parser.add_argument('--kc-sparsity', action='store_true')
    parser.add_argument('--kc-energy-only', action='store_true')
    # Ablations
    parser.add_argument('--no-gap', action='store_true', help='C1(i): disable gap junctions')
    parser.add_argument('--no-apl', action='store_true', help='C1(ii)/C4(ii): disable APL')
    parser.add_argument('--shuffle', action='store_true', help='C1(iii): shuffle connectome')
    # Sensitivity
    parser.add_argument('--ln-quantile', type=float, default=None,
                        help='R4: LN fan-out quantile (default 0.33)')
    parser.add_argument('--apl-boost', type=float, default=APL_BOOST_DEFAULT,
                        help='APL gain boost factor (default 4.0)')
    parser.add_argument('--apl-lr-mult', type=float, default=1.0,
                        help='Multiplier on APL gain learning rate')
    parser.add_argument('--recompute', action='store_true',
                        help='Eval-only: load the saved model_<label>.pt and recompute all '
                             'metrics live from the checkpoint (no training, no checkpoint overwrite)')
    args = parser.parse_args()

    # Locate the dataset directory by probing several likely roots (handles the
    # Locate the dataset directory.
    _parent = Path(__file__).resolve().parent.parent.parent
    _data_candidates = [
        Path(__file__).resolve().parent.parent.parent / 'data',
    ]
    data_dir = next((p for p in _data_candidates if (p / 'kreher2008').is_dir()), None)
    if data_dir is None:
        raise FileNotFoundError('Cannot find connectome data.')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build train/test datasets with multiplicative trial noise, then loaders.
    train_ds, test_ds, odor_names = load_kreher2008_all_odors(
        data_dir, train_repeats=10, test_repeats=5,
        noise_std=NOISE_STD, noise_type=NOISE_TYPE)
    train_loader, test_loader = create_dataloaders(train_ds, test_ds, batch_size=16)
    n_odors = len(odor_names)
    # Clean (noise-free) normalised OR responses [n_odors, n_or_types] for analysis.
    df = pd.read_csv(data_dir / "kreher2008/orn_responses_normalized.csv", index_col=0)
    or_responses = torch.from_numpy(df.values).float()

    # --train-teacher: only train/cache the teacher, then exit.
    if args.train_teacher:
        train_teacher(args.seed, data_dir, n_odors, train_loader, test_loader)
        return

    # C1(iii) fair shuffle: build a per-seed degree-preserving shuffled connectome
    # and train BOTH the teacher and the student on it, so they share identical
    # shuffled wiring. Different seeds -> different shuffles (see make_shuffled_connectome).
    conn_dir, cache_tag = data_dir, ''
    if args.shuffle:
        conn_dir = make_shuffled_connectome(data_dir, args.seed)
        cache_tag = '_shuffle'
        print(f"  [shuffle] teacher + student will train on shuffled connectome: {conn_dir}")

    # Otherwise: get the (cached) teacher and run a full student training.
    teacher_state = train_teacher(args.seed, conn_dir, n_odors, train_loader, test_loader, cache_tag=cache_tag)
    train_student(args, conn_dir, train_loader, test_loader,
                  n_odors, or_responses, teacher_state)



# ============================================================================
# POST-HOC ABLATION  (merged from former run_posthoc_ablation.py; eval-only)
# ============================================================================
# Output / input locations for the eval-only post-hoc path.
POSTHOC_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / 'results' / 'posthoc_ablations'
CANONICAL_DIR = Path(__file__).resolve().parent.parent.parent / 'results' / 'all_connections_nonad_canonical'
POSTHOC_N_CONC_TRIALS = 15  # more concentration trials than training (tighter CI)
# Dataset directory for the post-hoc path (probes the in-package and external copies).
POSTHOC_DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'data'


class _PosthocSpikeAccumulator:
    """ORN/LN spike accumulator for the post-hoc path.

    Functionally identical to :class:`SpikeAccumulator` but registers its hooks
    in ``__init__`` (given the model up front) rather than via a separate
    ``register`` call. Sums ``output[1]`` (spikes, [batch, n]) over the inner
    time steps for the ORN and LN layers.
    """
    def __init__(self, model):
        self.orn_spikes = None
        self.ln_spikes = None
        self._hooks = []
        def orn_hook(module, input, output):
            spk = output[1]
            self.orn_spikes = spk if self.orn_spikes is None else self.orn_spikes + spk
        def ln_hook(module, input, output):
            spk = output[1]
            self.ln_spikes = spk if self.ln_spikes is None else self.ln_spikes + spk
        self._hooks.append(model.antennal_lobe.orn_neurons.register_forward_hook(orn_hook))
        self._hooks.append(model.antennal_lobe.ln_neurons.register_forward_hook(ln_hook))

    def reset(self):
        """Clear accumulated ORN/LN spike sums before a new forward pass."""
        self.orn_spikes = None
        self.ln_spikes = None

    def remove(self):
        """Detach all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def load_or_responses():
    """Load Kreher 2008 OR response data.

    Returns the normalised receptor-response matrix and its dimensions. Prefers
    a cached ``.pt`` tensor; falls back to the CSV.

    Returns:
        tuple: (or_responses [n_odors, n_or_types] float tensor, n_odors, n_or_types).
    """
    kreher_dir = POSTHOC_DATA_DIR / 'kreher2008'
    pt_path = kreher_dir / 'orn_responses_normalized.pt'
    csv_path = kreher_dir / 'orn_responses_normalized.csv'

    if pt_path.exists():
        or_responses = torch.load(pt_path, weights_only=True)
    else:
        import pandas as pd
        df = pd.read_csv(csv_path, index_col=0)
        or_responses = torch.tensor(df.values, dtype=torch.float32)

    n_odors, n_or_types = or_responses.shape
    print(f"Kreher 2008 data: {n_odors} odors x {n_or_types} OR types")
    return or_responses, n_odors, n_or_types


# ============================================================================
# BUILD FRESH MODEL AND LOAD WEIGHTS
# ============================================================================
def load_canonical_model(seed, n_odors):
    """Build a fresh model and load canonical trained weights.

    Used by the post-hoc path: construct the spiking model with the canonical
    N_STEPS and realistic noise, then overwrite its parameters with the trained
    canonical checkpoint for ``seed``. Returned in eval mode.

    Args:
        seed: identifies the canonical checkpoint to load.
        n_odors: number of odor classes (decoder output dim).

    Returns:
        SpikingConnectomeConstrainedModel in eval mode with canonical weights.
    """
    model = SpikingConnectomeConstrainedModel.from_data_dir(
        POSTHOC_DATA_DIR, n_odors=n_odors, n_or_types=21, target_sparsity=0.10,
        params=REALISTIC_PARAMS, include_nonad=True)
    model.n_steps_al = N_STEPS
    model.n_steps_kc = N_STEPS

    # Load trained canonical weights
    model_path = CANONICAL_DIR / f'model_seed{seed}.pt'
    state = torch.load(model_path, weights_only=False, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model


# ============================================================================
# ABLATION FUNCTIONS (post-hoc, no retraining)
# ============================================================================
def ablate_gap_junctions(model):
    """Zero out all gap junction conductances.  [C1 post-hoc (i)]

    Sets every log_g_gap_* to log(1e-30) (~0 S) in place so the electrical
    coupling current I = g*(V_pre - V_post) vanishes. Unlike the training-time
    version, gradients are irrelevant here (eval only), so requires_grad is left
    untouched.
    """
    with torch.no_grad():
        if hasattr(model.antennal_lobe, 'log_g_gap_ln'):
            model.antennal_lobe.log_g_gap_ln.fill_(np.log(1e-30))
        if hasattr(model.antennal_lobe, 'log_g_gap_pn'):
            model.antennal_lobe.log_g_gap_pn.fill_(np.log(1e-30))
        if hasattr(model.antennal_lobe, 'log_g_gap_eln_pn'):
            model.antennal_lobe.log_g_gap_eln_pn.fill_(np.log(1e-30))
    print("  Post-hoc: gap junctions zeroed (all conductances -> ~0)")


def ablate_apl(model):
    """Disable APL by setting gain to -100 (softplus -> ~0).  [C1 post-hoc (ii)]

    Because the effective APL gain is softplus(apl_gain), a raw value of -100
    drives the effective gain to ~0, removing the divisive (shunting) feedback
    inhibition onto the KCs.
    """
    with torch.no_grad():
        model.kc_layer.apl.apl_gain.fill_(-100.0)
    print(f"  Post-hoc: APL disabled (gain=-100, softplus={F.softplus(torch.tensor(-100.0)).item():.2e})")


# ============================================================================
# EVALUATION (same metrics as training scripts)
# ============================================================================
def _posthoc_evaluate_model(model, or_responses, seed, label, skip_mancini=False):
    """Run full evaluation suite on a model.

    Generates noisy test data, scores classification accuracy and per-layer
    activity, then computes decorrelation, (optionally) the Mancini APL test,
    centroid accuracy and concentration invariance. Returns a results dict.

    Args:
        model: a loaded (and possibly ablated) spiking model, eval mode.
        or_responses: [n_odors, n_or_types] clean receptor responses.
        seed: RNG seed for the concentration-invariance sweep.
        label: condition label stored in the results.
        skip_mancini: True for the no-APL condition (test is meaningless then).

    Returns:
        dict of metrics (accuracy, sparsity, decorrelation, Mancini, etc.).
    """
    n_odors = or_responses.shape[0]
    # Generate test data
    # MULTIPLICATIVE noise (30% CV), matching training/canonical eval. (Was additive — a bug:
    # additive 0.3 SD on OR values ~0.05-1.0 is wildly out-of-distribution and artificially
    # collapsed post-hoc accuracy, e.g. gap 22% instead of the true ~70%.)
    # noisy[odor, trial, or_type] = clean * (1 + N(0,1)*0.3); 5 noisy trials/odor.
    noisy = or_responses.unsqueeze(1) * (1.0 + torch.randn(n_odors, 5, or_responses.shape[1]) * NOISE_STD)
    noisy = noisy.clamp(min=0)  # receptor responses cannot be negative
    test_X = noisy.reshape(-1, or_responses.shape[1])          # [n_odors*5, n_or_types]
    test_y = torch.arange(n_odors).unsqueeze(1).expand(-1, 5).reshape(-1)  # labels

    # Forward pass with spike accumulator (batched in chunks of 28 = #odors).
    accumulator = _PosthocSpikeAccumulator(model)
    model.eval()
    with torch.no_grad():
        all_orn, all_ln, all_pn, all_kc = [], [], [], []
        tc, tt = 0, 0
        for i in range(0, len(test_X), 28):
            bx = test_X[i:i+28]
            by = test_y[i:i+28]
            accumulator.reset()
            logits, info = model(bx, return_all=True)
            tc += (logits.argmax(-1) == by).sum().item()
            tt += len(by)
            # Fraction of each population active (>0 spikes) per batch.
            all_orn.append((accumulator.orn_spikes > 0).float().mean().item())
            all_ln.append((accumulator.ln_spikes > 0).float().mean().item())
            all_pn.append((info['pn_spikes'] > 0).float().mean().item())
            all_kc.append((info['kc_spikes'] > 0).float().mean().item())

    accumulator.remove()
    test_acc = tc / tt
    per_type = {
        'orn': float(np.mean(all_orn)), 'ln': float(np.mean(all_ln)),
        'pn': float(np.mean(all_pn)), 'kc': float(np.mean(all_kc)),
    }

    # Decorrelation
    pp = _compute_per_pair_decorrelation(model, or_responses, 10, NOISE_STD)

    # Mancini (skip for no-APL)
    if skip_mancini:
        manc = {'ratio': float('nan'), 'passes': False,
                'baseline_spikes': float('nan'), 'boosted_spikes': float('nan'),
                'skipped': True}
    else:
        manc = _run_mancini_test(model)

    # Centroid accuracy
    cent_acc = _centroid_accuracy(model, or_responses, 20, NOISE_STD)

    # Concentration invariance
    conc_results, conc_tests = _run_concentration_invariance(
        model, or_responses, seed, CONCENTRATIONS, HILL_EC50, HILL_N,
        POSTHOC_N_CONC_TRIALS, NOISE_STD)

    # Recover g_soma (S->nS) and effective APL gain (softplus) for reporting.
    g_soma = np.exp(model.kc_layer.kc_neurons.log_g_soma.item()) * 1e9
    apl_gain = F.softplus(torch.tensor(model.kc_layer.apl.apl_gain.item())).item()

    # Print summary
    manc_str = 'SKIP' if skip_mancini else f"{manc['ratio']:.2f}"
    print(f"  Acc:     linear={test_acc:.1%}, centroid={cent_acc:.1%}")
    print(f"  Sparse:  ORN={per_type['orn']:.1%}, LN={per_type['ln']:.1%}, "
          f"PN={per_type['pn']:.1%}, KC={per_type['kc']:.1%}")
    print(f"  Decorr:  AL={pp['al_decorr']:.1f}%, MB={pp['mb_decorr']:.1f}%")
    print(f"  Mancini: {manc_str}")
    print(f"  FlatKC:  {conc_tests['flat_kc_activity']}, "
          f"KC range: {conc_tests['kc_range']:.2f}x")

    return {
        'label': label, 'seed': seed,
        'accuracy': test_acc, 'centroid_accuracy': cent_acc,
        'per_type_sparsity': per_type,
        'decorrelation': {
            'al': pp['al_decorr'], 'mb': pp['mb_decorr'], 'total': pp['total_decorr'],
        },
        'mancini': {'ratio': manc['ratio'], 'passes': manc.get('passes', False),
                    'baseline': manc.get('baseline_spikes', 0),
                    'boosted': manc.get('boosted_spikes', 0)},
        'concentration_invariance': conc_tests,
        'g_soma_nS': g_soma, 'apl_gain': apl_gain,
    }


# ============================================================================
# MAIN

def _main_posthoc():
    """CLI entry point for the ``posthoc`` subcommand (eval-only ablations).

    For each seed in the (module-level) ``SEEDS`` list, loads the canonical
    checkpoint twice — once to ablate gap junctions, once to ablate APL — scores
    both, then writes per-condition JSON and prints a 5-seed summary table.

    NOTE: ``SEEDS`` is not defined at module scope in this merged file; this
    function relies on it existing (a side effect that did not survive the merge
    of the former run_posthoc_ablation.py). Documented, not fixed.
    """
    or_responses, n_odors, n_or_types = load_or_responses()

    all_results = {'no_gap_posthoc': [], 'no_apl_posthoc': []}

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        # (i) Gap junctions removed post-hoc
        print(f"\n--- Post-hoc: Remove gap junctions (seed {seed}) ---")
        model = load_canonical_model(seed, n_odors)
        ablate_gap_junctions(model)
        result = _posthoc_evaluate_model(model, or_responses, seed,
                                f'no_gap_posthoc_s{seed}')
        all_results['no_gap_posthoc'].append(result)
        del model  # free the model before loading the next copy

        # (ii) APL disabled post-hoc (skip Mancini — meaningless without APL).
        print(f"\n--- Post-hoc: Disable APL (seed {seed}) ---")
        model = load_canonical_model(seed, n_odors)
        ablate_apl(model)
        result = _posthoc_evaluate_model(model, or_responses, seed,
                                f'no_apl_posthoc_s{seed}', skip_mancini=True)
        all_results['no_apl_posthoc'].append(result)
        del model

    # Save results — one JSON per condition.
    POSTHOC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)   # ensure output dir exists before writing
    for condition, results in all_results.items():
        out_path = POSTHOC_OUTPUT_DIR / f'results_{condition}.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")

    # Summary table — mean +/- std across seeds for each condition.
    print(f"\n{'='*90}")
    print("POST-HOC ABLATION SUMMARY (5 seeds)")
    print(f"{'='*90}")
    print(f"{'Condition':<25} {'Accuracy':>12} {'Centroid':>12} {'KC%':>7} "
          f"{'AL dec':>8} {'MB dec':>8} {'Mancini':>8} {'FlatKC':>7}")
    print('-' * 90)

    for condition, label in [('no_gap_posthoc', 'No Gap (post-hoc)'),
                              ('no_apl_posthoc', 'No APL (post-hoc)')]:
        data = all_results[condition]
        accs = [r['accuracy'] for r in data]
        cents = [r['centroid_accuracy'] for r in data]
        kcs = [r['per_type_sparsity']['kc'] for r in data]
        als = [r['decorrelation']['al'] for r in data]
        mbs = [r['decorrelation']['mb'] for r in data]
        # Count how many seeds showed flat (concentration-invariant) KC activity;
        # tolerate either the new ('flat_kc_activity') or old ('flat_kc') key.
        flat = sum(1 for r in data
                   if r['concentration_invariance'].get('flat_kc_activity',
                      r['concentration_invariance'].get('flat_kc', False)))

        # Average only the non-NaN Mancini ratios (NaN = skipped for no-APL).
        manc_vals = [r['mancini']['ratio'] for r in data
                     if not np.isnan(r['mancini']['ratio'])]
        manc_str = f"{np.mean(manc_vals):.2f}" if manc_vals else "SKIP"

        print(f"{label:<25} {np.mean(accs):>5.1%}+/-{np.std(accs):.1%} "
              f"{np.mean(cents):>5.1%}+/-{np.std(cents):.1%} "
              f"{np.mean(kcs):>6.1%} {np.mean(als):>7.1f} {np.mean(mbs):>7.1f} "
              f"{manc_str:>8} {flat}/{len(data)}")


# ============================================================================
# STD ABLATION  (merged from former run_std_ablation.py)
# ============================================================================
STD_ABLATION_DIR = Path(__file__).resolve().parent.parent.parent / 'results' / 'std_ablation'


def _std_param_groups(model):
    """Per-parameter LR groups for the STD-ablation path.

    Near-identical to :func:`get_param_groups` but with NO 'apl_gain' special
    case and NO apl_lr_mult argument — APL gain falls through to the generic
    'apl' branch (KC_LR). Returns Adam-ready param groups (lr = BASE_LR * mult).
    """
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # STD parameters: recovery tau and release-prob logit U.
        if 'log_tau_rec' in name or 'logit_U' in name:
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
            mult = AL_LR
        elif 'ln_pn_excit' in name:
            mult = AL_LR
        elif 'ln_ln' in name or 'pn_ln' in name:
            mult = AL_LR
        elif 'v_th' in name:
            if 'kc' in name:
                mult = KC_VTH_LR
            elif 'ln' in name:
                mult = LN_VTH_LR
            elif 'orn' in name:
                mult = ORN_VTH_LR
            elif 'pn' in name:
                mult = PN_VTH_LR
            else:
                mult = 0.01
        elif 'or_to_orn' in name or 'or_gains' in name:
            mult = 0.5
        elif 'orn_neurons' in name or 'ln_neurons' in name or 'pn_neurons' in name or 'antennal_lobe' in name:
            mult = AL_LR
        elif 'log_tau_apl' in name:
            mult = APL_TAU_LR
        elif 'kc_layer' in name or 'kc_neurons' in name or 'apl' in name:
            mult = KC_LR
        else:
            mult = 1.0
        param_groups.append({'params': [param], 'lr': BASE_LR * mult})
    return param_groups


def compute_per_pair_decorrelation(model, or_responses, n_trials=10, disable_std=False):
    """Wrapper around canonical decorrelation, with optional STD disable.

    When ``disable_std`` is True, temporarily monkeypatches model.forward so all
    internal forward calls run with Tsodyks-Markram short-term depression turned
    off, runs the canonical per-pair decorrelation, then restores forward.

    Returns:
        dict remapping the canonical keys to this script's naming
        (kc_or/kc_pn/pn_or ratios and al/mb/total decorrelation percentages).
    """
    if disable_std:
        _patch_model_disable_std(model)
    r = _compute_per_pair_decorrelation(model, or_responses, n_trials, NOISE_STD)
    if disable_std:
        _unpatch_model(model)
    return {
        'kc_or': r['kc_or_ratio'], 'kc_pn': r['kc_pn_ratio'], 'pn_or': r['pn_or_ratio'],
        'total_decorr_pct': r['total_decorr'], 'mb_decorr_pct': r['mb_decorr'],
        'al_decorr_pct': r['al_decorr'],
    }


def run_mancini(model, disable_std=False, carbachol=1e-10, apl_inject=0.7):
    """Mancini 2023 test. STD disable applies to baseline-activity pass; the
    Mancini protocol itself uses disable_apl / apl_inject_current flags that
    are layered on top.

    Args:
        model: spiking model under test.
        disable_std: if True, run with STD off (via the forward monkeypatch).
        carbachol: cholinergic-agonist drive level (A) used by the protocol.
        apl_inject: injected APL current fraction used by the protocol.

    Returns:
        float: the Mancini APL-modulation ratio.
    """
    if disable_std:
        _patch_model_disable_std(model)
    result = _run_mancini_test(model, carbachol, apl_inject)
    if disable_std:
        _unpatch_model(model)
    return result['ratio']


def run_concentration_invariance(model, or_responses, seed, disable_std=False):
    """Concentration-invariance sweep wrapper (optionally with STD disabled).

    Returns the canonical ``(per_concentration_results, summary_tests)`` tuple.
    """
    if disable_std:
        _patch_model_disable_std(model)
    r = _run_concentration_invariance(
        model, or_responses, seed, CONCENTRATIONS, HILL_EC50, HILL_N, N_CONC_TRIALS, NOISE_STD)
    if disable_std:
        _unpatch_model(model)
    return r


def centroid_accuracy(model, or_responses, n_trials=20, disable_std=False):
    """Centroid (nearest-prototype) accuracy wrapper, optionally with STD off."""
    if disable_std:
        _patch_model_disable_std(model)
    r = _centroid_accuracy(model, or_responses, n_trials, NOISE_STD)
    if disable_std:
        _unpatch_model(model)
    return r


def _std_evaluate_model(model, test_loader, disable_std=False):
    """Score linear-decoder accuracy and mean KC sparsity on the test loader.

    Args:
        model: spiking model in eval mode.
        test_loader: DataLoader of (inputs, labels).
        disable_std: pass-through to the model forward to turn STD off.

    Returns:
        tuple: (accuracy, mean sparsity) where sparsity is the model-reported
        KC activity fraction averaged over batches.
    """
    model.eval()
    correct, total, sparsities = 0, 0, []
    with torch.no_grad():
        for bx, by in test_loader:
            # disable_std is consumed directly by the model's forward here.
            logits, info = model(bx, return_all=True, disable_std=disable_std)
            correct += (logits.argmax(-1) == by).sum().item()
            total += len(by)
            sparsities.append(info['sparsity'])
    return correct / total, np.mean(sparsities)


def compute_mean_sim_decorr(model, or_responses, disable_std=False):
    """Mean-similarity decorrelation wrapper, optionally with STD disabled."""
    if disable_std:
        _patch_model_disable_std(model)
    r = compute_mean_sim_decorrelation(model, or_responses)
    if disable_std:
        _unpatch_model(model)
    return r


# ============================================================================
# MODEL PATCHING HELPERS
# Temporarily wrap model.forward so all downstream analysis calls (decorr,
# Mancini, conc-invariance) automatically use disable_std=True.
# ============================================================================

def _patch_model_disable_std(model):
    """Monkeypatch model.forward to always pass disable_std=True.

    Many analysis helpers call ``model.forward`` internally without exposing a
    disable_std flag. To force STD off for an entire analysis without editing
    those helpers, we replace ``model.forward`` with a wrapper that injects
    ``disable_std=True`` (via setdefault, so explicit values still win). Idempotent:
    a second call is a no-op while already patched. The original is stashed on
    ``model._orig_forward_std`` for :func:`_unpatch_model` to restore.
    """
    if hasattr(model, '_orig_forward_std'):
        return  # already patched
    orig = model.forward

    def _patched(*args, **kwargs):
        kwargs.setdefault('disable_std', True)
        return orig(*args, **kwargs)

    model._orig_forward_std = orig
    model.forward = _patched


def _unpatch_model(model):
    """Restore original forward after patching.

    Reverses :func:`_patch_model_disable_std`; no-op if the model was not patched.
    """
    if hasattr(model, '_orig_forward_std'):
        model.forward = model._orig_forward_std
        del model._orig_forward_std


# ============================================================================
# COLLECT METRICS
# ============================================================================

def collect_metrics(seed, student, test_loader, or_responses, disable_std):
    """Run full evaluation suite and return results dict.

    The STD-path counterpart of the trained-path final evaluation. Computes
    accuracy, centroid accuracy, sparsity, per-pair decorrelation, the Mancini
    ratio (passing band 1.5..2.5), and concentration invariance — all with STD
    optionally disabled — then reads out the learned biophysical parameters
    (gap-junction conductances, LN->PN split, g_soma, APL gain, KC->APL strength,
    nonad strengths) recovered from their log-space storage.

    Args:
        seed: RNG seed (recorded and used for the concentration sweep).
        student: trained spiking model, eval mode.
        test_loader: held-out DataLoader.
        or_responses: clean OR responses for the analyses.
        disable_std: whether to run all evaluation with STD off.

    Returns:
        dict: full metrics record for this seed.
    """
    student.eval()
    ds = disable_std

    test_acc, sparsity = _std_evaluate_model(student, test_loader, disable_std=ds)
    cent_acc = centroid_accuracy(student, or_responses, disable_std=ds)
    pp_decorr = compute_per_pair_decorrelation(student, or_responses, disable_std=ds)
    mancini = run_mancini(student, disable_std=ds)
    mancini_pass = 1.5 <= mancini <= 2.5  # canonical Mancini pass band
    conc_results, conc_tests = run_concentration_invariance(student, or_responses, seed, disable_std=ds)

    # Recover gap-junction conductances (S) from log space; ln_ln may be absent.
    g_gap_ln = float(np.exp(student.antennal_lobe.log_g_gap_ln.item())) \
        if student.antennal_lobe.log_g_gap_ln is not None else None
    g_gap_pn = float(np.exp(student.antennal_lobe.log_g_gap_pn.item()))
    g_gap_eln = float(np.exp(student.antennal_lobe.log_g_gap_eln_pn.item()))
    # LN->PN inhibitory and excitatory synapse strengths (S).
    ln_pn_inhib_str = float(np.exp(student.antennal_lobe.ln_pn.log_strength.item()))
    ln_pn_excit_str = float(np.exp(student.antennal_lobe.ln_pn_excit.log_strength.item()))
    g_soma = np.exp(student.kc_layer.kc_neurons.log_g_soma.item()) * 1e9   # nS
    apl_gain = F.softplus(torch.tensor(student.kc_layer.apl.apl_gain.item())).item()
    kc_apl_strength = float(np.exp(student.kc_layer.apl.kc_apl_log_strength.item()))

    # Recover all present developmental ("nonad") pathway strengths (S).
    nonad_strengths = {}
    for name, layer in [
        ('orn_ln_nonad', student.antennal_lobe.orn_ln_nonad),
        ('ln_pn_nonad', student.antennal_lobe.ln_pn_nonad),
        ('ln_pn_excit_nonad', student.antennal_lobe.ln_pn_excit_nonad),
        ('ln_ln_nonad', student.antennal_lobe.ln_ln_nonad),
        ('pn_ln_nonad', student.antennal_lobe.pn_ln_nonad),
        ('ln_orn_nonad', student.antennal_lobe.ln_orn_nonad),
        ('pn_kc_nonad', student.kc_layer.pn_kc_nonad),
    ]:
        if layer is not None:
            nonad_strengths[name] = float(np.exp(layer.log_strength.item()))

    print(f"    Acc: {test_acc:.1%}, Centroid: {cent_acc:.1%}, Sp: {sparsity:.1%}")
    print(f"    AL: {pp_decorr['al_decorr_pct']:.1f}%, MB: {pp_decorr['mb_decorr_pct']:.1f}%, "
          f"Total: {pp_decorr['total_decorr_pct']:.1f}%")
    print(f"    Mancini: {mancini:.2f} ({'PASS' if mancini_pass else 'FAIL'})")
    print(f"    Conc: SubPN={'P' if conc_tests['sublinear_pn_gain'] else 'F'}, "
          f"FlatKC={'P' if conc_tests['flat_kc_activity'] else 'F'}, "
          f"RobClass={'P' if conc_tests['robust_classification'] else 'F'}, "
          f"Identity={conc_tests['odor_identity_preservation']}")
    print(f"    g_soma: {g_soma:.1f} nS, APL gain: {apl_gain:.2f}, KC->APL: {kc_apl_strength:.2e}")

    return {
        'seed': seed,
        'accuracy': test_acc,
        'centroid_accuracy': cent_acc,
        'sparsity': sparsity,
        'per_pair_decorrelation': pp_decorr,
        'mancini': mancini,
        'mancini_pass': mancini_pass,
        'concentration_invariance': {
            'per_concentration': conc_results,
            # keep only JSON-serialisable scalar summary fields
            'predictions': {k: v for k, v in conc_tests.items()
                            if isinstance(v, (bool, float, int, str))},
        },
        'g_soma_nS': g_soma,
        'apl_gain_effective': apl_gain,
        'kc_apl_strength': kc_apl_strength,
        'gap_junction_conductances': {'ln_ln': g_gap_ln, 'pn_pn': g_gap_pn, 'eln_pn': g_gap_eln},
        'ln_pn_split': {'inhibitory_strength': ln_pn_inhib_str, 'excitatory_strength': ln_pn_excit_str},
        'nonad_strengths': nonad_strengths,
    }


# ============================================================================
# CONDITION 1: NO-STD TRAINING
# ============================================================================

def run_no_std_training(data_dir, train_loader, test_loader, n_odors, or_responses, output_dir):
    """Train from scratch with STD disabled throughout.

    For each seed: train the rate teacher, build + warm-start a spiking student
    (same initialisation as :func:`initialize_student` but inline and ablation-
    free), then train ``STUDENT_EPOCHS`` always passing ``disable_std=True`` so
    Tsodyks-Markram depression never acts. Snapshots state at epoch 300, scores
    it, and writes per-seed checkpoints plus a combined summary.

    Args:
        data_dir: dataset / connectome root.
        train_loader, test_loader: DataLoaders.
        n_odors: number of odor classes.
        or_responses: clean OR responses for analyses.
        output_dir: where to save checkpoints + results.json.

    Returns:
        list[dict]: per-seed metrics.

    NOTE: iterates the module-level ``SEEDS`` (see file header caveat).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"NO-STD TRAINING (seed {seed})")
        print(f"{'='*70}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Per-checkpoint training-curve history (sampled every 50 epochs).
        history = {'train_acc': [], 'test_acc': [], 'sparsity': [],
                   'al_decorr': [], 'mb_decorr': [], 'mancini': [], 'g_soma': []}

        # --- Teacher (rate-based, unchanged) ---
        print(f"\n  Training teacher ({TEACHER_EPOCHS} epochs)...")
        teacher = ConnectomeConstrainedModel.from_data_dir(
            data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10)
        opt = torch.optim.Adam(teacher.parameters(), lr=1e-2)
        for ep in range(TEACHER_EPOCHS):
            teacher.train()
            for bx, by in train_loader:
                opt.zero_grad()
                loss, _ = teacher.compute_loss(bx, by, sparsity_weight=2.0)
                loss.backward()
                opt.step()
            if (ep + 1) % 100 == 0:
                teacher.eval()
                c, t = 0, 0
                with torch.no_grad():
                    for bx, by in test_loader:
                        c += (teacher(bx).argmax(-1) == by).sum().item()
                        t += len(by)
                print(f"    Teacher epoch {ep+1}: {c/t:.1%}")

        # --- Student (STD disabled for all forward calls) ---
        print(f"\n  Setting up student (STD DISABLED, all connections)...")
        student = SpikingConnectomeConstrainedModel.from_data_dir(
            data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10,
            params=REALISTIC_PARAMS, include_nonad=True)
        student.n_steps_al = N_STEPS
        student.n_steps_kc = N_STEPS

        # Inline warm start (mirrors initialize_student, no ablations). Note this
        # uses the undefined global APL_BOOST (see file header) rather than
        # APL_BOOST_DEFAULT.
        with torch.no_grad():
            for name, param in student.named_parameters():
                if 'v_th' in name:
                    param.fill_(LN_VTH_INIT if 'ln' in name else -0.0425)
            student.decoder.weight.copy_(teacher.decoder.weight)
            student.decoder.bias.copy_(teacher.decoder.bias)
            student.or_to_orn.or_gains.copy_(teacher.or_to_orn.or_gains)
            student.kc_layer.apl.apl_gain.data = teacher.kc_layer.apl.apl_gain.data.clone() * APL_BOOST
            student.kc_layer.kc_neurons.v_th.data += 0.005
            student.kc_layer.kc_kc_aa.log_strength.fill_(np.log(1e-11))
            if student.kc_layer.kc_kc_ad is not None:
                student.kc_layer.kc_kc_ad.log_strength.fill_(np.log(1e-13))
            student.kc_layer.kc_neurons.log_g_soma.fill_(np.log(10e-9))
            if student.antennal_lobe.ln_ln is not None:
                student.antennal_lobe.ln_ln.log_strength.fill_(np.log(1e-13))
            if student.antennal_lobe.pn_ln is not None:
                student.antennal_lobe.pn_ln.log_strength.fill_(np.log(1e-13))
            if student.antennal_lobe.ln_orn is not None:
                student.antennal_lobe.ln_orn.log_strength.fill_(np.log(1e-13))
            ln_pn_orig = student.antennal_lobe.ln_pn.log_strength.item()
            student.antennal_lobe.ln_pn.log_strength.fill_(ln_pn_orig + np.log(LN_PN_SCALE))
            ln_pn_inhib_strength = np.exp(student.antennal_lobe.ln_pn.log_strength.item())
            student.antennal_lobe.ln_pn_excit.log_strength.fill_(np.log(ln_pn_inhib_strength * 0.1))
            orn_pn_orig = student.antennal_lobe.orn_pn.log_strength.item()
            student.antennal_lobe.orn_pn.log_strength.fill_(orn_pn_orig + np.log(ORN_PN_SCALE))
            for name, attr in [
                ('orn_ln_nonad', student.antennal_lobe.orn_ln_nonad),
                ('ln_pn_nonad', student.antennal_lobe.ln_pn_nonad),
                ('ln_pn_excit_nonad', student.antennal_lobe.ln_pn_excit_nonad),
                ('ln_ln_nonad', student.antennal_lobe.ln_ln_nonad),
                ('pn_ln_nonad', student.antennal_lobe.pn_ln_nonad),
                ('ln_orn_nonad', student.antennal_lobe.ln_orn_nonad),
                ('pn_kc_nonad', student.kc_layer.pn_kc_nonad),
            ]:
                if attr is not None:
                    attr.log_strength.fill_(NONAD_INIT)

        print(f"\n  Training student ({STUDENT_EPOCHS} epochs, STD DISABLED)...")
        param_groups = _std_param_groups(student)
        optimizer = torch.optim.Adam(param_groups)
        ep300_state = None  # snapshot of weights at epoch 300 (the reported point)

        for epoch in range(STUDENT_EPOCHS):
            # Sparsity-loss ramp over the first 60 epochs (literal 60 here,
            # matching the canonical ENERGY_RAMP_EPOCHS value).
            progress = min(1.0, epoch / 60)
            sp_w = progress * MAX_SP_WEIGHT

            student.train()
            train_correct, train_total = 0, 0
            for bx, by in train_loader:
                optimizer.zero_grad()
                # *** STD DISABLED during training ***
                logits, info = student(bx, return_all=True, disable_std=True)
                ce_loss = F.cross_entropy(logits, by)
                kc_rates = info['kc_spikes'] / N_STEPS
                # Same canonical KC-sparsity penalty (offset 0.02, target 0.05).
                sp_loss = (torch.sigmoid((kc_rates - 0.02) * 50).mean() - 0.05) ** 2
                (ce_loss + sp_w * sp_loss).backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
                optimizer.step()
                clamp_biological(student)
                train_correct += (logits.argmax(-1) == by).sum().item()
                train_total += len(by)

            if (epoch + 1) % 50 == 0:
                # Periodic STD-off diagnostics, appended to history.
                student.eval()
                test_acc, sparsity = _std_evaluate_model(student, test_loader, disable_std=True)
                decorr = compute_mean_sim_decorr(student, or_responses, disable_std=True)
                mancini = run_mancini(student, disable_std=True)
                g_soma = np.exp(student.kc_layer.kc_neurons.log_g_soma.item()) * 1e9

                history['train_acc'].append(train_correct / train_total)
                history['test_acc'].append(test_acc)
                history['sparsity'].append(sparsity)
                history['al_decorr'].append(decorr['al_decorr'])
                history['mb_decorr'].append(decorr['mb_decorr'])
                history['mancini'].append(mancini)
                history['g_soma'].append(g_soma)

                print(f"  Ep {epoch+1}: Train={train_correct/train_total:.1%}, Test={test_acc:.1%}, "
                      f"Sp={sparsity:.1%}, AL={decorr['al_decorr']:.1f}%, MB={decorr['mb_decorr']:.1f}%, "
                      f"Manc={mancini:.2f}")

            if (epoch + 1) == 300:
                # Capture the canonical epoch-300 weights for final scoring.
                ep300_state = {k: v.clone() for k, v in student.state_dict().items()}

        # Evaluate at epoch 300 with STD disabled
        student.load_state_dict(ep300_state)
        student.eval()
        print(f"\n  --- Seed {seed} Epoch 300 Evaluation (no_std_train) ---")
        results = collect_metrics(seed, student, test_loader, or_responses, disable_std=True)
        results['history'] = history

        torch.save(student.state_dict(), output_dir / f'model_seed{seed}.pt')
        all_results.append(results)

    _save_and_print_summary('no_std_train', all_results, output_dir)
    return all_results


# ============================================================================
# CONDITION 2: POST-HOC STD REMOVAL
# ============================================================================

def run_posthoc_std_off(data_dir, test_loader, n_odors, or_responses, output_dir):
    """Load canonical models; evaluate with STD disabled.

    The cheaper STD condition: take the already-trained canonical checkpoints and
    simply re-score them with ``disable_std=True`` (no retraining). Tests how much
    the trained behaviour depends on STD being active at inference.

    Args:
        data_dir: dataset / connectome root.
        test_loader: held-out DataLoader.
        n_odors: number of odor classes.
        or_responses: clean OR responses for analyses.
        output_dir: where to write results.json.

    Returns:
        list[dict]: per-seed metrics.

    Raises:
        FileNotFoundError: if the canonical model directory is missing.

    NOTE: iterates the module-level ``SEEDS`` (see file header caveat).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not CANONICAL_DIR.exists():
        raise FileNotFoundError(
            f"Canonical model directory not found: {CANONICAL_DIR}\n"
            "Run run_training.py first to generate canonical models.")

    all_results = []

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"POST-HOC STD OFF — loading canonical seed {seed}")
        print(f"{'='*70}")

        model = SpikingConnectomeConstrainedModel.from_data_dir(
            data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10,
            params=REALISTIC_PARAMS, include_nonad=True)
        model.load_state_dict(torch.load(
            CANONICAL_DIR / f'model_seed{seed}.pt', weights_only=True))
        model.n_steps_al = N_STEPS
        model.n_steps_kc = N_STEPS
        model.eval()

        print(f"  Loaded. Evaluating with STD DISABLED...")
        results = collect_metrics(seed, model, test_loader, or_responses, disable_std=True)
        all_results.append(results)

    _save_and_print_summary('posthoc_std_off', all_results, output_dir)
    return all_results


# ============================================================================
# SUMMARY + SAVE
# ============================================================================

def _save_and_print_summary(condition, all_results, output_dir):
    """Write the STD-condition results to JSON and print a per-seed + aggregate table.

    Args:
        condition: condition name (e.g. 'no_std_train', 'posthoc_std_off').
        all_results: list of per-seed metric dicts from :func:`collect_metrics`.
        output_dir: destination directory (results.json written here).

    Side effects: writes results.json; prints the summary table (mean/std rows
    and the Mancini pass count).

    NOTE: uses the module-level ``SEEDS`` for the seed count (see file header).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")

    # Pull each metric across seeds into parallel lists for aggregation.
    accs   = [r['accuracy'] for r in all_results]
    cents  = [r['centroid_accuracy'] for r in all_results]
    sps    = [r['sparsity'] for r in all_results]
    al_ds  = [r['per_pair_decorrelation']['al_decorr_pct'] for r in all_results]
    mb_ds  = [r['per_pair_decorrelation']['mb_decorr_pct'] for r in all_results]
    tot_ds = [r['per_pair_decorrelation']['total_decorr_pct'] for r in all_results]
    mancs  = [r['mancini'] for r in all_results]
    gsomas = [r['g_soma_nS'] for r in all_results]

    print(f"\n{'='*80}")
    print(f"STD ABLATION — {condition.upper()} — SUMMARY ({len(SEEDS)} seeds)")
    print(f"{'='*80}")
    print(f"{'Seed':<6} {'Acc':<8} {'Cent':<8} {'Sp':<8} {'AL%':<8} {'MB%':<8} {'Tot%':<8} {'Manc':<8} {'g_soma':<8}")
    print("-" * 72)
    for r in all_results:
        pp = r['per_pair_decorrelation']
        m_ok = "P" if r['mancini_pass'] else "F"
        print(f"{r['seed']:<6} {r['accuracy']:<8.1%} {r['centroid_accuracy']:<8.1%} "
              f"{r['sparsity']:<8.1%} "
              f"{pp['al_decorr_pct']:<8.1f} {pp['mb_decorr_pct']:<8.1f} {pp['total_decorr_pct']:<8.1f} "
              f"{r['mancini']:.2f}{m_ok:<3} {r['g_soma_nS']:<8.1f}")
    print("-" * 72)
    # Mean and std (decorr % printed as-is; accuracy/sparsity std scaled to %).
    print(f"{'Mean':<6} {np.mean(accs):<8.1%} {np.mean(cents):<8.1%} {np.mean(sps):<8.1%} "
          f"{np.mean(al_ds):<8.1f} {np.mean(mb_ds):<8.1f} {np.mean(tot_ds):<8.1f} "
          f"{np.mean(mancs):<8.2f} {np.mean(gsomas):<8.1f}")
    print(f"{'Std':<6} {np.std(accs)*100:<8.1f} {np.std(cents)*100:<8.1f} {np.std(sps)*100:<8.1f} "
          f"{np.std(al_ds):<8.1f} {np.std(mb_ds):<8.1f} {np.std(tot_ds):<8.1f} "
          f"{np.std(mancs):<8.2f} {np.std(gsomas):<8.1f}")
    n_pass = sum(1 for r in all_results if r['mancini_pass'])
    print(f"\nMancini: {n_pass}/{len(SEEDS)} pass")


# ============================================================================
# MAIN
# ============================================================================

def _main_std():
    """CLI entry point for the ``std`` subcommand (STD ablation study).

    Parses --condition (no_std_train / posthoc_std_off / both) and an optional
    --seed override, locates data, builds loaders, and runs the requested STD
    condition(s). The --seed override rebinds the module-level ``SEEDS`` to a
    single seed (handy for parallelising seeds across terminals or resuming).
    """
    parser = argparse.ArgumentParser(description='STD Ablation Study')
    parser.add_argument('--condition', choices=['no_std_train', 'posthoc_std_off', 'both'],
                        default='both',
                        help='Which condition to run (default: both)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Run only this seed (default: all 5 seeds). '
                             'Useful for parallelising across terminal windows.')
    args = parser.parse_args()

    # Locate the dataset directory.
    _pkg_root = Path(__file__).resolve().parent.parent.parent
    _parent = _pkg_root.parent
    _data_candidates = [
        _pkg_root / 'data',
    ]
    data_dir = next((p for p in _data_candidates if (p / 'kreher2008').is_dir()), None)
    if data_dir is None:
        raise FileNotFoundError('Cannot find connectome data (kreher2008/).')

    STD_ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    # Allow single-seed override (useful when running seeds in parallel terminals
    # or when a long run was interrupted and you want to resume from a specific seed).
    if args.seed is not None:
        global SEEDS
        SEEDS = [args.seed]

    print("=" * 80)
    print("STD ABLATION STUDY")
    print("=" * 80)
    print(f"\nCondition : {args.condition}")
    print(f"Seeds     : {SEEDS}")
    print(f"Output    : {STD_ABLATION_DIR}")
    print(f"\nSTD model : Tsodyks-Markram (tau_rec ~200ms, U ~0.3)")
    print(f"  no_std_train   — train from scratch, disable_std=True always")
    print(f"  posthoc_std_off — load canonical weights, evaluate disable_std=True")

    # Build datasets/loaders and the clean OR-response matrix (shared by both
    # conditions below).
    train_dataset, test_dataset, odor_names = load_kreher2008_all_odors(
        data_dir, train_repeats=10, test_repeats=5,
        noise_std=NOISE_STD, noise_type=NOISE_TYPE)
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size=16)
    n_odors = len(odor_names)

    df = pd.read_csv(data_dir / "kreher2008/orn_responses_normalized.csv", index_col=0)
    or_responses = torch.from_numpy(df.values).float()

    # Condition 2 (cheaper): re-score canonical checkpoints with STD off.
    if args.condition in ('posthoc_std_off', 'both'):
        print(f"\n{'#'*70}")
        print("CONDITION: POST-HOC STD REMOVAL (canonical weights, STD disabled at eval)")
        print(f"{'#'*70}")
        run_posthoc_std_off(
            data_dir, test_loader, n_odors, or_responses,
            STD_ABLATION_DIR / 'posthoc_std_off')

    # Condition 1 (expensive): retrain from scratch with STD off throughout.
    if args.condition in ('no_std_train', 'both'):
        print(f"\n{'#'*70}")
        print("CONDITION: NO-STD TRAINING (train from scratch, STD disabled throughout)")
        print(f"{'#'*70}")
        run_no_std_training(
            data_dir, train_loader, test_loader, n_odors, or_responses,
            STD_ABLATION_DIR / 'no_std_train')

    print(f"\n{'='*80}")
    print("STD ABLATION COMPLETE")
    print(f"Results: {STD_ABLATION_DIR}")
    print(f"{'='*80}")



def main():
    """Subcommand dispatch:  (none)->trained ablation | posthoc->post-hoc | std->STD ablation."""
    # Pop the subcommand token (if any) so the sub-main's argparse sees a clean
    # argv, then dispatch to the matching entry point.
    if len(sys.argv) > 1 and sys.argv[1] == 'posthoc':
        sys.argv.pop(1); _main_posthoc()
    elif len(sys.argv) > 1 and sys.argv[1] == 'std':
        sys.argv.pop(1); _main_std()
    else:
        _main_trained()


if __name__ == '__main__':
    main()
