"""
Canonical training script: All-Connections Non-AD (5 seeds x 300 epochs).

This is the CANONICAL training script for the paper. It trains 5 spiking models,
each from its own independently-trained rate-based teacher, using every synapse
in the Winding et al. 2023 connectome (AD + non-AD + gap junctions + realistic noise).

Non-AD connections (initialized weak at 1e-13 A, LR 0.05x):
  ORN→LN (182), LN→PN (582), LN→LN (605), PN→LN (518), PN→KC (3), LN→ORN (310)

Reports per seed at epoch 300: accuracy (linear + centroid), sparsity,
  per-pair decorrelation (AL + MB), Mancini, concentration invariance (4 tests).
Aggregates mean +/- std across 5 seeds.

Results saved to: results/all_connections_nonad_canonical/

Notebook section: Section A — Original Paper Figures (primary results source).

Usage:
    python -m run_training

----------------------------------------------------------------------------
WHERE THIS FILE SITS IN THE PIPELINE
----------------------------------------------------------------------------
The full forward model (defined in model.py / layers.py) is:

  OR responses (Kreher 2008) -> ORN (LIF) -> LN (LIF) -> PN (LIF)
      -> KC (two-compartment LIF) <- APL (graded divisive inhibition)
      -> linear decoder over 28 odors

Connectivity is fixed by the Winding 2023 connectome; only ~449 biophysical
parameters are learned. This script performs the ANN-to-SNN knowledge transfer
that the pipeline depends on:

  1. Train a fast rate-based "teacher" (ConnectomeConstrainedModel) end-to-end.
  2. Build a spiking "student" (SpikingConnectomeConstrainedModel) on the same
     connectome, seed it from the teacher (decoder, OR gains, APL gain), and
     train it with surrogate-gradient backprop-through-time.

This module exposes TWO entry points behind a single CLI (see main()):
  * _main_canonical(): the bare 5-seed canonical run (no CLI args).
  * _main_energy():    energy/sparsity-loss ablation variants (CLI args), merged
                       here from the former run_training_energy_only.py.

IMPORTANT (per repo conventions): the canonical runtime constants are duplicated
here in the driver and OVERRIDE model.py's in-class defaults. In particular
N_STEPS=30 (matching model.py's default), the sparsity-loss offset 0.02 /
target 0.05, and the g_soma clamp [1, 20] nS are authoritative HERE.
"""
import sys
from pathlib import Path
# Make the package importable when run as a plain script (not just `python -m ...`):
# add the grandparent of this file (the directory that contains the
# code/ (which holds core/, analysis/, scripts/) to sys.path.
_pkg_parent = str(Path(__file__).parent.parent)
if _pkg_parent not in sys.path:
    sys.path.insert(0, _pkg_parent)

# Flush stdout after every line so long training runs stream progress live
# (important when piping logs to a file or a background job).
if hasattr(sys.stdout, 'reconfigure'):  # absent under Jupyter's OutStream
    sys.stdout.reconfigure(line_buffering=True)

import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend: render figures to PNG files, no display needed
import matplotlib.pyplot as plt

from core.model import SpikingConnectomeConstrainedModel
from core.layers import SpikingParams
# Canonical metric implementations live in the analysis subpackage; thin wrappers
# below adapt their return shapes / inject the noise level used in this script.
from analysis.compute import (
    compute_per_pair_decorrelation as _compute_per_pair_decorrelation,
    compute_mean_sim_decorrelation, run_mancini_test as _run_mancini_test,
    run_concentration_invariance as _run_concentration_invariance,
    centroid_accuracy as _centroid_accuracy,
)
from core.rate_model import ConnectomeConstrainedModel  # rate-based teacher
from core.dataset import load_kreher2008_all_odors, create_dataloaders

# ============================================================================
# CONFIGURATION (identical to realistic_noise_canonical + non-AD)
# ============================================================================
# KC somatic coupling conductance clamp (SI siemens): 1 nS .. 20 nS. This bounds
# the learnable g_soma that couples the two KC compartments to a biophysically
# plausible range; values are stored in log space (see clamp below).
G_SOMA_MIN, G_SOMA_MAX_BIO = 1e-9, 20e-9
# Lower clamp for KC-KC synaptic log-strength: log(1 fA) — synapses can be driven
# essentially "off" but never to log(0) = -inf (which would be unstable).
KCKC_LOG_MIN = np.log(1e-15)
# Upper clamp for chemical-synapse log-strength: log(10 nA) in current units.
LOG_STRENGTH_MAX = np.log(1e-8)

APL_BOOST = 4.0          # Multiplier applied to the teacher's APL gain when seeding the student (stronger inhibition in spikes)
TEACHER_EPOCHS = 300     # Rate-teacher training epochs
STUDENT_EPOCHS = 300     # Spiking-student training epochs
MAX_SP_WEIGHT = 15.0     # Peak weight of the KC sparsity loss after the warm-up ramp
BASE_LR = 1e-3           # Base Adam learning rate; per-parameter multipliers applied in get_param_groups()
KC_VTH_LR = 0.01         # LR multiplier for KC firing thresholds
LN_VTH_LR = 0.01         # LR multiplier for LN firing thresholds
ORN_VTH_LR = 0.01        # LR multiplier for ORN firing thresholds
PN_VTH_LR = 0.01         # LR multiplier for PN firing thresholds
GRAD_CLIP = 5.0          # Global grad-norm clip (stabilizes BPTT through 30 time steps)

LN_VTH_INIT = -0.0475    # Initial LN firing threshold (V, i.e. -47.5 mV); lower than other types so LNs spike readily
LN_PN_SCALE = 1.2        # Multiplicative boost to the (inhibitory) LN->PN synaptic strength at init
ORN_PN_SCALE = 0.7       # Multiplicative attenuation to the ORN->PN synaptic strength at init
N_STEPS = 30             # Simulated time steps per stimulus for BOTH AL and KC loops (canonical; matches model.py's default)

# Per-parameter-group LR multipliers (applied on top of BASE_LR). These shape the
# optimization so that fast/sensitive params (KC, APL) learn quickly while AL
# params and non-AD synapses learn slowly to avoid destabilizing the circuit.
AL_LR = 0.2
NONAD_LR = 0.05  # Non-AD connections: weak LR to avoid disrupting canonical
KC_LR = 4.0
KCKC_LR = 0.1
GSOMA_LR = 0.1
APL_TAU_LR = 0.05

# Stimulus / OR-response noise injected at the dataset level (input variability,
# distinct from the per-circuit noise sources in REALISTIC_PARAMS below).
NOISE_TYPE = 'multiplicative'
NOISE_STD = 0.3  # 30% CV (Kreher 2008 inter-fly variability)

# Boosted circuit noise (biologically realistic)
# Six biological noise sources injected inside the spiking circuit each step.
# Units: V for membrane/threshold, A for background current, dimensionless CV for
# synaptic/receptor multiplicative noise.
REALISTIC_PARAMS = SpikingParams(
    v_noise_std=1.0e-3,           # 1.0 mV  — Gaussian membrane-voltage noise
    i_noise_std=15e-12,           # 15 pA   — background input-current noise
    syn_noise_std=0.25,           # 25% CV  — multiplicative synaptic-release noise
    threshold_jitter_std=1.0e-3,  # 1.0 mV  — per-step firing-threshold jitter
    orn_receptor_noise_std=0.10,  # 10% CV  — ORN receptor (transduction) noise
    circuit_noise_enabled=True,   # master switch enabling the above
)

SEEDS = [42, 43, 44, 45, 46]  # 5 independent RNG seeds -> 5 teacher/student pairs for mean +/- std stats
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / 'results' / 'all_connections_nonad_canonical'

# Non-AD initialization
# Non-axon-dendrite (non-AD) connectome synapses are initialized essentially off
# so they only contribute if learning actively recruits them.
NONAD_INIT = np.log(1e-13)  # Very weak: ~0.1 fA

# Concentration invariance
# Sweep of relative odor concentrations used by the concentration-invariance test.
CONCENTRATIONS = [0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
HILL_EC50 = 1.0   # Half-max concentration of the Hill dose-response used to scale OR drive
HILL_N = 1        # Hill coefficient (n=1 -> simple saturating Michaelis-Menten form)
N_CONC_TRIALS = 10  # Noisy trials averaged per concentration


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def clamp_biological(model):
    """Project all learnable parameters back into biophysically valid ranges.

    Called after every optimizer step so gradient updates never push a parameter
    outside its physical bounds. Operates in-place under no_grad.

    Steps:
      1. model.clamp_to_biological_bounds(): clamps tau_m, g_soma, base synapse
         log-strengths, STD time constants, etc. (defined per-layer).
      2. Additionally clamp the KC-KC, KC soma, KC-KC(AD), and PN->KC(non-AD)
         log-parameters that live in the KC layer to this script's bounds.

    Args:
        model: a SpikingConnectomeConstrainedModel (the student).

    Side effects:
        Mutates the model's parameters in place; returns nothing.
    """
    model.clamp_to_biological_bounds()
    with torch.no_grad():
        # KC<->KC axo-axonic synapse strength (log A) into [log(1 fA), log(10 nA)]
        model.kc_layer.kc_kc_aa.log_strength.clamp_(KCKC_LOG_MIN, LOG_STRENGTH_MAX)
        # KC somatic coupling conductance (log S) into log([1 nS, 20 nS])
        model.kc_layer.kc_neurons.log_g_soma.clamp_(np.log(G_SOMA_MIN), np.log(G_SOMA_MAX_BIO))
        if model.kc_layer.kc_kc_ad is not None:
            # KC<->KC axo-dendritic synapse strength, same bounds
            model.kc_layer.kc_kc_ad.log_strength.clamp_(KCKC_LOG_MIN, LOG_STRENGTH_MAX)
        if model.kc_layer.pn_kc_nonad is not None:
            # Non-AD PN->KC synapse strength, same bounds
            model.kc_layer.pn_kc_nonad.log_strength.clamp_(KCKC_LOG_MIN, LOG_STRENGTH_MAX)


def get_param_groups(model):
    """Build per-parameter Adam parameter groups with custom learning rates.

    Each trainable parameter is assigned a learning rate BASE_LR * mult, where the
    multiplier depends on which part of the circuit the parameter belongs to
    (matched by substring of its dotted name). This lets sensitive/fast structures
    (KC, APL) learn aggressively while AL and weak non-AD synapses learn slowly.

    Args:
        model: the student model whose named_parameters() are inspected.

    Returns:
        list[dict]: Adam param_groups, each {'params': [p], 'lr': BASE_LR*mult},
        one group per trainable parameter (so every param can have its own LR).
    """
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen params (e.g. registered buffers / fixed connectome)
        if 'log_tau_rec' in name or 'logit_U' in name:
            # Tsodyks-Markram short-term-depression params (recovery tau, utilization U)
            mult = GSOMA_LR
        elif 'nonad' in name:
            mult = NONAD_LR  # Non-AD connections: weak LR
        elif 'kc_kc_aa' in name or 'kc_kc_ad' in name:
            mult = KCKC_LR   # KC-KC chemical synapses
        elif 'kc_kc_dd_gain' in name or 'kc_kc_da_gain' in name:
            mult = KCKC_LR   # KC-KC dendrite-dendrite / dendrite-axon gains
        elif 'log_g_soma' in name:
            mult = GSOMA_LR  # KC two-compartment coupling conductance
        elif 'log_g_gap' in name:
            mult = AL_LR     # Gap-junction conductances (LN-LN, PN-PN, eLN-PN)
        elif 'ln_pn_excit' in name:
            mult = AL_LR     # Excitatory LN->PN (Picky-LN) synapse
        elif 'ln_ln' in name or 'pn_ln' in name:
            mult = AL_LR     # AL recurrent synapses
        elif 'v_th' in name:
            # Firing thresholds: per-cell-type LR (typically slow, 0.01x)
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
            mult = 0.5       # OR->ORN receptor gains
        elif 'orn_neurons' in name or 'ln_neurons' in name or 'pn_neurons' in name or 'antennal_lobe' in name:
            mult = AL_LR     # Remaining AL neuron/layer params
        elif 'log_tau_apl' in name:
            mult = APL_TAU_LR  # APL integration time constant
        elif 'kc_layer' in name or 'kc_neurons' in name or 'apl' in name:
            mult = KC_LR     # Catch-all for remaining KC / APL params (fast LR)
        else:
            mult = 1.0       # Default: decoder, biases, anything unmatched
        param_groups.append({'params': [param], 'lr': BASE_LR * mult})
    return param_groups


# Delegate to analysis subpackage (canonical implementations)
def compute_per_pair_decorrelation(model, or_responses, n_trials=10):
    """Per-odor-pair decorrelation, re-keyed for this script's reporting.

    Thin wrapper over analysis.compute.compute_per_pair_decorrelation that injects
    the script-level NOISE_STD and renames the returned keys to the names used by
    the logging/JSON code below.

    Args:
        model: spiking student model.
        or_responses: tensor [n_odors, n_or_types], normalized OR responses.
        n_trials: noisy repeats per odor used to estimate representational similarity.

    Returns:
        dict with keys:
          'kc_or', 'kc_pn', 'pn_or'    — similarity ratios between stages,
          'total_decorr_pct'           — overall decorrelation (%),
          'mb_decorr_pct'              — mushroom-body (PN->KC) decorrelation (%),
          'al_decorr_pct'              — antennal-lobe (OR->PN) decorrelation (%).
    """
    r = _compute_per_pair_decorrelation(model, or_responses, n_trials, NOISE_STD)
    return {
        'kc_or': r['kc_or_ratio'], 'kc_pn': r['kc_pn_ratio'], 'pn_or': r['pn_or_ratio'],
        'total_decorr_pct': r['total_decorr'], 'mb_decorr_pct': r['mb_decorr'],
        'al_decorr_pct': r['al_decorr'],
    }


def run_mancini(model, carbachol=1e-10, apl_inject=0.7):
    """Run the Mancini APL-inhibition test and return its spike-count ratio.

    The Mancini test compares KC activity with vs without an APL perturbation
    (here a carbachol drive and an injected APL activation), quantifying how
    strongly the APL feedback loop gates KC sparseness.

    Args:
        model: spiking student model.
        carbachol: simulated carbachol current (A) exciting the circuit.
        apl_inject: fractional APL activation injected (dimensionless).

    Returns:
        float: the boosted/baseline spike-count ratio (target window ~1.5-2.5).
    """
    result = _run_mancini_test(model, carbachol, apl_inject)
    return result['ratio']


def evaluate_model(model, test_loader):
    """Evaluate linear-decoder accuracy and mean KC sparsity on the test set.

    Runs the full spiking forward pass (no grad) over the test loader.

    Args:
        model: spiking student model.
        test_loader: DataLoader yielding (batch_x, batch_y) pairs.

    Returns:
        tuple(float, float): (top-1 accuracy in [0,1], mean KC sparsity in [0,1]).

    Side effects:
        Sets the model to eval() mode (disables surrogate gradients / hard spikes).
    """
    model.eval()
    correct, total, sparsities = 0, 0, []
    with torch.no_grad():
        for bx, by in test_loader:
            logits, info = model(bx, return_all=True)
            correct += (logits.argmax(-1) == by).sum().item()
            total += len(by)
            sparsities.append(info['sparsity'])
    return correct / total, np.mean(sparsities)


def centroid_accuracy(model, or_responses, n_trials=20):
    """Centroid (nearest-class-mean) accuracy of the KC representation.

    Wrapper over analysis.compute.centroid_accuracy that injects NOISE_STD. Unlike
    the trained linear decoder, this classifies by nearest class centroid in KC
    space, measuring how linearly separable the learned representation is.

    Args:
        model: spiking student model.
        or_responses: tensor [n_odors, n_or_types].
        n_trials: noisy repeats per odor.

    Returns:
        float: centroid-classification accuracy in [0,1].
    """
    from analysis.compute import centroid_accuracy as _centroid_accuracy
    return _centroid_accuracy(model, or_responses, n_trials, NOISE_STD)


def run_concentration_invariance(model, or_responses, seed):
    """Run the 4-part concentration-invariance test suite.

    Wrapper passing this script's concentration sweep, Hill parameters, trial
    count, and noise level to analysis.compute.run_concentration_invariance.

    Args:
        model: spiking student model.
        or_responses: tensor [n_odors, n_or_types].
        seed: RNG seed for reproducible noisy trials.

    Returns:
        tuple(conc_results, conc_tests): per-concentration numeric results and a
        dict of pass/fail predictions (sublinear PN gain, flat KC activity,
        robust classification, odor-identity preservation).
    """
    return _run_concentration_invariance(
        model, or_responses, seed, CONCENTRATIONS, HILL_EC50, HILL_N, N_CONC_TRIALS, NOISE_STD)


# ============================================================================
# TRAIN SINGLE MODEL
# ============================================================================
def train_single_model(seed, data_dir, train_loader, test_loader, n_odors, or_responses, output_dir, teacher_only=False):
    """Train one teacher->student pair end-to-end for a single seed.

    Pipeline for this seed:
      1. Train the rate-based teacher (ConnectomeConstrainedModel) for
         TEACHER_EPOCHS.
      2. Build the spiking student, seed it from the teacher and a hand-tuned
         biophysical initialization, then train it for STUDENT_EPOCHS with
         CE + ramped KC-sparsity loss and surrogate-gradient BPTT.
      3. Snapshot the model at epoch 300 and run the full metric battery.

    Args:
        seed: int RNG seed (also fixes torch + numpy).
        data_dir: Path to the connectome data directory (contains kreher2008/).
        train_loader, test_loader: DataLoaders of (x, y) odor batches.
        n_odors: number of output classes (decoder dimension).
        or_responses: tensor [n_odors, n_or_types] used by the analysis metrics.
        output_dir: Path where the trained student state_dict is saved.

    Returns:
        tuple(results, student):
          results: dict of all epoch-300 metrics + training history for this seed.
          student: the trained SpikingConnectomeConstrainedModel.

    Side effects:
        Prints progress; writes model_seed{seed}.pt to output_dir.
    """
    print(f"\n{'='*70}")
    print(f"TRAINING MODEL (seed {seed})")
    print(f"{'='*70}")

    torch.manual_seed(seed)  # Make teacher init, student init, and noise reproducible
    np.random.seed(seed)

    # History buffers: one entry appended at each periodic (every-50-epoch) eval.
    history = {
        'train_acc': [], 'test_acc': [], 'sparsity': [],
        'al_decorr': [], 'mb_decorr': [], 'mancini': [], 'g_soma': [],
    }

    # ---- TEACHER ----
    print(f"\nTraining teacher ({TEACHER_EPOCHS} epochs)...")
    # Build the rate-based teacher on the same connectome; target_sparsity drives
    # its sparsity regularizer toward ~10% active KCs.
    teacher = ConnectomeConstrainedModel.from_data_dir(data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10)
    opt = torch.optim.Adam(teacher.parameters(), lr=1e-2)
    teacher_acc = float('nan')  # final rate-teacher test accuracy (captured at last eval, saved to results)
    for ep in range(TEACHER_EPOCHS):
        teacher.train()
        for bx, by in train_loader:
            opt.zero_grad()
            # Teacher's own loss = cross-entropy + sparsity (weight 2.0)
            loss, _ = teacher.compute_loss(bx, by, sparsity_weight=2.0)
            loss.backward()
            opt.step()
        if (ep + 1) % 100 == 0:
            # Periodic teacher accuracy report (rate model -> argmax of logits)
            teacher.eval()
            c, t = 0, 0
            with torch.no_grad():
                for bx, by in test_loader:
                    c += (teacher(bx).argmax(-1) == by).sum().item()
                    t += len(by)
            teacher_acc = c / t  # keep the latest (epoch-300) teacher test accuracy
            print(f"  Teacher epoch {ep+1}: {teacher_acc:.1%}")

    # Persist the rate teacher NOW — right after teacher training, before the student
    # phase — so it is saved even if the student phase is interrupted, and so a
    # teacher_only run reproduces the EXACT canonical teacher without the (slow) student
    # retrain. teacher_seed{s} <-> model_seed{s} therefore stay an exact same-run pair
    # for the R2 drift analysis. Same CPU-cloned state_dict format as the other teacher
    # caches, so run_teacher_consistency loads it identically.
    teacher_state = {k: v.cpu().clone() for k, v in teacher.state_dict().items()}
    torch.save(teacher_state, output_dir / f'teacher_seed{seed}.pt')  # matched rate teacher
    if teacher_only:
        return None, teacher

    # ---- STUDENT (all connections + non-AD) ----
    print(f"\nSetting up student (all connections, realistic noise, APL {APL_BOOST}x)...")
    # Build the spiking student with realistic noise and the non-AD connectome
    # synapses included.
    student = SpikingConnectomeConstrainedModel.from_data_dir(
        data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10,
        params=REALISTIC_PARAMS, include_nonad=True)
    student.n_steps_al = N_STEPS  # Override legacy default: 30 AL time steps
    student.n_steps_kc = N_STEPS  # Override legacy default: 30 KC time steps

    with torch.no_grad():
        # Initialize all firing thresholds to a fixed resting offset; LNs get a
        # lower (more excitable) threshold so AL inhibition engages early.
        for name, param in student.named_parameters():
            if 'v_th' in name:
                if 'ln' in name:
                    param.fill_(LN_VTH_INIT)   # -47.5 mV
                else:
                    param.fill_(-0.0425)       # -42.5 mV
        # Transfer the teacher's learned readout + receptor gains to the student
        # (knowledge distillation of the linear decoder and OR->ORN mapping).
        student.decoder.weight.copy_(teacher.decoder.weight)
        student.decoder.bias.copy_(teacher.decoder.bias)
        student.or_to_orn.or_gains.copy_(teacher.or_to_orn.or_gains)
        # Seed APL gain from the teacher but scale it up by APL_BOOST: spiking KCs
        # need stronger divisive inhibition than the rate model to stay sparse.
        student.kc_layer.apl.apl_gain.data = teacher.kc_layer.apl.apl_gain.data.clone() * APL_BOOST
        # Nudge KC threshold up slightly (harder to spike -> sparser MB code).
        student.kc_layer.kc_neurons.v_th.data += 0.005
        # KC-KC axo-axonic synapse: moderate init (~10 pA in log A).
        student.kc_layer.kc_kc_aa.log_strength.fill_(np.log(1e-11))
        if student.kc_layer.kc_kc_ad is not None:
            # KC-KC axo-dendritic synapse: weak init (~0.1 fA).
            student.kc_layer.kc_kc_ad.log_strength.fill_(np.log(1e-13))
        # KC two-compartment coupling conductance init at 10 nS (mid-range of clamp).
        student.kc_layer.kc_neurons.log_g_soma.fill_(np.log(10e-9))
        # AL recurrent synapses start essentially off (~0.1 fA) and are learned up.
        if student.antennal_lobe.ln_ln is not None:
            student.antennal_lobe.ln_ln.log_strength.fill_(np.log(1e-13))
        if student.antennal_lobe.pn_ln is not None:
            student.antennal_lobe.pn_ln.log_strength.fill_(np.log(1e-13))
        if student.antennal_lobe.ln_orn is not None:
            student.antennal_lobe.ln_orn.log_strength.fill_(np.log(1e-13))
        # Boost the inhibitory LN->PN strength by LN_PN_SCALE (in log space this is
        # an additive log(1.2)).
        ln_pn_orig = student.antennal_lobe.ln_pn.log_strength.item()
        student.antennal_lobe.ln_pn.log_strength.fill_(ln_pn_orig + np.log(LN_PN_SCALE))
        # Set the excitatory (Picky-LN) LN->PN strength to 10% of the inhibitory one.
        ln_pn_inhib_strength = np.exp(student.antennal_lobe.ln_pn.log_strength.item())
        student.antennal_lobe.ln_pn_excit.log_strength.fill_(np.log(ln_pn_inhib_strength * 0.1))
        # Attenuate the direct ORN->PN drive by ORN_PN_SCALE (additive log(0.7)).
        orn_pn_orig = student.antennal_lobe.orn_pn.log_strength.item()
        student.antennal_lobe.orn_pn.log_strength.fill_(orn_pn_orig + np.log(ORN_PN_SCALE))

        # Initialize non-AD connections very weak
        # All non-AD connectome synapses start at NONAD_INIT (~0.1 fA): present but
        # effectively silent until learning (at the slow NONAD_LR) recruits them.
        if student.antennal_lobe.orn_ln_nonad is not None:
            student.antennal_lobe.orn_ln_nonad.log_strength.fill_(NONAD_INIT)
        if student.antennal_lobe.ln_pn_nonad is not None:
            student.antennal_lobe.ln_pn_nonad.log_strength.fill_(NONAD_INIT)
        if student.antennal_lobe.ln_pn_excit_nonad is not None:
            student.antennal_lobe.ln_pn_excit_nonad.log_strength.fill_(NONAD_INIT)
        if student.antennal_lobe.ln_ln_nonad is not None:
            student.antennal_lobe.ln_ln_nonad.log_strength.fill_(NONAD_INIT)
        if student.antennal_lobe.pn_ln_nonad is not None:
            student.antennal_lobe.pn_ln_nonad.log_strength.fill_(NONAD_INIT)
        if student.antennal_lobe.ln_orn_nonad is not None:
            student.antennal_lobe.ln_orn_nonad.log_strength.fill_(NONAD_INIT)
        if student.kc_layer.pn_kc_nonad is not None:
            student.kc_layer.pn_kc_nonad.log_strength.fill_(NONAD_INIT)

    # ---- TRAIN ----
    print(f"\nTraining student ({STUDENT_EPOCHS} epochs)...")
    param_groups = get_param_groups(student)  # per-param LR multipliers
    optimizer = torch.optim.Adam(param_groups)
    ep300_state = None  # snapshot of weights at epoch 300 (used for final eval)

    for epoch in range(STUDENT_EPOCHS):
        # Warm-up ramp: linearly raise the sparsity-loss weight from 0 to
        # MAX_SP_WEIGHT over the first 60 epochs so the model first learns to
        # classify before being pushed toward sparse KC codes.
        progress = min(1.0, epoch / 60)
        sp_w = progress * MAX_SP_WEIGHT

        student.train()
        train_correct, train_total = 0, 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits, info = student(bx, return_all=True)  # full spiking forward pass (30 steps)
            ce_loss = F.cross_entropy(logits, by)
            # KC firing rate per cell = spike count / N_STEPS (fraction of steps active).
            kc_rates = info['kc_spikes'] / N_STEPS
            # Sparsity loss (CANONICAL form): push the soft-thresholded mean KC
            # activation toward 0.05. sigmoid((rate-0.02)*50) is a smooth indicator
            # of "active" (offset 0.02, steepness 50); its mean is the active
            # fraction, regressed toward target 0.05 via squared error.
            sp_loss = (torch.sigmoid((kc_rates - 0.02) * 50).mean() - 0.05) ** 2
            (ce_loss + sp_w * sp_loss).backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)  # stabilize BPTT
            optimizer.step()
            clamp_biological(student)  # re-project params into biophysical bounds
            train_correct += (logits.argmax(-1) == by).sum().item()
            train_total += len(by)

        if (epoch + 1) % 50 == 0:
            # Periodic full evaluation + history logging.
            student.eval()
            test_acc, sparsity = evaluate_model(student, test_loader)
            decorr = compute_mean_sim_decorrelation(student, or_responses)
            mancini = run_mancini(student)
            g_soma = np.exp(student.kc_layer.kc_neurons.log_g_soma.item()) * 1e9  # log S -> nS

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
            # Deep-copy the epoch-300 weights; final metrics are computed from this
            # exact snapshot (insulating them from any later epochs if added).
            ep300_state = {k: v.clone() for k, v in student.state_dict().items()}

    # ---- EVALUATE AT EPOCH 300 ----
    student.load_state_dict(ep300_state)  # restore the epoch-300 snapshot
    student.eval()
    print(f"\n  --- Seed {seed} Epoch 300 Evaluation ---")

    test_acc, sparsity = evaluate_model(student, test_loader)
    cent_acc = centroid_accuracy(student, or_responses)
    pp_decorr = compute_per_pair_decorrelation(student, or_responses)
    mancini = run_mancini(student)
    mancini_pass = bool(1.5 <= mancini <= 2.5)  # biologically plausible APL-gating window (python bool: JSON round-trips cleanly)
    conc_results, conc_tests = run_concentration_invariance(student, or_responses, seed)

    # Gap junction conductances
    # I_gap = g_gap * (V_pre - V_post): exponentiate log conductances to S for reporting.
    g_gap_ln = float(np.exp(student.antennal_lobe.log_g_gap_ln.item())) if student.antennal_lobe.log_g_gap_ln is not None else None
    g_gap_pn = float(np.exp(student.antennal_lobe.log_g_gap_pn.item()))
    g_gap_eln = float(np.exp(student.antennal_lobe.log_g_gap_eln_pn.item()))
    ln_pn_inhib_str = float(np.exp(student.antennal_lobe.ln_pn.log_strength.item()))   # learned inhibitory LN->PN (A)
    ln_pn_excit_str = float(np.exp(student.antennal_lobe.ln_pn_excit.log_strength.item()))  # learned excitatory LN->PN (A)
    g_soma = np.exp(student.kc_layer.kc_neurons.log_g_soma.item()) * 1e9  # nS
    apl_gain = F.softplus(torch.tensor(student.kc_layer.apl.apl_gain.item())).item()  # softplus keeps APL gain > 0
    kc_apl_strength = float(np.exp(student.kc_layer.apl.kc_apl_log_strength.item()))  # KC->APL drive (A)

    # Non-AD strengths
    # Collect the learned non-AD synaptic strengths (A) that are present in this model.
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

    print(f"  Acc: {test_acc:.1%}, Centroid: {cent_acc:.1%}, Sp: {sparsity:.1%}")
    print(f"  AL: {pp_decorr['al_decorr_pct']:.1f}%, MB: {pp_decorr['mb_decorr_pct']:.1f}%, Total: {pp_decorr['total_decorr_pct']:.1f}%")
    print(f"  Mancini: {mancini:.2f} ({'PASS' if mancini_pass else 'FAIL'})")
    print(f"  Conc: SubPN={'P' if conc_tests['sublinear_pn_gain'] else 'F'}, "
          f"FlatKC={'P' if conc_tests['flat_kc_activity'] else 'F'}, "
          f"RobClass={'P' if conc_tests['robust_classification'] else 'F'}, "
          f"Identity={conc_tests['odor_identity_preservation']}")
    # Assemble the per-seed results record (consumed by the summary/JSON/plots).
    results = {
        'seed': seed, 'eval_epoch': 300, 'teacher_accuracy': teacher_acc,
        'accuracy': test_acc, 'centroid_accuracy': cent_acc, 'sparsity': sparsity,
        'per_pair_decorrelation': pp_decorr,
        'mancini': mancini, 'mancini_pass': mancini_pass,
        'concentration_invariance': {
            'per_concentration': conc_results,
            # keep only JSON-serializable scalar predictions
            'predictions': {k: v for k, v in conc_tests.items() if isinstance(v, (bool, float, int, str))},
        },
        'g_soma_nS': g_soma, 'apl_gain_effective': apl_gain, 'kc_apl_strength': kc_apl_strength,
        'gap_junction_conductances': {'ln_ln': g_gap_ln, 'pn_pn': g_gap_pn, 'eln_pn': g_gap_eln},
        'ln_pn_split': {'inhibitory_strength': ln_pn_inhib_str, 'excitatory_strength': ln_pn_excit_str},
        'nonad_strengths': nonad_strengths,
        'history': history,
    }

    # Teacher was already saved (teacher_seed{seed}.pt) right after teacher training above.
    torch.save(student.state_dict(), output_dir / f'model_seed{seed}.pt')  # persist trained student
    return results, student


# ============================================================================
# MAIN
# ============================================================================
def _canonical_locate_data():
    """Find the connectome data directory (contains kreher2008/) and ensure OUTPUT_DIR exists.

    Returns:
        Path to the data directory.
    """
    # Probe likely locations for the connectome data directory (contains kreher2008/).
    _pkg_root = Path(__file__).resolve().parent.parent.parent
    _parent = _pkg_root.parent
    _data_candidates = [
        _pkg_root / 'data',
    ]
    data_dir = next((p for p in _data_candidates if (p / 'kreher2008').is_dir()), None)
    if data_dir is None:
        raise FileNotFoundError('Cannot find connectome data (kreher2008/).')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return data_dir


def _canonical_load_data(data_dir):
    """Load the Kreher-2008 dataset + clean OR responses for the canonical run.

    Returns:
        tuple(train_loader, test_loader, n_odors, or_responses).
    """
    # Load the Kreher-2008 28-odor dataset with multiplicative input noise and
    # build train/test DataLoaders (10 train repeats, 5 test repeats per odor).
    train_dataset, test_dataset, odor_names = load_kreher2008_all_odors(
        data_dir, train_repeats=10, test_repeats=5,
        noise_std=NOISE_STD, noise_type=NOISE_TYPE,
    )
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size=16)
    n_odors = len(odor_names)
    # Clean (noise-free) normalized OR responses [n_odors, n_or_types], used as the
    # reference stimulus set for the analysis metrics.
    df = pd.read_csv(data_dir / "kreher2008/orn_responses_normalized.csv", index_col=0)
    or_responses = torch.from_numpy(df.values).float()
    return train_loader, test_loader, n_odors, or_responses


def _canonical_banner():
    """Print the canonical-run configuration banner (printed once at the start of a run)."""
    print("=" * 80)
    print("ALL-CONNECTIONS CANONICAL (5 models)")
    print("=" * 80)
    print(f"\nSeeds: {SEEDS}")
    print(f"Epochs: {STUDENT_EPOCHS}")
    print(f"Output: {OUTPUT_DIR}")


def _to_jsonable(obj):
    """Recursively convert numpy scalars/arrays to native Python types.

    Ensures a per-seed results dict round-trips through JSON as real numbers/bools
    so the aggregator can recompute mean/std on the reloaded values.
    """
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _save_seed_results(seed, results):
    """Write one seed's results dict to results_seed{seed}.json (JSON-clean)."""
    path = OUTPUT_DIR / f'results_seed{seed}.json'
    with open(path, 'w') as f:
        json.dump(_to_jsonable(results), f, indent=2)
    return path


def _canonical_train_one_seed(seed):
    """Train ONE canonical seed (its own teacher+student) and save its per-seed results.

    Self-contained and independently seeded, so dispatching one of these per seed in
    parallel yields results bit-identical to the sequential loop. Writes
    model_seed{seed}.pt (via train_single_model) and results_seed{seed}.json.

    Side effects:
        Creates OUTPUT_DIR; writes the model + per-seed results JSON; prints progress.
    """
    data_dir = _canonical_locate_data()
    train_loader, test_loader, n_odors, or_responses = _canonical_load_data(data_dir)
    results, _ = train_single_model(
        seed, data_dir, train_loader, test_loader, n_odors, or_responses, OUTPUT_DIR)
    path = _save_seed_results(seed, results)
    print(f"\n  Per-seed results saved: {path}")


def _canonical_aggregate(all_results=None):
    """Aggregate per-seed canonical results into results.json + plots + summary.

    Args:
        all_results: list of per-seed results dicts. If None, load them from the
            results_seed{seed}.json files on disk (the parallel-dispatch path).

    Produces output identical to the in-memory sequential run.

    Side effects:
        Writes results.json and training_curves.png; prints the cross-seed summary.
    """
    if all_results is None:
        all_results = []
        for seed in SEEDS:
            path = OUTPUT_DIR / f'results_seed{seed}.json'
            if not path.exists():
                raise FileNotFoundError(
                    f'Missing per-seed results: {path} (train seed {seed} first).')
            with open(path) as f:
                all_results.append(json.load(f))

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*80}")
    print("ALL-CONNECTIONS RESULTS SUMMARY (5 models, epoch 300)")
    print(f"{'='*80}")

    # Gather the headline metrics across seeds for mean/std aggregation.
    accs = [r['accuracy'] for r in all_results]
    cents = [r['centroid_accuracy'] for r in all_results]
    sps = [r['sparsity'] for r in all_results]
    al_ds = [r['per_pair_decorrelation']['al_decorr_pct'] for r in all_results]
    mb_ds = [r['per_pair_decorrelation']['mb_decorr_pct'] for r in all_results]
    total_ds = [r['per_pair_decorrelation']['total_decorr_pct'] for r in all_results]
    mancs = [r['mancini'] for r in all_results]
    gsomas = [r['g_soma_nS'] for r in all_results]

    # Per-seed table
    print(f"\n{'Seed':<6} {'Acc':<8} {'Cent':<8} {'Sp':<8} {'AL%':<8} {'MB%':<8} {'Tot%':<8} {'Manc':<8}")
    print("-" * 62)
    for r in all_results:
        pp = r['per_pair_decorrelation']
        m_ok = "P" if r['mancini_pass'] else "F"
        print(f"{r['seed']:<6} {r['accuracy']:<8.1%} {r['centroid_accuracy']:<8.1%} {r['sparsity']:<8.1%} "
              f"{pp['al_decorr_pct']:<8.1f} {pp['mb_decorr_pct']:<8.1f} {pp['total_decorr_pct']:<8.1f} "
              f"{r['mancini']:.2f}{m_ok:<3}")
    print("-" * 62)
    # Mean and std (std reported in percentage points for the fraction-valued metrics).
    print(f"{'Mean':<6} {np.mean(accs):<8.1%} {np.mean(cents):<8.1%} {np.mean(sps):<8.1%} "
          f"{np.mean(al_ds):<8.1f} {np.mean(mb_ds):<8.1f} {np.mean(total_ds):<8.1f} "
          f"{np.mean(mancs):<8.2f}")
    print(f"{'Std':<6} {np.std(accs)*100:<8.1f} {np.std(cents)*100:<8.1f} {np.std(sps)*100:<8.1f} "
          f"{np.std(al_ds):<8.1f} {np.std(mb_ds):<8.1f} {np.std(total_ds):<8.1f} "
          f"{np.std(mancs):<8.2f}")

    # Mancini pass rate
    n_pass = sum(1 for r in all_results if r['mancini_pass'])
    print(f"\nMancini: {n_pass}/{len(SEEDS)} pass")

    # Concentration invariance summary
    # Report each of the 4 concentration tests: count passes for boolean tests,
    # else list raw values.
    print(f"\nConcentration Invariance:")
    for test_name in ['sublinear_pn_gain', 'flat_kc_activity', 'robust_classification', 'odor_identity_preservation']:
        vals = [r['concentration_invariance']['predictions'][test_name] for r in all_results]
        if isinstance(vals[0], bool):
            n_pass_t = sum(vals)
            print(f"  {test_name}: {n_pass_t}/{len(SEEDS)} pass")
        else:
            print(f"  {test_name}: {', '.join(str(v) for v in vals)}")


    # ========================================================================
    # SAVE
    # ========================================================================
    # Assemble the JSON payload: config (for reproducibility), cross-seed summary
    # stats, and the full per-seed records.
    results_json = {
        'config': {
            'apl_boost': APL_BOOST, 'teacher_epochs': TEACHER_EPOCHS, 'student_epochs': STUDENT_EPOCHS,
            'g_soma_min_nS': G_SOMA_MIN * 1e9, 'g_soma_max_nS': G_SOMA_MAX_BIO * 1e9,
            'noise_type': NOISE_TYPE, 'noise_std': NOISE_STD, 'seeds': SEEDS,
            'v_noise_std': REALISTIC_PARAMS.v_noise_std,
            'i_noise_std': REALISTIC_PARAMS.i_noise_std,
            'syn_noise_std': REALISTIC_PARAMS.syn_noise_std,
            'threshold_jitter_std': REALISTIC_PARAMS.threshold_jitter_std,
            'orn_receptor_noise_std': REALISTIC_PARAMS.orn_receptor_noise_std,
            'nonad_init': float(np.exp(NONAD_INIT)),
            'nonad_lr': NONAD_LR,
        },
        'summary': {
            'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
            'centroid_mean': float(np.mean(cents)), 'centroid_std': float(np.std(cents)),
            'sparsity_mean': float(np.mean(sps)), 'sparsity_std': float(np.std(sps)),
            'al_decorr_mean': float(np.mean(al_ds)), 'al_decorr_std': float(np.std(al_ds)),
            'mb_decorr_mean': float(np.mean(mb_ds)), 'mb_decorr_std': float(np.std(mb_ds)),
            'total_decorr_mean': float(np.mean(total_ds)), 'total_decorr_std': float(np.std(total_ds)),
            'mancini_mean': float(np.mean(mancs)), 'mancini_std': float(np.std(mancs)),
            'mancini_pass_rate': f"{n_pass}/{len(SEEDS)}",
            'g_soma_mean': float(np.mean(gsomas)), 'g_soma_std': float(np.std(gsomas)),
        },
        'teacher_accs': [r.get('teacher_accuracy') for r in all_results],
        'per_seed': [],
    }
    for r in all_results:
        # Re-attach the history with explicit epoch axis (50, 100, ..., STUDENT_EPOCHS).
        seed_data = {k: v for k, v in r.items() if k != 'history'}
        seed_data['history'] = {
            'epochs': list(range(50, STUDENT_EPOCHS + 1, 50)),
            **{k: v for k, v in r['history'].items()},
        }
        results_json['per_seed'].append(seed_data)

    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2, default=str)  # default=str: serialize any non-JSON types

    # ========================================================================
    # PLOTS
    # ========================================================================
    # 2x3 panel of training curves; x-axis is the periodic-eval epoch grid.
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    epochs = list(range(50, STUDENT_EPOCHS + 1, 50))

    # Panel (0,0): test accuracy vs epoch, one line per seed.
    ax = axes[0, 0]
    for r in all_results:
        ax.plot(epochs, [a*100 for a in r['history']['test_acc']], 'o-', alpha=0.7, label=f"Seed {r['seed']}")
    ax.set_xlabel('Epoch'); ax.set_ylabel('Test Accuracy (%)'); ax.set_title('Test Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel (0,1): KC sparsity vs epoch with a dashed 10% target line.
    ax = axes[0, 1]
    for r in all_results:
        ax.plot(epochs, [s*100 for s in r['history']['sparsity']], 'o-', alpha=0.7, label=f"Seed {r['seed']}")
    ax.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Target')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Sparsity (%)'); ax.set_title('KC Sparsity')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel (0,2): AL (dashed) vs MB (solid) decorrelation per seed.
    ax = axes[0, 2]
    for r in all_results:
        ax.plot(epochs, r['history']['al_decorr'], 's--', alpha=0.5, label=f"AL s{r['seed']}")
        ax.plot(epochs, r['history']['mb_decorr'], 'o-', alpha=0.7, label=f"MB s{r['seed']}")
    ax.set_xlabel('Epoch'); ax.set_ylabel('Decorrelation (%)'); ax.set_title('AL vs MB Decorrelation')
    ax.grid(True, alpha=0.3)

    # Panel (1,0): Mancini ratio vs epoch with target (2.0) and acceptance band (1.5-2.5).
    ax = axes[1, 0]
    for r in all_results:
        ax.plot(epochs, r['history']['mancini'], 'o-', alpha=0.7, label=f"Seed {r['seed']}")
    ax.axhline(y=2.0, color='g', linestyle='--', alpha=0.5)
    ax.axhline(y=1.5, color='r', linestyle='--', alpha=0.3)
    ax.axhline(y=2.5, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Mancini Ratio'); ax.set_title('APL Inhibition')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel (1,1): KC soma conductance vs epoch with clamp bounds [1, 20] nS.
    ax = axes[1, 1]
    for r in all_results:
        ax.plot(epochs, r['history']['g_soma'], 'o-', alpha=0.7, label=f"Seed {r['seed']}")
    ax.axhline(y=G_SOMA_MAX_BIO*1e9, color='r', linestyle='--', alpha=0.3)
    ax.axhline(y=G_SOMA_MIN*1e9, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('g_soma (nS)'); ax.set_title('KC Soma Conductance')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel (1,2): grouped bar chart of final per-seed metrics (Mancini scaled x20
    # to share the axis with the percentage-valued metrics).
    ax = axes[1, 2]
    x = np.arange(len(SEEDS)); width = 0.15
    ax.bar(x - 2*width, [r['accuracy']*100 for r in all_results], width, label='Acc', color='blue', alpha=0.7)
    ax.bar(x - width, [r['centroid_accuracy']*100 for r in all_results], width, label='Cent', color='cyan', alpha=0.7)
    ax.bar(x, [r['per_pair_decorrelation']['al_decorr_pct'] for r in all_results], width, label='AL%', color='orange', alpha=0.7)
    ax.bar(x + width, [r['per_pair_decorrelation']['mb_decorr_pct'] for r in all_results], width, label='MB%', color='red', alpha=0.7)
    ax.bar(x + 2*width, [r['mancini']*20 for r in all_results], width, label='Manc*20', color='purple', alpha=0.7)
    ax.set_xlabel('Seed'); ax.set_ylabel('Value'); ax.set_title('Final Metrics by Seed')
    ax.set_xticks(x); ax.set_xticklabels(SEEDS); ax.legend(loc='upper right'); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=150)
    plt.close()

    print(f"\nResults saved to {OUTPUT_DIR}/")


def _main_canonical():
    """Full canonical 5-seed run, sequential (the backward-compatible no-arg entry point).

    Trains every seed in one process and aggregates in memory. The notebook instead
    dispatches one `canonical --seed N` subprocess per seed in parallel and then calls
    `canonical-aggregate`; both paths write an identical results.json.

    Side effects:
        Creates OUTPUT_DIR; writes per-seed model + results JSON, results.json, and
        training_curves.png; prints the banner, per-seed progress, and final summary.
    """
    data_dir = _canonical_locate_data()
    _canonical_banner()
    train_loader, test_loader, n_odors, or_responses = _canonical_load_data(data_dir)
    all_results = []
    for seed in SEEDS:
        results, _ = train_single_model(
            seed, data_dir, train_loader, test_loader, n_odors, or_responses, OUTPUT_DIR)
        _save_seed_results(seed, results)  # also persist per-seed JSON (parity with parallel path)
        all_results.append(results)
    _canonical_aggregate(all_results)



# ============================================================================
# ENERGY / SPARSITY-LOSS VARIANTS  (merged from former run_training_energy_only.py)
# Reuses the canonical constants/clamp_biological/get_param_groups above (identical).
# ============================================================================
# These variants explore alternative loss formulations (C7 ablation): cross-entropy
# plus an explicit metabolic "energy" penalty on firing rates, with the canonical
# KC-sparsity loss optionally removed, to test whether sparseness can arise from an
# energy objective rather than a direct sparsity regularizer.
DEVICE = torch.device('cpu')  # All energy-variant training runs on CPU
ENERGY_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / 'results' / 'energy_only_r1'
ENERGY_TEACHER_PATH = None            # set in _main_energy()
ENERGY_RAMP_EPOCHS = 60               # warm-up length for the sparsity/energy weight ramp
SEED = 42                             # default seed (overridable via --seed)


class SpikeAccumulator:
    """Hook-based accumulator for ORN and LN spike counts.

    The energy loss needs per-step ORN and LN spikes, which are emitted inside the
    AL forward pass but not returned in the top-level `info` dict (unlike PN/KC).
    This class registers forward hooks on the ORN and LN neuron modules and sums
    each layer's spike tensor across the time steps of a single forward call.

    Attributes:
        orn_spikes: running sum [batch, n_ORN] of ORN spikes for the current pass
                    (None before any step), or None after reset().
        ln_spikes:  running sum [batch, n_LN] of LN spikes, same convention.
        _hooks: list of removable forward-hook handles.
    """
    def __init__(self):
        self.orn_spikes = None
        self.ln_spikes = None
        self._hooks = []

    def register(self, model):
        """Attach forward hooks to the model's ORN and LN neuron modules.

        Each hook reads spikes = output[1] (the LIFNeuron returns (v, spikes)) and
        accumulates them across the time-loop calls within one forward pass.

        Args:
            model: spiking student model.

        Returns:
            self (for chaining, e.g. SpikeAccumulator().register(student)).
        """
        def orn_hook(module, input, output):
            spk = output[1]  # LIFNeuron.forward returns (v_mem, spikes); take spikes
            if self.orn_spikes is None:
                self.orn_spikes = spk
            else:
                self.orn_spikes = self.orn_spikes + spk  # accumulate across time steps

        def ln_hook(module, input, output):
            spk = output[1]
            if self.ln_spikes is None:
                self.ln_spikes = spk
            else:
                self.ln_spikes = self.ln_spikes + spk

        self._hooks.append(
            model.antennal_lobe.orn_neurons.register_forward_hook(orn_hook))
        self._hooks.append(
            model.antennal_lobe.ln_neurons.register_forward_hook(ln_hook))
        return self

    def reset(self):
        """Clear accumulated spikes before each new forward pass."""
        self.orn_spikes = None
        self.ln_spikes = None

    def remove(self):
        """Detach all registered hooks (call when done to avoid leaks)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

def run_analysis(model, or_responses, seed):
    """Run all analysis metrics. Model must be on CPU.

    Convenience aggregator used by the energy-variant evaluation: computes per-pair
    decorrelation, the Mancini APL test, centroid accuracy, and the concentration-
    invariance suite, all at this script's NOISE_STD.

    Args:
        model: spiking student model (must already be on CPU).
        or_responses: tensor [n_odors, n_or_types].
        seed: RNG seed for reproducible noisy trials.

    Returns:
        tuple(pp, manc, cent, conc_results, conc_tests):
          pp           — per-pair decorrelation dict,
          manc         — Mancini result dict (ratio, passes, spike counts),
          cent         — centroid accuracy (float),
          conc_results — per-concentration numeric results,
          conc_tests   — concentration pass/fail prediction dict.
    """
    pp = _compute_per_pair_decorrelation(model, or_responses, 10, NOISE_STD)
    manc = _run_mancini_test(model)
    cent = _centroid_accuracy(model, or_responses, 20, NOISE_STD)
    conc_results, conc_tests = _run_concentration_invariance(
        model, or_responses, seed, CONCENTRATIONS, HILL_EC50, HILL_N,
        N_CONC_TRIALS, NOISE_STD)
    return pp, manc, cent, conc_results, conc_tests


# ============================================================================
# STUDENT INITIALIZATION
# ============================================================================
def initialize_student(data_dir, n_odors, teacher_state):
    """Create and initialize a student model from teacher state.

    Mirrors the inline student setup in train_single_model(), but takes a saved
    teacher state_dict (so the teacher can be trained once and reused across energy
    variants) and returns the student already moved to DEVICE.

    Args:
        data_dir: Path to connectome data.
        n_odors: number of output classes.
        teacher_state: state_dict of a trained ConnectomeConstrainedModel.

    Returns:
        SpikingConnectomeConstrainedModel on DEVICE, seeded from the teacher and
        the same biophysical init used in the canonical run.
    """
    student = SpikingConnectomeConstrainedModel.from_data_dir(
        data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10,
        params=REALISTIC_PARAMS, include_nonad=True)
    student.n_steps_al = N_STEPS  # 30 AL steps (canonical)
    student.n_steps_kc = N_STEPS  # 30 KC steps

    # Rebuild the teacher object only to load the saved weights for seeding.
    teacher = ConnectomeConstrainedModel.from_data_dir(
        data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10)
    teacher.load_state_dict(teacher_state)

    with torch.no_grad():
        # Thresholds: LNs more excitable (-47.5 mV) than other types (-42.5 mV).
        for name, param in student.named_parameters():
            if 'v_th' in name:
                param.fill_(LN_VTH_INIT if 'ln' in name else -0.0425)

        # Knowledge transfer from teacher: decoder, OR gains, APL gain (x boost).
        student.decoder.weight.copy_(teacher.decoder.weight)
        student.decoder.bias.copy_(teacher.decoder.bias)
        student.or_to_orn.or_gains.copy_(teacher.or_to_orn.or_gains)
        student.kc_layer.apl.apl_gain.data = teacher.kc_layer.apl.apl_gain.data.clone() * APL_BOOST

        student.kc_layer.kc_neurons.v_th.data += 0.005  # nudge KC threshold up for sparseness
        student.kc_layer.kc_kc_aa.log_strength.fill_(np.log(1e-11))  # KC-KC axo-axonic init (~10 pA)
        if student.kc_layer.kc_kc_ad is not None:
            student.kc_layer.kc_kc_ad.log_strength.fill_(np.log(1e-13))  # KC-KC axo-dendritic init (~0.1 fA)
        student.kc_layer.kc_neurons.log_g_soma.fill_(np.log(10e-9))  # KC coupling conductance init 10 nS

        # AL recurrent synapses start effectively off and are learned up.
        if student.antennal_lobe.ln_ln is not None:
            student.antennal_lobe.ln_ln.log_strength.fill_(np.log(1e-13))
        if student.antennal_lobe.pn_ln is not None:
            student.antennal_lobe.pn_ln.log_strength.fill_(np.log(1e-13))
        if student.antennal_lobe.ln_orn is not None:
            student.antennal_lobe.ln_orn.log_strength.fill_(np.log(1e-13))
        # Boost inhibitory LN->PN by LN_PN_SCALE; set excitatory LN->PN to 10% of it.
        ln_pn_orig = student.antennal_lobe.ln_pn.log_strength.item()
        student.antennal_lobe.ln_pn.log_strength.fill_(ln_pn_orig + np.log(LN_PN_SCALE))
        ln_pn_inhib_strength = np.exp(student.antennal_lobe.ln_pn.log_strength.item())
        student.antennal_lobe.ln_pn_excit.log_strength.fill_(
            np.log(ln_pn_inhib_strength * 0.1))
        # Attenuate direct ORN->PN drive by ORN_PN_SCALE.
        orn_pn_orig = student.antennal_lobe.orn_pn.log_strength.item()
        student.antennal_lobe.orn_pn.log_strength.fill_(orn_pn_orig + np.log(ORN_PN_SCALE))

        # Initialize all AL non-AD synapses very weak (loop over present attrs).
        for attr in ['orn_ln_nonad', 'ln_pn_nonad', 'ln_pn_excit_nonad',
                      'ln_ln_nonad', 'pn_ln_nonad', 'ln_orn_nonad']:
            layer = getattr(student.antennal_lobe, attr, None)
            if layer is not None:
                layer.log_strength.fill_(NONAD_INIT)
        if student.kc_layer.pn_kc_nonad is not None:
            student.kc_layer.pn_kc_nonad.log_strength.fill_(NONAD_INIT)

    return student.to(DEVICE)


# ============================================================================
# TRAIN TEACHER (run once, save to disk)
# ============================================================================
def train_teacher(data_dir, n_odors, train_loader, test_loader):
    """Train rate-based teacher and save state dict.

    Used by the energy variants so the (expensive) teacher is trained once per seed
    and cached to ENERGY_TEACHER_PATH for reuse across loss-formulation runs.

    Args:
        data_dir: Path to connectome data.
        n_odors: number of output classes.
        train_loader, test_loader: DataLoaders.

    Returns:
        dict: the teacher's CPU state_dict (also written to disk).

    Side effects:
        Creates ENERGY_OUTPUT_DIR; writes the teacher .pt; prints progress.
    """
    print(f"--- Training teacher (seed {SEED}, {TEACHER_EPOCHS} epochs) ---")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    teacher = ConnectomeConstrainedModel.from_data_dir(
        data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10)
    opt = torch.optim.Adam(teacher.parameters(), lr=1e-2)
    teacher_acc = float('nan')  # final rate-teacher test accuracy (captured at last eval, saved to results)
    for ep in range(TEACHER_EPOCHS):
        teacher.train()
        for bx, by in train_loader:
            opt.zero_grad()
            loss, _ = teacher.compute_loss(bx, by, sparsity_weight=2.0)  # CE + sparsity
            loss.backward()
            opt.step()
        if (ep + 1) % 100 == 0:
            teacher.eval()
            c, t = 0, 0
            with torch.no_grad():
                for bx, by in test_loader:
                    c += (teacher(bx).argmax(-1) == by).sum().item()
                    t += len(by)
            teacher_acc = c / t  # keep the latest (epoch-300) teacher test accuracy
            print(f"  Teacher epoch {ep+1}: {teacher_acc:.1%}")

    ENERGY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    teacher_state = {k: v.cpu().clone() for k, v in teacher.state_dict().items()}  # detach to CPU
    torch.save(teacher_state, ENERGY_TEACHER_PATH)
    print(f"  Teacher saved to {ENERGY_TEACHER_PATH}")
    return teacher_state


# ============================================================================
# TRAIN STUDENT (CE + optional energy, NO KC sparsity)
# ============================================================================
def train_student(energy_weight, label, data_dir, train_loader, test_loader,
                  n_odors, or_responses, teacher_state, kc_sparsity=False,
                  kc_energy_only=False):
    """Train student with CE + optional energy + optional KC sparsity.

    The C7 ablation core: trains a spiking student under a configurable loss that
    is always cross-entropy, optionally plus the canonical KC-sparsity term, and
    optionally plus a firing-rate "energy" penalty (either KC-only or averaged
    across all neuron types). Both extra terms share the same 0->1 warm-up ramp.

    Args:
        energy_weight: peak weight on the energy term (0 disables it).
        label: short string naming this run (used in filenames/prints).
        data_dir: Path to connectome data.
        train_loader, test_loader: DataLoaders.
        n_odors: number of output classes.
        or_responses: tensor [n_odors, n_or_types] for the analysis metrics.
        teacher_state: state_dict used to seed the student.
        kc_sparsity: if True, include the canonical KC-sparsity loss.
        kc_energy_only: if True, energy penalizes only KC rate; else mean over
                        ORN/LN/PN/KC rates.

    Side effects:
        Trains for STUDENT_EPOCHS; prints progress and a final evaluation; writes
        the student .pt and a results_{label}[_seed{SEED}].json to ENERGY_OUTPUT_DIR.
    """
    # Build a human-readable description of the active loss terms for the banner.
    loss_desc = "CE"
    if kc_sparsity:
        loss_desc += " + KC sparsity"
    if energy_weight > 0:
        energy_type = "KC-only energy" if kc_energy_only else "all-type energy"
        loss_desc += f" + {energy_weight} * {energy_type}"
    print(f"\n{'='*70}")
    print(f"C7: {label} (energy_weight={energy_weight}, seed {SEED})")
    print(f"Loss = {loss_desc}")
    print(f"{'='*70}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    student = initialize_student(data_dir, n_odors, teacher_state)
    accumulator = SpikeAccumulator().register(student)  # hooks to read ORN/LN spikes for the energy term

    param_groups = get_param_groups(student)
    optimizer = torch.optim.Adam(param_groups)

    for epoch in range(STUDENT_EPOCHS):
        # Ramp weights over first 60 epochs (same schedule as canonical)
        progress = min(1.0, epoch / ENERGY_RAMP_EPOCHS)
        sp_w = progress * 15.0 if kc_sparsity else 0.0  # KC sparsity weight
        e_w = progress * energy_weight                  # energy weight (ramped)

        student.train()
        train_correct, train_total = 0, 0

        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            accumulator.reset()  # clear ORN/LN spike sums before this forward pass

            logits, info = student(bx, return_all=True)

            ce_loss = F.cross_entropy(logits, by)
            total_loss = ce_loss

            # KC sparsity loss (canonical, if enabled)
            kc_rates = info['kc_spikes'] / N_STEPS  # per-cell fraction of active steps
            if kc_sparsity:
                # Same smooth-indicator sparsity loss as the canonical run.
                sp_loss = (torch.sigmoid((kc_rates - 0.02) * 50).mean() - 0.05) ** 2
                total_loss = total_loss + sp_w * sp_loss

            # Energy constraint (if energy_weight > 0)
            if energy_weight > 0:
                if kc_energy_only:
                    # KC-only energy: penalize only KC mean firing rate
                    energy_loss = kc_rates.mean()
                else:
                    # All-type energy: penalize mean rate across all neuron types equally
                    # (ORN/LN come from the accumulator hooks; PN/KC from `info`).
                    orn_rate = accumulator.orn_spikes.mean() / N_STEPS
                    ln_rate = accumulator.ln_spikes.mean() / N_STEPS
                    pn_rate = info['pn_spikes'].mean() / N_STEPS
                    kc_rate = kc_rates.mean()
                    energy_loss = (orn_rate + ln_rate + pn_rate + kc_rate) / 4.0
                total_loss = total_loss + e_w * energy_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
            optimizer.step()
            clamp_biological(student)  # re-project to biophysical bounds

            train_correct += (logits.argmax(-1) == by).sum().item()
            train_total += len(by)

        # Periodic evaluation (every 50 epochs)
        if (epoch + 1) % 50 == 0:
            student.eval()
            test_correct, test_total = 0, 0
            all_orn, all_ln, all_pn, all_kc = [], [], [], []  # per-type "active fraction" trackers
            with torch.no_grad():
                for bx, by in test_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    accumulator.reset()
                    logits, info = student(bx, return_all=True)
                    test_correct += (logits.argmax(-1) == by).sum().item()
                    test_total += len(by)
                    # Fraction of cells that fired at least once this pass (per type).
                    all_orn.append((accumulator.orn_spikes > 0).float().mean().item())
                    all_ln.append((accumulator.ln_spikes > 0).float().mean().item())
                    all_pn.append((info['pn_spikes'] > 0).float().mean().item())
                    all_kc.append((info['kc_spikes'] > 0).float().mean().item())

            # Decorrelation/Mancini metrics are CPU-only; move there and back.
            student.cpu()
            decorr = compute_mean_sim_decorrelation(student, or_responses)
            manc = _run_mancini_test(student)
            student.to(DEVICE)

            # Sp = KC active fraction (the KC sparsity proxy used across all runs).
            print(f"  Ep {epoch+1}: Train={train_correct/train_total:.1%}, Test={test_correct/test_total:.1%}, "
                  f"Sp={np.mean(all_kc):.1%}, AL={decorr['al_decorr']:.1f}%, MB={decorr['mb_decorr']:.1f}%, "
                  f"Manc={manc['ratio']:.2f}")

    # ---- FINAL EVALUATION ----
    student.eval()
    print(f"\n  --- FINAL EVALUATION: {label} ---")

    # Per-type sparsity
    # Recompute test accuracy and per-type active fractions for the final report.
    test_correct, test_total = 0, 0
    all_orn, all_ln, all_pn, all_kc = [], [], [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            accumulator.reset()
            logits, info = student(bx, return_all=True)
            test_correct += (logits.argmax(-1) == by).sum().item()
            test_total += len(by)
            all_orn.append((accumulator.orn_spikes > 0).float().mean().item())
            all_ln.append((accumulator.ln_spikes > 0).float().mean().item())
            all_pn.append((info['pn_spikes'] > 0).float().mean().item())
            all_kc.append((info['kc_spikes'] > 0).float().mean().item())

    accumulator.remove()  # detach hooks before the CPU analysis below
    test_acc = test_correct / test_total
    per_type_sp = {
        'orn': float(np.mean(all_orn)), 'ln': float(np.mean(all_ln)),
        'pn': float(np.mean(all_pn)), 'kc': float(np.mean(all_kc)),
    }

    # Full analysis on CPU
    student.cpu()
    pp, manc, cent_acc, conc_results, conc_tests = run_analysis(
        student, or_responses, SEED)

    g_soma = np.exp(student.kc_layer.kc_neurons.log_g_soma.item()) * 1e9  # nS
    apl_gain = F.softplus(torch.tensor(student.kc_layer.apl.apl_gain.item())).item()  # softplus -> positive gain

    print(f"  Accuracy:  linear={test_acc:.1%}, centroid={cent_acc:.1%}")
    print(f"  Sparsity:  ORN={per_type_sp['orn']:.1%}, LN={per_type_sp['ln']:.1%}, "
          f"PN={per_type_sp['pn']:.1%}, KC={per_type_sp['kc']:.1%}")
    print(f"  Decorr:    AL={pp['al_decorr']:.1f}%, MB={pp['mb_decorr']:.1f}%")
    print(f"  Mancini:   {manc['ratio']:.2f} ({'PASS' if manc['passes'] else 'FAIL'})")
    print(f"  Gain:      OR={conc_tests['or_range']:.2f}x, "
          f"PN={conc_tests['pn_range']:.2f}x, KC={conc_tests['kc_range']:.2f}x")
    print(f"  Flat KC:   {conc_tests['flat_kc_activity']}")
    print(f"  SubPN:     {conc_tests['sublinear_pn_gain']}")
    print(f"  RobClass:  {conc_tests['robust_classification']}")
    print(f"  OdorID:    {conc_tests['odor_identity_preservation']}")

    # Save model + results
    model_path = ENERGY_OUTPUT_DIR / f'model_{label}_seed{SEED}.pt'
    torch.save(student.state_dict(), model_path)

    results = {
        'label': label, 'energy_weight': energy_weight, 'seed': SEED,
        'loss_formulation': f'CE + {energy_weight} * energy (no KC sparsity)',
        'accuracy': test_acc, 'centroid_accuracy': cent_acc,
        'per_type_sparsity': per_type_sp,
        'decorrelation': {
            'al': pp['al_decorr'], 'mb': pp['mb_decorr'],
            'total': pp['total_decorr'],
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
    }

    # seed 42 -> results_{label}.json (no suffix) to match the notebook reader/skip-guard;
    # other seeds -> results_{label}_seed{SEED}.json
    results_path = ENERGY_OUTPUT_DIR / (f'results_{label}.json' if SEED == 42
                                 else f'results_{label}_seed{SEED}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Model saved: {model_path}")
    print(f"  Results saved: {results_path}")
    print(f"  DONE: {label}")

def _main_energy():
    """CLI entry point for the energy/sparsity-loss ablation variants.

    Parses arguments, locates data, loads the dataset, ensures a teacher exists
    (training and caching one if needed, or just training one when --train-teacher
    is set), then trains a single student under the requested loss configuration.

    CLI flags:
        --train-teacher : train and save the teacher only, then exit.
        --energy-weight : peak energy weight (0 = CE only).
        --kc-sparsity   : also include the canonical KC-sparsity loss.
        --kc-energy-only: energy penalizes KC rate only (not all types).
        --label         : run label for output filenames.
        --seed          : RNG seed (also selects the teacher checkpoint).

    Side effects:
        Sets the module-level SEED and ENERGY_TEACHER_PATH globals.
    """
    parser = argparse.ArgumentParser(description='C7: CE + energy (no KC sparsity)')
    parser.add_argument('--train-teacher', action='store_true',
                        help='Train and save teacher only')
    parser.add_argument('--energy-weight', type=float, default=0.0,
                        help='Energy constraint weight (0 = CE only)')
    parser.add_argument('--kc-sparsity', action='store_true',
                        help='Include canonical KC sparsity loss (CE + KC sp)')
    parser.add_argument('--kc-energy-only', action='store_true',
                        help='Energy penalty on KC firing rate only (not all types)')
    parser.add_argument('--label', type=str, default='ce_only',
                        help='Label for this run')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    # Promote parsed seed to the module globals so the helpers above pick it up,
    # and derive the per-seed teacher checkpoint path.
    global SEED, ENERGY_TEACHER_PATH
    SEED = args.seed
    ENERGY_TEACHER_PATH = ENERGY_OUTPUT_DIR / f'teacher_seed{SEED}.pt'

    # Find data
    _parent = Path(__file__).resolve().parent.parent.parent
    _data_candidates = [
        Path(__file__).resolve().parent.parent.parent / 'data',
    ]
    data_dir = next((p for p in _data_candidates if (p / 'kreher2008').is_dir()), None)
    if data_dir is None:
        raise FileNotFoundError('Cannot find connectome data (kreher2008/).')
    ENERGY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    train_ds, test_ds, odor_names = load_kreher2008_all_odors(
        data_dir, train_repeats=10, test_repeats=5,
        noise_std=NOISE_STD, noise_type=NOISE_TYPE)
    train_loader, test_loader = create_dataloaders(
        train_ds, test_ds, batch_size=16)
    n_odors = len(odor_names)
    df = pd.read_csv(data_dir / "kreher2008/orn_responses_normalized.csv", index_col=0)
    or_responses = torch.from_numpy(df.values).float()

    if args.train_teacher:
        # Train teacher only, then exit (used to pre-populate the cache).
        train_teacher(data_dir, n_odors, train_loader, test_loader)
        return

    # Load teacher
    # Reuse a cached teacher if present; otherwise train one now.
    if not ENERGY_TEACHER_PATH.exists():
        print("Teacher not found, training...")
        teacher_state = train_teacher(data_dir, n_odors, train_loader, test_loader)
    else:
        print(f"Loading teacher from {ENERGY_TEACHER_PATH}")
        teacher_state = torch.load(ENERGY_TEACHER_PATH, weights_only=False)

    train_student(args.energy_weight, args.label, data_dir,
                  train_loader, test_loader, n_odors, or_responses,
                  teacher_state, kc_sparsity=args.kc_sparsity,
                  kc_energy_only=args.kc_energy_only)

def main():
    """Dispatch on argv.

    No args                     -> full sequential canonical run (5 seeds + aggregate).
    canonical --seed N          -> train ONE canonical seed (parallel per-seed dispatch).
    canonical-banner            -> print the canonical config banner once, then exit.
    canonical-aggregate         -> aggregate existing per-seed results into results.json.
    anything else (e.g. --label)-> energy/sparsity-loss variant (argparse in _main_energy()).
    """
    argv = sys.argv[1:]
    if not argv:
        _main_canonical()
    elif argv[0] == 'canonical-banner':
        _canonical_banner()
    elif argv[0] == 'canonical-aggregate':
        _canonical_aggregate()
    elif argv[0] == 'canonical':
        # Per-seed canonical training: `run_training.py canonical --seed N`.
        p = argparse.ArgumentParser(prog='run_training.py canonical')
        p.add_argument('--seed', type=int, required=True, help='Seed to train (one model).')
        a = p.parse_args(argv[1:])
        _canonical_train_one_seed(a.seed)
    else:
        _main_energy()


if __name__ == "__main__":
    main()
