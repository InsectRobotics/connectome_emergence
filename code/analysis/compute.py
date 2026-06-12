"""Computation functions for connectome model analysis.

This module holds all of the *non-plotting* analysis used to turn trained
checkpoints into the quantitative results reported in the CCN 2026 paper. It
sits at the top of the analysis stack:

    checkpoints (.pt)  ->  this module  ->  numbers / arrays  ->  plotting.py / notebook

Every function here loads or operates on a fully-built
``SpikingConnectomeConstrainedModel`` (the connectome-constrained spiking
network: OR responses -> ORN(LIF) -> LN(LIF) -> PN(LIF) -> KC(two-compartment)
<-> APL(divisive inhibition) -> linear decoder over 28 odors) and computes a
specific metric. None of the biophysics (LIF integration, two-compartment KC
matrix exponential, Tsodyks-Markram short-term depression, gap junctions,
divisive APL inhibition) lives here -- that is all inside ``model.py`` /
``layers.py``. This file only *drives* the forward pass and aggregates outputs.

Contents (in file order):
  - REALISTIC_PARAMS : the canonical six-source noise configuration.
  - load_models      : build + restore checkpoints with the canonical config.
  - evaluate_model   : accuracy / sparsity / per-stage cosine similarity.
  - decorrelation    : per-pair (canonical) and mean-similarity variants.
  - run_mancini_test : APL-inhibition validation (Mancini 2023).
  - per-odor metrics : single-model and across-model accuracy/sparsity/KC.
  - consistency      : cross-model prediction and KC-pattern agreement.
  - centroid_accuracy: nearest-centroid classification in KC space.
  - circuit extraction: gap-junction conductances and non-AD strengths.
  - concentration invariance: Hill-scaled dose-response robustness tests.
  - compute_few_param_cv: per-parameter coefficient of variation for small groups.
"""

import numpy as np
import torch
from pathlib import Path

# The model class and its noise-parameter dataclass. Note the package is the
# code/ directory (core/, analysis/ live there).
from core.model import SpikingConnectomeConstrainedModel
from core.layers import SpikingParams

# Shared analysis primitives (cosine similarity, the unified noisy forward pass,
# the canonical per-pair decorrelation ratios, nearest-centroid classifier, and
# the Hill dose-response used by the concentration-invariance test).
from .utils import (
    cosine_sim_matrix, cosine_sim_pair, noisy_forward_pass,
    per_pair_similarity_ratios, centroid_classify,
    hill_effective_concentration,
)

# Canonical (paper) noise parameters — the six biologically-motivated noise sources
# at the exact magnitudes reported in the paper Methods. Identical to scripts/run_training.py
# REALISTIC_PARAMS; these are the values the saved checkpoints were trained/evaluated with.
# These deliberately OVERRIDE the smaller defaults baked into SpikingParams in
# layers.py (e.g. v_noise_std default is 0.5 mV there): analysis must reproduce the
# noisier regime the checkpoints actually saw. Units are SI (volts, amps, dimensionless CV).
REALISTIC_PARAMS = SpikingParams(
    v_noise_std=1.0e-3,           # 1.0 mV  : additive Gaussian voltage noise per LIF step (membrane noise)
    i_noise_std=15e-12,           # 15 pA   : additive Gaussian synaptic-input current noise per step
    syn_noise_std=0.25,           # 25% CV  : multiplicative noise on chemical-synapse transmission
    threshold_jitter_std=1.0e-3,  # 1.0 mV  : per-step jitter added to the spike threshold v_th
    orn_receptor_noise_std=0.10,  # 10% CV  : multiplicative noise on the OR receptor drive (sensory transduction)
    circuit_noise_enabled=True,   # master switch: all of the above are only injected when True
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(data_dir, model_dir, seeds, n_steps=30, include_nonad=True):
    """Load all trained models.

    Builds one ``SpikingConnectomeConstrainedModel`` per seed from the connectome
    data, restores its learned ~449 biological parameters from the matching
    checkpoint, forces the canonical timestep count, and puts it in eval mode.

    Args:
        data_dir: Path to the data/ directory
        model_dir: Path to results directory with model_seed{seed}.pt files
        seeds: list of random seeds
        n_steps: simulation timesteps (MUST be 30 for canonical models)
        include_nonad: whether to include non-AD connections

    Returns:
        list of eval-mode models, one per entry in ``seeds`` (same order).
    """
    models = []
    for seed in seeds:
        # Reconstruct the architecture from the connectome (fixed connectivity
        # masks, neuron counts). 28 odors / 21 OR receptor types are the dataset
        # dimensions; target_sparsity=0.10 only matters for the (legacy) in-class
        # loss and is harmless at eval. Inject the canonical noise config here so
        # the freshly-built model matches the regime its weights were trained in.
        model = SpikingConnectomeConstrainedModel.from_data_dir(
            data_dir, n_odors=28, n_or_types=21, target_sparsity=0.10,
            params=REALISTIC_PARAMS, include_nonad=include_nonad,
        )
        # weights_only=True is the safe torch.load mode: load tensors only, no pickled code.
        model.load_state_dict(torch.load(model_dir / f'model_seed{seed}.pt', weights_only=True))
        # Set the model's n_steps to the canonical value (matches the default) with the
        # canonical 30 used in the paper, for BOTH the AL loop and the KC loop.
        model.n_steps_al = n_steps
        model.n_steps_kc = n_steps
        model.eval()  # disable any train-only behaviour (e.g. parameter clamping hooks)
        models.append(model)
    return models


# ---------------------------------------------------------------------------
# Basic evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, test_loader):
    """Evaluate model accuracy and sparsity on test data.

    Runs the full pipeline over a DataLoader, accumulating decoder accuracy, the
    KC population sparsity reported by the model, and the raw OR / PN / KC spike
    patterns so that stage-wise representational cosine similarity can be measured.

    Args:
        model: a built ``SpikingConnectomeConstrainedModel``.
        test_loader: iterable yielding (bx, by) where bx is (B, n_or) OR drive
            and by is (B,) integer odor labels.

    Returns:
        dict with:
            accuracy : fraction of correctly decoded odors (float).
            sparsity : mean reported KC sparsity over batches (float).
            or_sim/pn_sim/kc_sim : mean off-diagonal cosine similarity of the OR,
                PN-spike, and KC-spike patterns respectively (decorrelation proxy).
    """
    model.eval()
    correct, total, sparsities = 0, 0, []
    or_all, pn_all, kc_all = [], [], []
    with torch.no_grad():  # pure inference: no autograd graph, no surrogate gradients needed
        for bx, by in test_loader:
            # return_all=True yields the intermediate spike tensors + sparsity in `info`.
            logits, info = model(bx, return_all=True)
            correct += (logits.argmax(-1) == by).sum().item()  # argmax over the 28-way decoder
            total += len(by)
            sparsities.append(info['sparsity'])
            or_all.append(bx)                          # OR drive: (B, n_or)
            pn_all.append(info['pn_spikes'].float())   # PN spike counts/rates: (B, n_pn)
            kc_all.append(info['kc_spikes'].float())   # KC spike counts/rates: (B, n_kc)
    return {
        'accuracy': correct / total,
        'sparsity': np.mean(sparsities),
        # Concatenate across batches then take mean pairwise cosine similarity.
        # Lower similarity downstream (kc < pn < or) indicates decorrelation.
        'or_sim': cosine_sim_matrix(torch.cat(or_all)),
        'pn_sim': cosine_sim_matrix(torch.cat(pn_all)),
        'kc_sim': cosine_sim_matrix(torch.cat(kc_all)),
    }


# ---------------------------------------------------------------------------
# Decorrelation
# ---------------------------------------------------------------------------

def compute_per_pair_decorrelation(model, or_responses, n_trials=10, noise_std=0.3):
    """Compute per-pair ratio decorrelation (CANONICAL method).

    For each odor pair, compute ratio of downstream to upstream cosine similarity.
    Averaging these *per-pair* ratios (rather than the ratio of averaged
    similarities) is the conservative, paper-canonical definition. A ratio < 1
    means the downstream stage pulled the two odors apart.

    Pipeline: noisy_forward_pass runs each odor through the network n_trials times
    with multiplicative input noise; the per-odor mean over trials gives one OR,
    PN, and KC pattern per odor, which feed per_pair_similarity_ratios.

    Args:
        model: built model (set to eval inside noisy_forward_pass).
        or_responses: (n_odors, n_or) tensor of baseline OR drive per odor.
        n_trials: noisy repeats per odor used to estimate the mean pattern.
        noise_std: multiplicative input-noise CV (0.3 = 30%).

    Returns:
        dict with mean kc/pn/kc-pn similarity ratios and the corresponding
        decorrelation percentages (total = OR->KC, al = OR->PN, mb = PN->KC).
    """
    # Fixed seed 99999 makes the decorrelation metric deterministic across runs.
    data = noisy_forward_pass(model, or_responses, n_trials, noise_std, seed=99999)
    # Collapse the n_trials axis to one mean pattern per odor at each stage.
    or_pats = np.array([p.mean(axis=0) for p in data['or_patterns']])   # (n_odors, n_or)
    pn_pats = np.array([p.mean(axis=0) for p in data['pn_patterns']])   # (n_odors, n_pn)
    kc_pats = np.array([p.mean(axis=0) for p in data['kc_patterns']])   # (n_odors, n_kc)

    ratios = per_pair_similarity_ratios(or_pats, pn_pats, kc_pats)
    return {
        'kc_or_ratio': np.mean(ratios['kc_or_ratios']),  # KC-vs-OR similarity ratio (full pathway)
        'pn_or_ratio': np.mean(ratios['pn_or_ratios']),  # PN-vs-OR (antennal-lobe stage)
        'kc_pn_ratio': np.mean(ratios['kc_pn_ratios']),  # KC-vs-PN (mushroom-body stage)
        # Decorrelation % = 100*(1 - similarity ratio): how much similarity was removed.
        'total_decorr': (1 - np.mean(ratios['kc_or_ratios'])) * 100,
        'al_decorr': (1 - np.mean(ratios['pn_or_ratios'])) * 100,
        'mb_decorr': (1 - np.mean(ratios['kc_pn_ratios'])) * 100,
    }


def compute_mean_sim_decorrelation(model, or_responses, n_trials=10, noise_std=0.3):
    """Compute mean-similarity decorrelation (used for training monitoring).

    Less conservative than per-pair method. Uses mean pairwise cosine similarity
    across all patterns, then computes ratios. Because it divides averaged
    similarities (instead of averaging per-pair ratios) it tends to report higher
    decorrelation; kept as a cheap monitor, not the headline metric.

    Args:
        model: built model.
        or_responses: (n_odors, n_or) baseline OR drive.
        n_trials: noisy repeats per odor.
        noise_std: multiplicative input-noise CV.

    Returns:
        dict with the OR/PN/KC mean similarity scalars and the al/mb/total
        decorrelation percentages derived from their ratios.
    """
    data = noisy_forward_pass(model, or_responses, n_trials, noise_std, seed=99999)
    # cosine_sim_matrix expects a torch tensor, so wrap the per-odor mean patterns.
    or_pats = torch.from_numpy(np.array([p.mean(axis=0) for p in data['or_patterns']]))
    pn_pats = torch.from_numpy(np.array([p.mean(axis=0) for p in data['pn_patterns']]))
    kc_pats = torch.from_numpy(np.array([p.mean(axis=0) for p in data['kc_patterns']]))
    or_sim = cosine_sim_matrix(or_pats)  # scalar: mean off-diagonal cosine similarity
    pn_sim = cosine_sim_matrix(pn_pats)
    kc_sim = cosine_sim_matrix(kc_pats)
    return {
        'or_sim': or_sim, 'pn_sim': pn_sim, 'kc_sim': kc_sim,
        # max(..., 1e-8) guards against divide-by-zero when an upstream stage is fully decorrelated.
        'al_decorr': (1 - pn_sim / max(or_sim, 1e-8)) * 100,
        'mb_decorr': (1 - kc_sim / max(pn_sim, 1e-8)) * 100,
        'total_decorr': (1 - kc_sim / max(or_sim, 1e-8)) * 100,
    }


# ---------------------------------------------------------------------------
# Mancini APL inhibition test
# ---------------------------------------------------------------------------

def run_mancini_test(model, carbachol=1e-10, apl_inject=0.7, n_trials=20):
    """Mancini APL inhibition test. Expected ratio ~2.0.

    Replicates Mancini 2023: carbachol-like tonic drive is injected into all KCs
    to elicit baseline spiking; then the APL is also optogenetically activated.
    Because the APL applies divisive (shunting) inhibition onto KCs, the extra
    APL drive should roughly halve KC spiking, giving baseline/boosted ~= 2.0.

    Inputs are zero OR drive (torch.zeros(1, 21)) so the only excitation is the
    injected current — isolating the APL effect from odor-driven activity.

    Args:
        model: built model (eval mode forced internally).
        carbachol: KC current injection (kc_inject_current) emulating the ACh
            agonist; small constant added to every KC's input current.
        apl_inject: extra APL activation added on the boosted trials
            (apl_inject_current); 0.0 on baseline trials.
        n_trials: noisy repeats per condition (noise comes from REALISTIC_PARAMS).

    Returns:
        dict with the spike ratio, the two mean spike counts, and a boolean
        `passes` flag (True when the ratio lands in the expected [1.5, 2.5] band).
    """
    model.eval()
    with torch.no_grad():
        baseline_spikes = []
        for _ in range(n_trials):
            # Baseline: carbachol drive to KCs, NO extra APL activation.
            _, info = model(torch.zeros(1, 21), return_all=True,
                           kc_inject_current=carbachol, apl_inject_current=0.0)
            baseline_spikes.append(info['kc_spikes'].sum().item())  # total KC spikes this trial
        boosted_spikes = []
        for _ in range(n_trials):
            # Boosted: same KC drive PLUS optogenetic APL activation -> stronger inhibition.
            _, info = model(torch.zeros(1, 21), return_all=True,
                           kc_inject_current=carbachol, apl_inject_current=apl_inject)
            boosted_spikes.append(info['kc_spikes'].sum().item())
    baseline_mean = np.mean(baseline_spikes)
    boosted_mean = np.mean(boosted_spikes)
    # max(..., 0.1) avoids divide-by-zero / huge ratios if the APL fully silences KCs.
    ratio = baseline_mean / max(boosted_mean, 0.1)
    return {
        'ratio': ratio,
        'baseline_spikes': baseline_mean,
        'boosted_spikes': boosted_mean,
        'passes': 1.5 <= ratio <= 2.5,  # acceptance band centred on the expected ~2.0
    }


# ---------------------------------------------------------------------------
# Per-odor evaluation
# ---------------------------------------------------------------------------

def evaluate_per_odor(model, or_responses, odor_names, n_trials=20,
                      noise_std=0.3, seed=42):
    """Evaluate per-odor accuracy and sparsity.

    For each odor, run n_trials noisy forward passes and measure how often the
    decoder picks the correct class, the mean KC sparsity, and the mean KC spike
    pattern. The odor index doubles as the ground-truth label here (the trial's
    true class is `odor_idx`).

    Args:
        model: built model.
        or_responses: (n_odors, n_or) baseline OR drive.
        odor_names: sequence whose length defines n_odors.
        n_trials: noisy repeats per odor.
        noise_std: multiplicative input-noise CV.
        seed: RNG seed for the noise (varied by caller for independent samples).

    Returns:
        (per_odor_acc, per_odor_sparsity, per_odor_kc) lists, one entry per odor:
            per_odor_acc      : decode accuracy in [0, 1].
            per_odor_sparsity : mean KC sparsity.
            per_odor_kc       : (n_kc,) mean KC spike pattern (torch tensor).
    """
    data = noisy_forward_pass(model, or_responses, n_trials, noise_std, seed)
    n_odors = len(odor_names)
    per_odor_acc, per_odor_sparsity, per_odor_kc = [], [], []
    for odor_idx in range(n_odors):
        logits = data['logits'][odor_idx]  # (n_trials, n_classes) for this odor
        # Count trials whose argmax decode equals this odor's index (true label).
        correct = sum(1 for l in logits if np.argmax(l) == odor_idx)
        per_odor_acc.append(correct / n_trials)
        per_odor_sparsity.append(np.mean(data['sparsities'][odor_idx]))
        # Mean KC spike pattern across trials, as a float torch tensor (n_kc,).
        kc_mean = torch.from_numpy(data['kc_patterns'][odor_idx]).float().mean(dim=0)
        per_odor_kc.append(kc_mean)
    return per_odor_acc, per_odor_sparsity, per_odor_kc


def evaluate_per_odor_all_models(models, or_responses, odor_names,
                                 n_trials=20, noise_std=0.3):
    """Evaluate per-odor metrics across ALL models.

    Aggregates evaluate_per_odor over the seed ensemble: accuracy and sparsity are
    summarised as (mean, std) across models per odor; the representative KC pattern
    is taken from the single best-accuracy model.

    Args:
        models: list of built models (the seed ensemble).
        or_responses: (n_odors, n_or) baseline OR drive.
        odor_names: sequence defining n_odors.
        n_trials: noisy repeats per odor.
        noise_std: multiplicative input-noise CV.

    Returns:
        (per_odor_acc, per_odor_sparsity, per_odor_kc) where acc/sparsity
        are lists of (mean, std) tuples and per_odor_kc is from the best model.
    """
    n_odors = len(odor_names)
    all_accs, all_sps = [], []
    for m_idx, model in enumerate(models):
        # Offset the seed per model (42 + m_idx) so each model sees independent noise draws.
        acc, sp, _ = evaluate_per_odor(model, or_responses, odor_names,
                                       n_trials, noise_std, seed=42 + m_idx)
        all_accs.append(acc)
        all_sps.append(sp)
    all_accs = np.array(all_accs)  # (n_models, n_odors)
    all_sps = np.array(all_sps)    # (n_models, n_odors)
    # Mean +/- std across models, computed column-wise (per odor).
    per_odor_acc = [(float(np.mean(all_accs[:, i])), float(np.std(all_accs[:, i])))
                    for i in range(n_odors)]
    per_odor_sparsity = [(float(np.mean(all_sps[:, i])), float(np.std(all_sps[:, i])))
                         for i in range(n_odors)]
    # Pick the model with the highest mean per-odor accuracy as the exemplar for KC patterns.
    best_idx = np.argmax([np.mean(accs) for accs in all_accs])
    _, _, per_odor_kc = evaluate_per_odor(models[best_idx], or_responses, odor_names,
                                          n_trials, noise_std, seed=42 + best_idx)
    return per_odor_acc, per_odor_sparsity, per_odor_kc


# ---------------------------------------------------------------------------
# Cross-model consistency
# ---------------------------------------------------------------------------

def compute_cross_model_consistency(models, or_responses, odor_names,
                                    n_trials=10, noise_std=0.3):
    """Compute prediction consistency across models.

    Measures how often independently-trained seeds agree on the decoded class for
    the *same* noisy stimulus. A single shared RNG is advanced across all models,
    so model j on trial t sees the exact same noise instance as model i did —
    isolating genuine model disagreement from differing noise.

    Args:
        models: list of built models (the ensemble).
        or_responses: (n_odors, n_or) baseline OR drive.
        odor_names: sequence defining n_odors.
        n_trials: noisy repeats per odor.
        noise_std: multiplicative input-noise CV.

    Returns:
        (mean_consistency, per_odor_consistency_list): mean fraction of agreeing
        model pairs overall, and the same fraction per odor.
    """
    rng = np.random.default_rng(12345)  # one shared RNG => identical noise sequence per (odor, trial)
    n_odors = len(odor_names)
    n_models = len(models)
    all_preds = np.zeros((n_odors, n_trials, n_models), dtype=int)  # stored decoded class per cell
    for m_idx, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            for odor_idx in range(n_odors):
                for trial in range(n_trials):
                    x = or_responses[odor_idx].clone()
                    # Multiplicative Gaussian noise on the OR drive (1 + N(0, noise_std)).
                    noise = torch.from_numpy(
                        rng.normal(0, noise_std, x.shape).astype(np.float32))
                    x = torch.clamp(x * (1.0 + noise), min=0).unsqueeze(0)  # clamp>=0, add batch dim
                    logits = model(x)
                    all_preds[odor_idx, trial, m_idx] = logits.argmax(-1).item()
    consistency_per_odor = []
    for odor_idx in range(n_odors):
        agreements, total_pairs = 0, 0
        for trial in range(n_trials):
            preds = all_preds[odor_idx, trial, :]  # the n_models predictions for this stimulus
            # Count agreeing unordered model pairs (i < j) for this trial.
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    if preds[i] == preds[j]:
                        agreements += 1
                    total_pairs += 1
        consistency_per_odor.append(agreements / total_pairs if total_pairs > 0 else 0)
    return np.mean(consistency_per_odor), consistency_per_odor


def compute_kc_consistency_per_odor(models, or_responses, odor_names,
                                    n_trials=10, noise_std=0.3):
    """Compute KC pattern consistency per odor across models.

    Like compute_cross_model_consistency but in the continuous KC-representation
    space: each model's mean KC spike vector for an odor is L2-normalised, and the
    pairwise cosine similarity between models quantifies how alike their learned
    KC codes are. Uses a shared RNG for matched noise across models.

    Args:
        models: list of built models.
        or_responses: (n_odors, n_or) baseline OR drive.
        odor_names: sequence defining n_odors.
        n_trials: noisy repeats per odor.
        noise_std: multiplicative input-noise CV.

    Returns:
        (mean_consistency, list of (mean, std) tuples): mean pairwise cosine
        similarity across odors, and per-odor (mean, std) of those similarities.
    """
    rng = np.random.default_rng(54321)  # shared RNG (different stream than prediction-consistency)
    n_odors = len(odor_names)
    n_models = len(models)
    consistency_per_odor = []
    for odor_idx in range(n_odors):
        all_kc = []
        for model in models:
            model.eval()
            kc_trials = []
            with torch.no_grad():
                for _ in range(n_trials):
                    x = or_responses[odor_idx].clone()
                    noise = torch.from_numpy(
                        rng.normal(0, noise_std, x.shape).astype(np.float32))
                    x = torch.clamp(x * (1.0 + noise), min=0).unsqueeze(0)
                    _, info = model(x, return_all=True)
                    kc_trials.append(info['kc_spikes'].squeeze(0))  # (n_kc,) this trial
            all_kc.append(torch.stack(kc_trials).mean(dim=0))  # mean KC vector for this model
        kc_stack = torch.stack(all_kc)  # (n_models, n_kc)
        # L2-normalise each model's KC vector so the Gram matrix below is cosine similarity.
        norms = kc_stack.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = kc_stack / norms
        corr_matrix = normalized @ normalized.T  # (n_models, n_models) cosine similarities
        # Upper-triangular (diagonal=1) mask selects each unordered model pair once.
        mask = torch.triu(torch.ones(n_models, n_models), diagonal=1).bool()
        pairwise = corr_matrix[mask].cpu().numpy()
        consistency_per_odor.append((float(np.mean(pairwise)), float(np.std(pairwise))))
    return np.mean([c[0] for c in consistency_per_odor]), consistency_per_odor


# ---------------------------------------------------------------------------
# Centroid accuracy
# ---------------------------------------------------------------------------

def centroid_accuracy(model, or_responses, n_trials=20, noise_std=0.3):
    """Compute centroid-based classification accuracy.

    A decoder-free readout of KC-code separability: build a per-odor centroid from
    the first half of the noisy trials, then classify each held-out (second-half)
    trial by nearest centroid (cosine similarity). High accuracy means the KC
    representation alone keeps odors linearly separable without the trained decoder.

    Args:
        model: built model.
        or_responses: (n_odors, n_or) baseline OR drive; len() gives n_odors.
        n_trials: total noisy trials per odor (split 50/50 train/test).
        noise_std: multiplicative input-noise CV.

    Returns:
        Fraction of held-out trials classified to the correct centroid (float;
        0.0 if no test trials were classifiable).
    """
    data = noisy_forward_pass(model, or_responses, n_trials, noise_std, seed=12345)
    n_odors = len(or_responses)
    half = n_trials // 2
    # Centroid = mean KC pattern over the FIRST half of trials, per odor.
    centroids = np.array([data['kc_patterns'][i][:half].mean(axis=0)
                          for i in range(n_odors)])
    correct, total = 0, 0
    for odor_idx in range(n_odors):
        # Classify each SECOND-half trial against the centroids.
        for trial in data['kc_patterns'][odor_idx][half:]:
            pred = centroid_classify(centroids, trial)
            if pred == -1:  # near-zero KC pattern -> unclassifiable, skip (not counted)
                continue
            if pred == odor_idx:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Circuit property extraction
# ---------------------------------------------------------------------------

def extract_gap_junction_info(model):
    """Extract gap junction conductances and LN->PN split.

    Reads the learned electrical-coupling (gap-junction) conductances and the
    inhibitory/excitatory split of the LN->PN pathway directly from the antennal
    lobe submodule. Conductances are stored in LOG space (nn.Parameter holds
    log g) so they stay strictly positive during optimisation; we exponentiate to
    recover the physical conductance (siemens). Gap-junction current obeys
    I = g * (V_pre - V_post), so g is the coupling strength reported here.

    Args:
        model: built model whose ``antennal_lobe`` holds the parameters.

    Returns:
        (gap_info, ln_pn_split):
            gap_info: dict {'ln_ln','pn_pn','eln_pn'} -> conductance (S) or None
                if that gap-junction type is absent from this model.
            ln_pn_split: dict with inhibitory/excitatory LN->PN strengths and the
                excitatory-LN counts, populated only for attributes present.
    """
    al = model.antennal_lobe
    gap_info = {}
    # Three gap-junction families: LN-LN, PN-PN (sister PNs), and eLN-PN.
    for attr, key in [('log_g_gap_ln', 'ln_ln'), ('log_g_gap_pn', 'pn_pn'),
                      ('log_g_gap_eln_pn', 'eln_pn')]:
        val = getattr(al, attr, None)
        # exp(log_g) -> conductance in siemens; None when the model lacks this coupling.
        gap_info[key] = float(torch.exp(val).item()) if val is not None else None

    ln_pn_split = {}
    if hasattr(al, 'ln_pn'):
        # Inhibitory LN->PN synaptic strength, again exp() of a log-space parameter.
        ln_pn_split['inhibitory_strength'] = float(torch.exp(al.ln_pn.log_strength).item())
    if hasattr(al, 'ln_pn_excit'):
        # Some LNs are excitatory; their LN->PN strength is stored on a separate layer.
        ln_pn_split['excitatory_strength'] = float(torch.exp(al.ln_pn_excit.log_strength).item())
    if hasattr(al, 'is_excitatory_ln'):
        # Boolean mask over LNs: count excitatory vs. total to report the split.
        ln_pn_split['n_excitatory_ln'] = int(al.is_excitatory_ln.sum().item())
        ln_pn_split['n_total_ln'] = len(al.is_excitatory_ln)

    return gap_info, ln_pn_split


def extract_nonad_strengths(model):
    """Extract non-AD connection strengths.

    The connectome distinguishes axo-dendritic (AD) synapses from "non-AD"
    contacts (e.g. axo-axonic / dendro-dendritic). Each modelled non-AD pathway
    carries a single learnable scalar ``log_strength``; this returns the physical
    strengths via exp() (log-space storage keeps them positive during training).

    Args:
        model: built model with optional non-AD sublayers on ``antennal_lobe``
            and ``kc_layer``.

    Returns:
        dict mapping pathway name -> strength (float), including only the non-AD
        pathways actually present on this model.
    """
    nonad = {}
    # Each tuple is (output key, the optional non-AD sublayer or None).
    for name, layer in [
        ('orn_ln_nonad', getattr(model.antennal_lobe, 'orn_ln_nonad', None)),
        ('ln_pn_nonad', getattr(model.antennal_lobe, 'ln_pn_nonad', None)),
        ('ln_pn_excit_nonad', getattr(model.antennal_lobe, 'ln_pn_excit_nonad', None)),
        ('ln_ln_nonad', getattr(model.antennal_lobe, 'ln_ln_nonad', None)),
        ('pn_ln_nonad', getattr(model.antennal_lobe, 'pn_ln_nonad', None)),
        ('ln_orn_nonad', getattr(model.antennal_lobe, 'ln_orn_nonad', None)),
        ('pn_kc_nonad', getattr(model.kc_layer, 'pn_kc_nonad', None)),
    ]:
        if layer is not None:
            # np.exp on the raw .item() (a Python float) recovers strength from log space.
            nonad[name] = float(np.exp(layer.log_strength.item()))
    return nonad


# ---------------------------------------------------------------------------
# Concentration invariance
# ---------------------------------------------------------------------------

def run_concentration_invariance(model, or_responses, seed, concentrations=None,
                                 hill_ec50=1.0, hill_n=1, n_trials=10, noise_std=0.3):
    """Run concentration invariance test for a single model.

    Probes whether the circuit recognises an odor across a wide concentration
    range. Each concentration is mapped through a Hill dose-response onto an
    "effective" scale, used to interpolate every odor's OR drive between the
    spontaneous firing rate (SFR, the no-odor baseline at or_responses[0]) and its
    full-strength response. The network is then run at each concentration and we
    check four properties:

        sublinear_pn_gain         : PN output grows less than OR input (gain control).
        flat_kc_activity          : total KC activity stays roughly constant.
        robust_classification     : decoder accuracy holds over moderate concentrations.
        odor_identity_preservation: KC codes at off-baseline concentrations still
                                     classify to the right baseline centroid.

    Args:
        model: built model.
        or_responses: (n_odors, n_or) drive; index 0 is the SFR baseline, indices
            1: are the full-strength per-odor responses.
        seed: RNG seed for the input noise (varied per model by the caller).
        concentrations: list of relative concentrations; defaults to a 7-point
            log-ish sweep if None.
        hill_ec50: Hill half-maximal concentration (relative units).
        hill_n: Hill coefficient (cooperativity); 1 = non-cooperative.
        n_trials: noisy repeats per odor per concentration.
        noise_std: multiplicative input-noise CV.

    Returns:
        (conc_results, tests):
            conc_results: dict keyed by str(concentration) with per-stage activity,
                decoder accuracy, centroid accuracy, and similarity-to-baseline.
            tests: dict of the four pass/fail summary properties plus the raw
                gain ranges and accuracies they were computed from.
    """
    if concentrations is None:
        # Canonical 7-point concentration sweep (relative to reference c=1.0).
        concentrations = [0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

    sfr = or_responses[0]  # spontaneous firing rate vector (no-odor baseline), (n_or,)
    rng = np.random.default_rng(seed)
    model.eval()

    def get_patterns(scaled_resp):
        """Run all odors at a given concentration and collect mean patterns + accuracy.

        Args:
            scaled_resp: (n_odors, n_or) OR drive already scaled to this concentration.

        Returns:
            (or_pats, pn_pats, kc_pats, kc_trials_all, decoder_acc):
                *_pats are (n_odors, n_*) trial-mean patterns; kc_trials_all is a
                list of (n_trials, n_kc) arrays (raw per-trial KC for centroid use);
                decoder_acc is the fraction of trials decoded correctly.
        """
        or_pats, pn_pats, kc_pats, kc_trials_all = [], [], [], []
        decoder_correct, decoder_total = 0, 0
        with torch.no_grad():
            for odor_idx in range(len(scaled_resp)):
                or_t, pn_t, kc_t = [], [], []
                # Note: ground-truth label is odor_idx+1 because index 0 is the SFR/no-odor row.
                true_label = odor_idx + 1
                for _ in range(n_trials):
                    base = scaled_resp[odor_idx]
                    noise = torch.from_numpy(
                        rng.normal(0, noise_std, base.shape).astype(np.float32))
                    x = (base * (1.0 + noise)).clamp(0)  # multiplicative noise, clamp>=0
                    logits, info = model(x.unsqueeze(0), return_all=True)
                    or_t.append(x.numpy())
                    pn_t.append(info['pn_spikes'].float().squeeze().numpy())
                    kc_t.append(info['kc_spikes'].float().squeeze().numpy())
                    if logits.argmax(-1).item() == true_label:
                        decoder_correct += 1
                    decoder_total += 1
                or_pats.append(np.mean(or_t, 0))     # trial-mean OR pattern for this odor
                pn_pats.append(np.mean(pn_t, 0))     # trial-mean PN pattern
                kc_pats.append(np.mean(kc_t, 0))     # trial-mean KC pattern
                kc_trials_all.append(np.array(kc_t)) # keep raw trials for centroid classification
        return (np.array(or_pats), np.array(pn_pats), np.array(kc_pats),
                kc_trials_all, decoder_correct / max(decoder_total, 1))

    # Build the reference (c=1.0) representation. Effective concentration at c=1
    # normalises the Hill curve so the reference matches full-strength responses.
    baseline_eff = hill_effective_concentration(1.0, hill_ec50, hill_n)
    # Interpolate each odor between SFR and its full response by baseline_eff.
    baseline_resp = sfr.unsqueeze(0) + (or_responses[1:] - sfr.unsqueeze(0)) * baseline_eff
    baseline_or, baseline_pn, baseline_kc, _, _ = get_patterns(baseline_resp)
    kc_centroids = baseline_kc  # baseline KC patterns serve as the per-odor centroids
    sfr_np = sfr.numpy()

    def representation_similarity(pats_test, pats_base):
        """Mean per-odor cosine similarity between two stacks of patterns.

        Args:
            pats_test/pats_base: (n_odors, n_features) arrays compared row-by-row.

        Returns:
            Mean cosine similarity across odors (float).
        """
        return float(np.mean([cosine_sim_pair(pats_test[i], pats_base[i])
                              for i in range(len(pats_test))]))

    conc_results = {}
    for c in concentrations:
        eff = hill_effective_concentration(c, hill_ec50, hill_n)  # Hill-mapped effective scale
        # Scale every odor's drive for this concentration (clamped to be non-negative).
        scaled = (sfr.unsqueeze(0) + (or_responses[1:] - sfr.unsqueeze(0)) * eff).clamp(0)
        or_p, pn_p, kc_p, kc_trials, dec_acc = get_patterns(scaled)

        # Centroid classification: classify each raw KC trial against the baseline centroids.
        centroid_correct, centroid_total = 0, 0
        for odor_idx, trials in enumerate(kc_trials):
            for trial in trials:
                pred = centroid_classify(kc_centroids, trial)
                if pred == -1:  # unclassifiable (near-zero KC pattern), skip
                    continue
                if pred == odor_idx:
                    centroid_correct += 1
                centroid_total += 1

        # Similarity to baseline. For OR we subtract the SFR first so the similarity
        # reflects the odor-specific component, not the shared spontaneous baseline.
        or_test_sub = or_p - sfr_np
        or_base_sub = baseline_or - sfr_np
        conc_results[str(c)] = {
            'effective_c': eff, 'decoder_acc': dec_acc,
            # mean_* = mean over odors of the total activity (sum over units) at each stage.
            'mean_or': float(np.mean([np.sum(p) for p in or_p])),
            'mean_pn': float(np.mean([np.sum(p) for p in pn_p])),
            'mean_kc': float(np.mean([np.sum(p) for p in kc_p])),
            'kc_centroid_acc': centroid_correct / max(centroid_total, 1),
            'or_sim': representation_similarity(or_test_sub, or_base_sub),
            'pn_sim': representation_similarity(pn_p, baseline_pn),
            'kc_sim': representation_similarity(kc_p, baseline_kc),
        }

    c_vals = [float(c) for c in concentrations]
    # "Range" = activity at the HIGHEST concentration divided by activity at the LOWEST.
    # A smaller range means flatter (more concentration-invariant) activity at that stage.
    or_range = conc_results[str(c_vals[-1])]['mean_or'] / max(conc_results[str(c_vals[0])]['mean_or'], 1e-8)
    pn_range = conc_results[str(c_vals[-1])]['mean_pn'] / max(conc_results[str(c_vals[0])]['mean_pn'], 1e-8)
    kc_range = conc_results[str(c_vals[-1])]['mean_kc'] / max(conc_results[str(c_vals[0])]['mean_kc'], 1e-8)
    moderate_concs = [c for c in c_vals if 0.3 <= c <= 5.0]  # "moderate" band for the accuracy test
    mean_moderate_acc = np.mean([conc_results[str(c)]['decoder_acc'] for c in moderate_concs])
    # Cross-concentration centroid accuracy excludes the c=1.0 reference (the centroids' own source).
    non_baseline_accs = [conc_results[str(c)]['kc_centroid_acc'] for c in c_vals if c != 1.0]
    mean_cross_conc_acc = np.mean(non_baseline_accs)

    tests = {
        # PN gain is sublinear if PN activity range < OR activity range (gain control in the AL).
        'sublinear_pn_gain': bool(pn_range < or_range),
        # KC activity is "flat" if it varies less than 20% across the full concentration range.
        'flat_kc_activity': bool(kc_range < 1.2),
        # Decoder is robust if it beats 50% accuracy averaged over the moderate band.
        'robust_classification': bool(mean_moderate_acc > 0.5),
        # Graded PASS/PARTIAL/FAIL on whether off-baseline codes still classify correctly.
        'odor_identity_preservation': ('PASS' if mean_cross_conc_acc > 0.6
                                       else 'PARTIAL' if mean_cross_conc_acc > 0.3
                                       else 'FAIL'),
        'or_range': float(or_range), 'pn_range': float(pn_range), 'kc_range': float(kc_range),
        'mean_moderate_acc': float(mean_moderate_acc),
        'mean_cross_conc_centroid_acc': float(mean_cross_conc_acc),
    }
    return conc_results, tests




def compute_few_param_cv(all_params, min_params=10):
    """Compute coefficient of variation for few-parameter groups.

    Quantifies how reproducible each parameter is across the seed ensemble. For a
    given category and element, CV = std / |mean| across models — small CV means
    the seeds converged on nearly the same value (a consistent biological
    parameter). Categories with many elements (>= min_params) are handled
    elsewhere via Pearson correlation (compute_pairwise_correlations); this
    function deliberately covers only the *small* groups, where element-wise CV is
    the more informative statistic.

    Args:
        all_params: list (per model) of dicts {category -> 1D parameter tensor},
            as produced by extract_all_parameters().
        min_params: groups with at least this many elements are skipped here.

    Returns:
        dict mapping category -> {mean_cv, per_element_cv, n_params,
        element_means, element_stds}, for the small groups that had >=2 models
        and at least one element with a non-negligible mean.
    """
    # Union of every category seen across models (models may differ in which exist).
    all_categories = set()
    for mp in all_params:
        all_categories.update(mp.keys())

    cv_results = {}
    for cat in sorted(all_categories):
        if cat in ('Total', 'Overall'):  # aggregate buckets, not real parameter groups
            continue
        cat_params = [mp[cat].cpu().numpy() for mp in all_params if cat in mp]
        if len(cat_params) < 2:  # need >=2 models to compute a cross-model std
            continue
        n_params = len(cat_params[0])
        if n_params >= min_params:  # large groups are handled by correlation, not CV
            continue
        stacked = np.array(cat_params)        # (n_models, n_params)
        means = np.mean(stacked, axis=0)      # per-element mean across models
        stds = np.std(stacked, axis=0)        # per-element std across models
        # Skip elements whose mean is ~0 (CV is undefined / explodes there).
        valid = np.abs(means) > 1e-10
        if valid.any():
            cvs = stds[valid] / np.abs(means[valid])  # coefficient of variation per valid element
            cv_results[cat] = {
                'mean_cv': float(np.mean(cvs)),
                'per_element_cv': [float(c) for c in cvs],
                'n_params': n_params,
                'element_means': [float(m) for m in means],
                'element_stds': [float(s) for s in stds],
            }
    return cv_results
