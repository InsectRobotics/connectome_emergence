"""Shared primitive functions for connectome model analysis.

This module is a small, dependency-light toolbox of *pure* helper routines that
are reused by the two heavier analysis modules in this package:
``analysis/compute.py`` (which derives the paper's quantitative metrics from
trained checkpoints) and ``analysis/plotting.py`` (which renders the figures).
Keeping these primitives here avoids duplicating the exact metric definitions
across the recompute scripts, the notebook, and the figure code, so every part
of the pipeline measures decorrelation, accuracy, and parameter consistency in
identical fashion.

Where this sits in the overall pipeline: the spiking model maps
OR responses (Kreher 2008) -> ORN (LIF) -> LN (LIF) -> PN (LIF) ->
KC (two-compartment) <- APL (graded divisive inhibition) -> linear decoder over
28 odors. After a model is trained (connectivity fixed by the Winding 2023
connectome; ~449 biological parameters learned), these utilities run the trained
model under biological noise and quantify its behaviour:

  - Cosine similarity (pairwise, matrix) -- the geometric distance measure used
    throughout to compare population activity patterns.
  - Noisy forward pass -- the single, canonical noisy-evaluation loop shared by
    every downstream metric so they all see statistically comparable trials.
  - Per-pair similarity ratios -- the canonical decorrelation method (downstream
    similarity normalised by upstream similarity, per odor pair).
  - Centroid classification -- nearest-centroid (cosine) decoding of KC patterns.
  - Hill dose-response -- concentration scaling for the concentration experiments.
  - Parameter extraction and cross-model consistency metrics -- pull learned
    biological parameters out of a checkpoint and measure how reproducible they
    are across independently trained seeds.

All units follow SI conventions internally (volts, siemens, seconds); a few
helpers convert to mV / nS only for human-readable reporting.
"""

import numpy as np
import torch
from itertools import combinations
from scipy.stats import pearsonr  # Pearson correlation for cross-seed parameter consistency


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_sim_pair(a, b):
    """Cosine similarity between two numpy vectors.

    Computes ``<a, b> / (||a|| ||b||)``, the cosine of the angle between the two
    activity patterns. This is the scale-invariant similarity used everywhere in
    the decorrelation analysis: it ignores overall firing magnitude and reports
    only how *aligned* two population responses are.

    Args:
        a: 1-D numpy vector (e.g. a mean OR/PN/KC pattern, shape (n_features,)).
        b: 1-D numpy vector of the same length as ``a``.

    Returns:
        float in roughly [-1, 1]. Returns ``0.0`` when either vector has a
        near-zero norm (an all-silent pattern), since the cosine is undefined
        there and 0 is the neutral "no information" value.
    """
    na, nb = np.linalg.norm(a), np.linalg.norm(b)  # L2 norms of the two patterns
    if na < 1e-8 or nb < 1e-8:
        # Degenerate (silent) pattern: cosine is undefined; treat as uncorrelated.
        return 0.0
    return float(np.dot(a, b) / (na * nb))  # dot product normalised by both norms


def cosine_sim_matrix(patterns):
    """Mean absolute pairwise cosine similarity between pattern rows.

    Works on torch tensors. Returns the mean over all off-diagonal pairs.

    This is a vectorised, "how correlated is this set of patterns on average"
    summary: it builds the full pattern-by-pattern cosine matrix and averages the
    magnitude of every off-diagonal entry. Taking the absolute value treats
    strong negative correlations as just as "non-decorrelated" as strong positive
    ones. Lower values mean the patterns are more mutually orthogonal (more
    decorrelated), which is the desired KC-layer behaviour.

    Args:
        patterns: (n_patterns, n_features) tensor. A 1-D input is promoted to a
            single row; inputs with more than two dims are flattened along all
            trailing axes into (n_patterns, n_features).

    Returns:
        Mean absolute cosine similarity across all off-diagonal pairs (Python
        float). Returns ``1.0`` when there are fewer than two patterns (a single
        pattern is trivially "perfectly similar" to itself).
    """
    if patterns.dim() == 1:
        patterns = patterns.unsqueeze(0)  # (n_features,) -> (1, n_features)
    if patterns.dim() > 2:
        # Collapse any extra dims (e.g. trials/time) into the feature axis.
        patterns = patterns.reshape(patterns.shape[0], -1)
    n = patterns.shape[0]
    if n < 2:
        # No pairs to compare; define the similarity of a singleton set as 1.0.
        return 1.0
    norms = patterns.norm(dim=1, keepdim=True).clamp(min=1e-8)  # per-row L2 norm, floored to avoid /0
    normalized = patterns / norms  # unit-length rows: row dot products are now cosines
    sim_matrix = normalized @ normalized.T  # (n, n) Gram matrix of cosine similarities
    mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)  # True off the diagonal (exclude self-pairs)
    off_diag = sim_matrix[mask]  # flatten all off-diagonal cosines
    return off_diag.abs().mean().item()  # mean |cosine| over every distinct ordered pair


# ---------------------------------------------------------------------------
# Noisy forward pass
# ---------------------------------------------------------------------------

def noisy_forward_pass(model, or_responses, n_trials, noise_std, seed):
    """Run model on all odors with multiplicative noise, collecting per-trial results.

    This is the unified noise loop used by compute_per_pair_decorrelation,
    evaluate_per_odor, and centroid_accuracy. Each uses its own seed for
    reproducibility.

    The model already injects its own six internal biological noise sources; the
    noise applied *here* is an additional input-level olfactory variability term:
    each OR response channel is scaled by ``(1 + eps)`` with ``eps ~ N(0,
    noise_std)``. This is multiplicative (Weber-like / coefficient-of-variation)
    noise, so larger baseline responses fluctuate by larger absolute amounts.
    Running many such trials lets the downstream metrics measure how robustly the
    network separates odors under realistic trial-to-trial variability.

    Args:
        model: SpikingConnectomeConstrainedModel (set to eval mode internally so
            stochastic layers behave deterministically apart from the noise we
            inject; gradients are disabled via ``torch.no_grad``).
        or_responses: (n_odors, n_or) tensor of baseline OR responses
            (dimensionless firing-rate-like activations, one row per odor).
        n_trials: number of noisy trials per odor.
        noise_std: multiplicative noise CV (e.g. 0.3 for 30%).
        seed: random seed for reproducibility (drives a dedicated numpy
            Generator so callers with different seeds get independent draws).

    Returns:
        dict with per-odor lists (each list has n_odors entries):
            or_patterns: list of (n_trials, n_or) numpy arrays -- the actual
                noisy OR inputs that were fed in.
            pn_patterns: list of (n_trials, n_pn) numpy arrays -- PN spike counts.
            kc_patterns: list of (n_trials, n_kc) numpy arrays -- KC spike counts.
            logits: list of (n_trials, n_classes) numpy arrays -- decoder outputs.
            sparsities: list of n_trials-length lists -- fraction of KCs active
                per trial.
    """
    rng = np.random.default_rng(seed)  # dedicated PRNG so noise is reproducible and seed-isolated
    model.eval()  # disable training-time stochasticity (dropout-like / BN-like behaviour)
    n_odors = len(or_responses)
    all_or, all_pn, all_kc, all_logits, all_sp = [], [], [], [], []  # per-odor accumulators
    with torch.no_grad():  # pure inference: no autograd graph, saves memory/time
        for odor_idx in range(n_odors):
            or_t, pn_t, kc_t, logit_t, sp_t = [], [], [], [], []  # per-trial accumulators for this odor
            for _ in range(n_trials):
                base = or_responses[odor_idx]  # (n_or,) noise-free OR response for this odor
                noise = torch.from_numpy(
                    rng.normal(0, noise_std, base.shape).astype(np.float32)  # eps ~ N(0, noise_std), one per OR channel
                )
                x = (base * (1.0 + noise)).clamp(0)  # multiplicative noise, clamped >=0 (rates cannot be negative)
                logits, info = model(x.unsqueeze(0), return_all=True)  # add batch dim -> (1, n_or); request intermediates
                or_t.append(x.numpy())  # store the realised noisy input
                pn_t.append(info['pn_spikes'].float().squeeze().numpy())  # PN spike COUNTS over the time loop, (n_pn,)
                kc_t.append(info['kc_spikes'].float().squeeze().numpy())  # KC spike COUNTS over the time loop, (n_kc,)
                logit_t.append(logits.squeeze().numpy())  # decoder logits over 28 odor classes
                sp_t.append(info['sparsity'])  # scalar fraction of KCs that fired this trial
            all_or.append(np.array(or_t))      # (n_trials, n_or)
            all_pn.append(np.array(pn_t))      # (n_trials, n_pn)
            all_kc.append(np.array(kc_t))      # (n_trials, n_kc)
            all_logits.append(np.array(logit_t))  # (n_trials, n_classes)
            all_sp.append(sp_t)                # list of n_trials sparsity scalars
    return {
        'or_patterns': all_or,
        'pn_patterns': all_pn,
        'kc_patterns': all_kc,
        'logits': all_logits,
        'sparsities': all_sp,
    }


# ---------------------------------------------------------------------------
# Per-pair similarity ratios (canonical decorrelation)
# ---------------------------------------------------------------------------

def per_pair_similarity_ratios(or_pats, pn_pats, kc_pats):
    """Compute per-pair cosine similarity ratios.

    This is the canonical decorrelation method: for each odor pair, compute
    the ratio of downstream similarity to upstream similarity.

    For every unordered pair of odors (i, j) we measure how similar their mean
    patterns are at three stages -- OR (input), PN (antennal-lobe output), and KC
    (mushroom-body code) -- and then form ratios. A ratio below 1 means the
    downstream stage pulled that odor pair *apart* relative to the upstream stage
    (decorrelation); above 1 means it pushed them together. Reporting the ratio
    *per pair* (rather than averaging similarities first) is the canonical choice
    because it normalises each pair by how confusable it already was at the input,
    so pairs that start nearly identical do not dominate the statistic.

    Args:
        or_pats: (n_odors, n_features) mean OR patterns (numpy).
        pn_pats: (n_odors, n_features) mean PN patterns (numpy).
        kc_pats: (n_odors, n_features) mean KC patterns (numpy).

    Returns:
        dict with kc_or_ratios, pn_or_ratios, kc_pn_ratios (lists of floats),
        one entry per valid odor pair:
            kc_or_ratios: KC-vs-OR similarity ratio (full input->code transform).
            pn_or_ratios: PN-vs-OR similarity ratio (antennal-lobe stage only).
            kc_pn_ratios: KC-vs-PN similarity ratio (mushroom-body stage only).
    """
    n_odors = len(or_pats)
    kc_or_ratios, pn_or_ratios, kc_pn_ratios = [], [], []
    for i in range(n_odors):
        for j in range(i + 1, n_odors):  # each unordered pair once (j > i)
            or_sim = cosine_sim_pair(or_pats[i], or_pats[j])  # input-stage similarity
            pn_sim = cosine_sim_pair(pn_pats[i], pn_pats[j])  # AL-output similarity
            kc_sim = cosine_sim_pair(kc_pats[i], kc_pats[j])  # KC-code similarity
            if abs(or_sim) > 1e-8:  # guard against dividing by an ~uncorrelated input pair
                kc_or_ratios.append(kc_sim / or_sim)  # how much OR->KC changed pairwise similarity
                pn_or_ratios.append(pn_sim / or_sim)  # how much OR->PN changed pairwise similarity
            if abs(pn_sim) > 1e-8:  # guard against dividing by an ~uncorrelated PN pair
                kc_pn_ratios.append(kc_sim / pn_sim)  # how much PN->KC changed pairwise similarity
    return {
        'kc_or_ratios': kc_or_ratios,
        'pn_or_ratios': pn_or_ratios,
        'kc_pn_ratios': kc_pn_ratios,
    }


# ---------------------------------------------------------------------------
# Centroid classification
# ---------------------------------------------------------------------------

def centroid_classify(centroids, test_trial):
    """Classify a single trial by nearest centroid (cosine similarity).

    A simple template-matching decoder: the predicted odor is the class whose
    mean (centroid) pattern is the most cosine-similar to the test trial. Using
    cosine (rather than Euclidean) distance makes the decision invariant to the
    overall firing magnitude of the trial, matching the scale-invariant geometry
    used elsewhere in the analysis.

    Args:
        centroids: (n_classes, n_features) numpy array of per-class mean patterns.
        test_trial: (n_features,) numpy array -- the single trial to classify.

    Returns:
        Predicted class index (int), or -1 if trial has near-zero norm (a silent
        trial carries no information to match against any centroid).
    """
    norm_t = np.linalg.norm(test_trial)
    if norm_t < 1e-8:
        return -1  # silent/degenerate trial -> undecodable
    sims = [
        # cosine(test_trial, c); each centroid norm floored to 1e-8 to avoid /0
        np.dot(test_trial, c) / (norm_t * max(np.linalg.norm(c), 1e-8))
        for c in centroids
    ]
    return int(np.argmax(sims))  # index of the most similar centroid = predicted class


# ---------------------------------------------------------------------------
# Hill dose-response
# ---------------------------------------------------------------------------

def hill_effective_concentration(c, ec50=1.0, n=1):
    """Hill dose-response function for concentration scaling.

    Returns the effective concentration relative to reference (c=1.0).

    Implements the standard Hill equation ``response = c^n / (c^n + ec50^n)`` --
    a saturating sigmoid (on log-concentration) that models receptor occupancy --
    and then divides by the response at the reference concentration ``c = 1.0`` so
    the function returns 1.0 at the reference. This lets the concentration
    experiments rescale OR inputs in a biologically plausible, saturating way
    rather than scaling them linearly.

    Args:
        c: concentration (dimensionless, relative to the reference c=1.0).
        ec50: half-maximal effective concentration (Hill K), default 1.0.
        n: Hill coefficient (cooperativity / steepness), default 1
            (non-cooperative, hyperbolic occupancy).

    Returns:
        Effective concentration as a fraction of the reference response (float;
        equals 1.0 when c == 1.0 and ec50 == 1.0).
    """
    response_at_c = (c ** n) / (c ** n + ec50 ** n)  # Hill occupancy at concentration c
    response_at_ref = 1.0 / (1.0 + ec50 ** n)         # Hill occupancy at the reference c = 1.0
    return response_at_c / response_at_ref            # normalise so reference -> 1.0


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------

def extract_all_parameters(model):
    """Extract learnable parameters by category (including non-AD and gap junctions).

    Walks the model's named parameters and bins each one into a human-readable
    biological category (e.g. "ORN v_th", "PN->KC", "Gap LN-LN", "Decoder") by
    pattern-matching on the parameter's dotted name. The categories mirror the
    biological structure of the pipeline (AL neuron properties, AL synapses, KC
    synapses, APL inhibition, gap junctions, decoder). "nonad" variants are the
    non-AD (non-axo-axonic) compartment copies of a given connection type and are
    kept as separate categories so they can be analysed independently.

    Returns dict mapping category name -> concatenated parameter tensor.
    Also includes a 'Total' key with all parameters concatenated.

    Notes:
        - Only trainable parameters (``requires_grad``) are collected; frozen
          connectome masks/buffers are skipped.
        - Parameter values are taken as-stored, which for most weights is the
          raw learned value. The one deliberate exception is ``KC->APL strength``
          (see below), which is exponentiated back into real space.
    """
    params = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen buffers / non-learnable connectome structure
        flat = param.data.clone().flatten()  # detached, flattened copy; safe to bin and concat

        # OR→ORN
        if 'or_to_orn' in name or 'or_gains' in name:
            params.setdefault('OR→ORN', []).append(flat)

        # Antennal lobe
        elif 'antennal_lobe' in name:
            if 'log_or_gain' in name:
                # OR gain is stored in log-space (exp() at use) to keep it
                # strictly positive while remaining unconstrained for the optimiser.
                params.setdefault('OR gain', []).append(flat)
            elif 'orn_neurons' in name:
                if 'v_th' in name:
                    params.setdefault('ORN v_th', []).append(flat)  # LIF spike threshold (V)
                elif 'tau' in name or 'log_tau' in name:
                    params.setdefault('ORN τ_m', []).append(flat)   # membrane time constant (s)
            elif 'ln_neurons' in name:
                if 'v_th' in name:
                    params.setdefault('LN v_th', []).append(flat)
                elif 'tau' in name or 'log_tau' in name:
                    params.setdefault('LN τ_m', []).append(flat)
            elif 'pn_neurons' in name:
                if 'v_th' in name:
                    params.setdefault('PN v_th', []).append(flat)
                elif 'tau' in name or 'log_tau' in name:
                    params.setdefault('PN τ_m', []).append(flat)
            elif 'orn_pn' in name or 'orn_to_pn' in name:
                # Each chemical-synapse class may have a "nonad" (non-AD compartment)
                # twin; bin it separately so the two compartments don't get merged.
                key = 'ORN→PN nonad' if 'nonad' in name else 'ORN→PN'
                params.setdefault(key, []).append(flat)
            elif 'orn_ln' in name or 'orn_to_ln' in name:
                key = 'ORN→LN nonad' if 'nonad' in name else 'ORN→LN'
                params.setdefault(key, []).append(flat)
            elif 'ln_pn_excit' in name:
                # Excitatory LN->PN (from the eLN subset) checked BEFORE the generic
                # 'ln_pn' branch so it isn't swallowed by it.
                key = 'LN→PN excit nonad' if 'nonad' in name else 'LN→PN excit'
                params.setdefault(key, []).append(flat)
            elif 'ln_pn' in name or 'ln_to_pn' in name:
                key = 'LN→PN nonad' if 'nonad' in name else 'LN→PN'
                params.setdefault(key, []).append(flat)
            elif 'ln_ln' in name:
                key = 'LN→LN nonad' if 'nonad' in name else 'LN→LN'
                params.setdefault(key, []).append(flat)
            elif 'pn_ln' in name:
                key = 'PN→LN nonad' if 'nonad' in name else 'PN→LN'
                params.setdefault(key, []).append(flat)
            elif 'ln_orn' in name:
                key = 'LN→ORN nonad' if 'nonad' in name else 'LN→ORN'
                params.setdefault(key, []).append(flat)
            elif 'log_g_gap' in name:
                # Gap-junction conductances (stored in log-space, exp() at use so
                # they stay positive). The three electrical-coupling motifs are
                # disambiguated purely by substrings in the parameter name:
                if 'ln' in name and 'pn' not in name:
                    params.setdefault('Gap LN-LN', []).append(flat)   # LN<->LN coupling
                elif 'pn' in name and 'eln' not in name:
                    params.setdefault('Gap PN-PN', []).append(flat)   # sister PN<->PN coupling
                elif 'eln' in name:
                    params.setdefault('Gap eLN-PN', []).append(flat)  # excitatory-LN<->PN coupling

        # KC layer
        elif 'kc_layer' in name:
            if 'pn_kc' in name or 'pn_to_kc' in name:
                key = 'PN→KC nonad' if 'nonad' in name else 'PN→KC'
                params.setdefault(key, []).append(flat)
            elif 'kc_dend_gain' in name:
                params.setdefault('KC dend gain', []).append(flat)  # KC dendritic-compartment input gain
            elif 'kc_neurons' in name:
                if 'v_th' in name:
                    params.setdefault('KC v_th', []).append(flat)   # KC spike threshold (V)
                elif 'g_soma' in name:
                    # Soma<->dendrite coupling conductance of the two-compartment KC.
                    # Stored as log_g_soma internally; here it is binned as-stored.
                    params.setdefault('KC g_soma', []).append(flat)
            elif 'kc_apl_log_strength' in name:
                # Store in real space (exp) since log-space CV is misleading when mean≈0
                # (KC->APL drive strength is parameterised in log-space; exponentiating
                # gives the physical conductance, whose coefficient of variation is the
                # quantity the consistency analysis actually wants).
                params.setdefault('KC→APL strength', []).append(torch.exp(flat))
            elif 'kc_apl' in name:
                params.setdefault('KC→APL', []).append(flat)        # KC -> APL feedforward weights
            elif 'apl_kc' in name:
                params.setdefault('APL→KC', []).append(flat)        # APL -> KC divisive (shunting) inhibition weights
            elif 'kc_kc_aa' in name:
                params.setdefault('KC-KC aa', []).append(flat)      # KC<->KC axon-axon coupling
            elif 'kc_kc_ad' in name:
                params.setdefault('KC-KC ad', []).append(flat)      # KC<->KC axon-dendrite coupling
            elif 'apl_gain' in name:
                params.setdefault('APL gain', []).append(flat)      # global APL inhibition gain
            elif 'apl' in name and 'tau' in name:
                params.setdefault('APL τ', []).append(flat)         # APL membrane/inhibition time constant (s)

        # Decoder
        elif 'decoder' in name:
            params.setdefault('Decoder', []).append(flat)           # linear readout over 28 odor classes

    # Concatenate lists
    for key in params:
        if params[key]:
            # Each category accumulated a list of flat tensors; merge into one 1-D tensor.
            params[key] = torch.cat(params[key])

    # Create Total
    all_p = [p for p in params.values()]
    if all_p:
        # 'Total' = every learnable parameter concatenated, for an overall summary.
        params['Total'] = torch.cat(all_p)

    return params


# ---------------------------------------------------------------------------
# Cross-model parameter consistency
# ---------------------------------------------------------------------------

def compute_pairwise_correlations(models_params):
    """Compute all pairwise Pearson correlations between models for each parameter category.

    Quantifies how reproducible the learned solution is across independently
    trained seeds: for each parameter category, every pair of models is compared
    via Pearson correlation of their (identically ordered) parameter vectors. High
    correlations mean different seeds converged to similar biological parameters
    despite random initialisation -- evidence that the connectome constraints, not
    the optimiser's luck, drive the solution.

    Args:
        models_params: list of dicts from extract_all_parameters() (one per seed).

    Returns:
        dict mapping category -> list of pairwise correlation values. Includes an
        'Overall' key built from all parameters concatenated, plus one entry per
        category that is present in enough models with matching length.
    """
    n_models = len(models_params)
    all_categories = set()
    for mp in models_params:
        all_categories.update(mp.keys())  # union of categories across all models

    correlations = {}

    # Overall correlation (all parameters concatenated)
    all_params_per_model = []
    for mp in models_params:
        # Concatenate categories in a deterministic (sorted) order so the same
        # index lines up across models; skip categories a given model lacks.
        all_flat = [mp[cat].cpu().numpy() for cat in sorted(all_categories) if cat in mp]
        if all_flat:
            all_params_per_model.append(np.concatenate(all_flat))

    overall_corrs = []
    for i, j in combinations(range(n_models), 2):  # every unordered pair of models
        if len(all_params_per_model[i]) == len(all_params_per_model[j]):  # only correlate equal-length vectors
            r, _ = pearsonr(all_params_per_model[i], all_params_per_model[j])
            if not np.isnan(r):  # drop pairs where a constant vector makes r undefined
                overall_corrs.append(r)
    if overall_corrs:
        correlations['Overall'] = overall_corrs

    # Per-category correlations
    for cat in sorted(all_categories):
        cat_params = [mp[cat].cpu().numpy() for mp in models_params if cat in mp]  # this category from every model that has it
        if (len(cat_params) >= 2 and len(cat_params[0]) >= 2          # need >=2 models and >=2 params (else r is meaningless)
                and all(len(p) == len(cat_params[0]) for p in cat_params)):  # all vectors same length to be comparable
            cat_corrs = []
            for i, j in combinations(range(len(cat_params)), 2):
                r, _ = pearsonr(cat_params[i], cat_params[j])
                if not np.isnan(r):
                    cat_corrs.append(r)
            if cat_corrs:
                correlations[cat] = cat_corrs

    return correlations


def analyze_biological_parameters(model):
    """Analyze learned biological parameters (v_th, g_soma) for realism.

    Checks whether the learned LIF spike thresholds (v_th, across ORN/LN/PN/KC)
    and the KC two-compartment soma conductance (g_soma) stayed within the
    biologically plausible bounds enforced during training, and reports summary
    statistics in human-readable units (mV for voltages, nS for conductances).
    This is the "are the learned parameters physiological?" sanity check used in
    the paper's supplementary analysis.

    Args:
        model: a trained SpikingConnectomeConstrainedModel exposing
            ``antennal_lobe`` (orn/ln/pn neurons) and ``kc_layer`` (kc neurons).

    Returns:
        dict with:
            'v_th': overall and per-population threshold stats (counts, fraction
                in-bounds, min/max/mean in mV).
            'g_soma' (only if the KC soma conductance exists): its value in nS,
                whether it is in-bounds, and the bound interval in nS.
    """
    # Bounds are imported lazily (inside the function) to avoid a hard import-time
    # dependency on the layers module / its CUDA-ish setup just to use this util.
    from core.layers import V_TH_MIN, V_TH_MAX, G_SOMA_MIN, G_SOMA_MAX

    results = {}
    all_vth = []           # every threshold pooled across populations
    vth_populations = {}   # per-population threshold arrays

    for pop_name, neurons in [
        ('ORN', model.antennal_lobe.orn_neurons),
        ('LN', model.antennal_lobe.ln_neurons),
        ('PN', model.antennal_lobe.pn_neurons),
        ('KC', model.kc_layer.kc_neurons),
    ]:
        vth = neurons.v_th.detach().cpu().numpy()  # learned thresholds for this population, in volts
        all_vth.extend(vth.tolist())
        vth_populations[pop_name] = vth

    all_vth = np.array(all_vth)
    eps = 1e-9  # tiny tolerance so values sitting exactly on a clamp boundary count as in-bounds
    in_bounds = np.sum((all_vth >= V_TH_MIN - eps) & (all_vth <= V_TH_MAX + eps))  # count thresholds within [V_TH_MIN, V_TH_MAX]

    results['v_th'] = {
        'total_neurons': len(all_vth),
        'in_bounds': int(in_bounds),
        'pct_in_bounds': 100 * in_bounds / len(all_vth),
        'min_mV': float(all_vth.min() * 1000),   # V -> mV for readability
        'max_mV': float(all_vth.max() * 1000),
        'mean_mV': float(all_vth.mean() * 1000),
        'populations': {
            name: {
                'n': len(v),
                'min_mV': float(v.min() * 1000),
                'max_mV': float(v.max() * 1000),
                'mean_mV': float(v.mean() * 1000),
            }
            for name, v in vth_populations.items()  # per-population threshold summary (mV)
        },
    }

    if hasattr(model.kc_layer.kc_neurons, 'g_soma'):
        # g_soma is exposed as a property returning exp(log_g_soma); .item() pulls
        # the single scalar soma<->dendrite coupling conductance (in siemens).
        g_soma = model.kc_layer.kc_neurons.g_soma.item()
        results['g_soma'] = {
            'value_nS': float(g_soma * 1e9),                          # S -> nS for readability
            'in_bounds': G_SOMA_MIN <= g_soma <= G_SOMA_MAX,          # within the clamp used during training
            'bounds_nS': [float(G_SOMA_MIN * 1e9), float(G_SOMA_MAX * 1e9)],  # report the bound interval in nS
        }

    return results

