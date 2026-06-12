"""Centralized ROBUST recompute engine for live-from-checkpoint metric computation.

Every metric the paper cites is recomputed from a loaded model checkpoint with a FIXED
eval seed + robust trial counts, so the result is deterministic and reproducible across
notebook runs (no run-to-run wobble from evaluation noise). This is the single source of
truth used by the notebook's recompute cells and the scripts' --recompute paths.

Pipeline context
----------------
The model under evaluation is the connectome-constrained spiking network of the larval
Drosophila olfactory pathway:

    OR responses (Kreher 2008)
      -> ORN (LIF) -> LN (LIF) -> PN (LIF)               [antennal lobe, "AL"]
      -> KC (two-compartment) <- APL (graded divisive inhibition)   [mushroom body, "MB"]
      -> linear decoder over 28 odors.

Connectivity is fixed by the Winding 2023 connectome; only the ~449 biological parameters
are learned. This file does NOT define any of that physics — it loads a trained checkpoint
back into the canonical architecture (see model.py / layers.py) and drives the analysis
functions in analysis/compute.py to regenerate the numbers reported in the paper.

Why "robust": the default dataset/eval settings used during training are intentionally
small/noisy (few test repeats, few noisy trials). Here we crank repeats/trials up and pin
the RNG seed so the recomputed scalars are stable to ~3 significant figures.
"""
import torch
import numpy as np

# --- Fixed evaluation configuration ----------------------------------------
# These constants make the recompute deterministic and statistically tight. They are eval-
# time choices only; none of them affect how the checkpoint was trained.
EVAL_SEED = 42         # fixed eval seed so live recompute is deterministic/reproducible
TEST_REPEATS = 50      # 28 odors x 50 = 1400 test samples (vs the noisy 5/140 default)
N_TRIALS = 50          # noisy trials for centroid / decorrelation / Mancini
NOISE_STD = 0.3        # multiplicative input-noise CV (30%) used for the noisy-trial metrics


def _seed():
    """Re-seed both RNGs to EVAL_SEED so every metric starts from the same noise stream.

    Called immediately before each stochastic metric (the model injects biological noise
    on every forward pass, and several metrics also generate noisy input patterns), so the
    sequence of random draws is identical across notebook runs. EVAL_SEED is read from the
    module global, which recompute_metrics may have rebound to the caller's eval_seed.

    Side effects: sets torch and numpy global RNG state. Returns None.
    """
    torch.manual_seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)


def recompute_metrics(model, data_dir, or_responses, *,
                      eval_seed=EVAL_SEED, test_repeats=TEST_REPEATS,
                      n_trials=N_TRIALS, noise_std=NOISE_STD, with_mancini=True,
                      with_concentration=False, seed_for_conc=42):
    """Recompute the standard paper metrics for a loaded model, deterministically.

    Drives the analysis functions in analysis/compute.py, re-seeding before each so the
    result is reproducible despite the six biological noise sources active at eval time.

    Args:
        model: a loaded SpikingConnectomeConstrainedModel (canonical or any ablation),
            already in eval mode (this fn also calls .eval()).
        data_dir: Path to the .../data directory holding the Kreher2008 + Winding2023
            tensors (used to build the robust test loader).
        or_responses: (n_odors, n_or_types) array/tensor of OR activations per odor — the
            clean input patterns that the noisy-trial metrics perturb. n_or_types = 21
            (larval ORs), n_odors = 28.
        eval_seed: overrides EVAL_SEED for this call; rebinds the module global so _seed()
            and any downstream code reading EVAL_SEED use it.
        test_repeats: number of noisy repeats per odor for the held-out test loader
            (28 * test_repeats total samples).
        n_trials: number of noisy forward passes for the centroid / decorrelation / Mancini
            metrics (more trials -> tighter estimates).
        noise_std: multiplicative input-noise CV for the noisy-trial metrics.
        with_mancini: if True, also run the Mancini APL-inhibition test (slower).
        with_concentration: if True, also run concentration-invariance (needs the script's
            Hill-curve constants; off by default).
        seed_for_conc: seed passed into the concentration-invariance routine.

    Returns:
        dict with keys: accuracy, sparsity, centroid, al_decorr, mb_decorr, total_decorr,
        and optionally 'mancini' and 'concentration'. All scalars are cast to plain float
        for clean JSON serialization.

    Side effects: rebinds module global EVAL_SEED; re-seeds RNGs repeatedly; sets the model
    to eval mode.
    """
    # Imports are local (not module-level) so that simply importing this module is cheap and
    # free of heavy/circular dependencies on the full package; they only fire when a metric
    # is actually recomputed.
    from core.dataset import load_kreher2008_all_odors, create_dataloaders
    from analysis.compute import (
        evaluate_model, centroid_accuracy, run_mancini_test,
        compute_per_pair_decorrelation, run_concentration_invariance)

    global EVAL_SEED
    EVAL_SEED = eval_seed   # rebind so _seed() (and any reader of EVAL_SEED) uses the caller's seed
    model.eval()            # disable dropout/grad-bookkeeping; noise injection stays on (it is param-gated)

    # Robust test loader (large, seed-independent)
    # Multiplicative noise = each OR response scaled by (1 + N(0, noise_std)); train_repeats
    # is irrelevant here (we only consume the test split) but the loader builds both splits.
    tr, te, _ = load_kreher2008_all_odors(
        data_dir, train_repeats=10, test_repeats=test_repeats,
        noise_std=noise_std, noise_type='multiplicative')
    _, test_loader = create_dataloaders(tr, te, batch_size=16)

    # Re-seed before EACH stochastic metric so they are mutually reproducible and order-
    # independent (each gets the same fresh noise stream from EVAL_SEED).
    _seed(); ev = evaluate_model(model, test_loader)                 # accuracy + sparsity
    _seed(); cent = centroid_accuracy(model, or_responses, n_trials, noise_std)            # nearest-centroid decode accuracy
    _seed(); pp = compute_per_pair_decorrelation(model, or_responses, n_trials, noise_std)  # AL/MB pattern decorrelation
    out = {
        'accuracy': float(ev['accuracy']),       # fraction of test odors correctly decoded
        'sparsity': float(ev['sparsity']),       # mean KC population sparsity (fraction active)
        'centroid': float(cent),                 # nearest-centroid classification accuracy
        'al_decorr': float(pp['al_decorr']),     # % decorrelation OR->PN (antennal lobe stage)
        'mb_decorr': float(pp['mb_decorr']),     # % decorrelation PN->KC (mushroom body stage)
        'total_decorr': float(pp['total_decorr']),  # % decorrelation OR->KC (end to end)
    }
    if with_mancini:
        # Mancini APL test: ratio of KC spiking with vs without APL drive; paper expects ~2.0.
        _seed(); out['mancini'] = float(run_mancini_test(model, n_trials=n_trials)['ratio'])
    if with_concentration:
        # Concentration-invariance needs the canonical Hill dose-response constants that live
        # in the training script (EC50, Hill coefficient n, and the concentration sweep grid).
        from scripts.run_training import (
            CONCENTRATIONS, HILL_EC50, HILL_N)
        _seed()
        cr, ct = run_concentration_invariance(
            model, or_responses, seed_for_conc, CONCENTRATIONS, HILL_EC50, HILL_N,
            n_trials, noise_std)
        # Keep per-concentration accuracies plus only the JSON-safe scalar test results
        # (drop any nested arrays/objects that ct may also contain).
        out['concentration'] = {'per_concentration': cr, 'tests': {
            k: v for k, v in ct.items() if isinstance(v, (bool, float, int, str))}}
    return out


# Canonical (paper) noise parameters used to instantiate any model we load. These are the
# six biologically-motivated noise sources at the exact magnitudes reported in the paper
# Methods; the checkpoints were trained/evaluated with these. Mirrors REALISTIC_PARAMS in
# analysis/compute.py and scripts/run_training.py. Units, in SI:
#   v_noise_std=1 mV (membrane-voltage noise), i_noise_std=15 pA (input-current noise),
#   syn_noise_std=0.25 (25% CV synaptic-release stochasticity),
#   threshold_jitter_std=1 mV (per-step spike-threshold jitter),
#   orn_receptor_noise_std=0.10 (10% CV receptor/transduction noise),
#   circuit_noise_enabled=True master switch for the synaptic-release + circuit noise.
REALISTIC_PARAMS_KW = dict(v_noise_std=1.0e-3, i_noise_std=15e-12, syn_noise_std=0.25,
                           threshold_jitter_std=1.0e-3, orn_receptor_noise_std=0.10,
                           circuit_noise_enabled=True)


def load_checkpoint(data_dir, ckpt_path, n_odors=28, n_steps=30, strict=False):
    """Build a canonical-architecture model and load a saved checkpoint into it.

    Works for the canonical models AND every variant/ablation checkpoint (no_gap, no_apl,
    shuffle, ln-quantile, STD, energy, task-complexity), because each ablation bakes its
    change into params/buffers that live in the state_dict — so loading the state_dict
    reconstitutes the ablated configuration without needing a different class.

    Args:
        data_dir: Path to .../data (from_data_dir reads the Winding2023 connectome tensors
            from data_dir/'winding2023' to fix the connectivity masks).
        ckpt_path: Path to the .pt checkpoint (a state_dict of learned params + buffers).
        n_odors: number of decoder output classes (28 for the full Kreher panel).
        n_steps: simulation timesteps for BOTH the AL and KC sub-simulations. The canonical
            value is 30; the model class's own default of 20 is LEGACY and must be overridden
            here to match how the checkpoints were trained.
        strict: passed to load_state_dict. Default False so ablations whose keys differ
            slightly from the canonical model still load (missing/unexpected keys tolerated).

    Returns:
        the loaded model, set to eval mode.
    """
    # Local imports to avoid importing the heavy model package unless a checkpoint is loaded.
    from core.model import SpikingConnectomeConstrainedModel
    from core.layers import SpikingParams
    # from_data_dir constructs the network with connectivity pinned by the Winding2023
    # connectome. n_or_types=21 = larval OR channels; target_sparsity=0.10 is the legacy
    # in-class KC sparsity target (only used by the legacy in-class compute_loss, irrelevant
    # at eval); include_nonad=True enables the non-axon-dendrite collapsed connections that
    # the canonical model uses.
    m = SpikingConnectomeConstrainedModel.from_data_dir(
        data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10,
        params=SpikingParams(**REALISTIC_PARAMS_KW), include_nonad=True)
    m.n_steps_al = n_steps   # canonical 30 AL integration steps
    m.n_steps_kc = n_steps   # canonical 30 KC integration steps
    # weights_only=False because checkpoints may pickle non-tensor objects; map_location='cpu'
    # so a GPU-trained checkpoint loads on a CPU-only machine.
    m.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=False), strict=strict)
    m.eval()
    return m


def kc_upper_bound_fraction(ckpt_path):
    """Param-based (deterministic, no eval): fraction of KC thresholds pinned at the -30 mV upper bound.

    A diagnostic for threshold saturation: v_th is clamped to [-55 mV, -30 mV] during
    training, so a KC sitting at the upper bound (-30 mV, the least excitable / hardest to
    fire) indicates the learner pushed that cell as far toward silence as allowed. Reads the
    state_dict directly — no forward pass, no RNG — so it is exactly reproducible.

    Args:
        ckpt_path: Path to the checkpoint .pt file.

    Returns:
        float in [0, 1]: fraction of KC v_th values at/above -30.5 mV (a small tolerance
        below the -30 mV clamp ceiling to count thresholds that landed essentially at it).
    """
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    v = sd['kc_layer.kc_neurons.v_th'].numpy() * 1e3  # v_th stored in Volts -> convert to mV
    return float((v >= -30.5).mean())  # share within ~0.5 mV of the -30 mV upper clamp


def disable_std_params(model):
    """Disable short-term depression at the PARAMETER level (the current model has no
    disable_std forward flag). Setting U->1 and tau_rec->0 makes the STD path
    I = matmul(spikes*U*x_std, W) collapse to the no-STD path I = matmul(spikes, W),
    since x_std stays ~1 (instant recovery). Apply to a loaded model before eval.

    Background — Tsodyks-Markram STD on chemical synapses (see layers.py):
        recovery:  x <- 1 - (1 - x) * exp(-dt / tau_rec)   (vesicle pool refills toward 1)
        current:   I = matmul(spikes * U * x, W)            (release prob U, available frac x)
        depletion: x <- x * (1 - U * spikes)               (spikes consume vesicles)
    With U=1 and tau_rec->0 the per-step recovery factor exp(-dt/tau_rec)->0, so after any
    step x snaps back to ~1 (instant recovery); the synapse never stays depressed and the
    current term reduces to matmul(spikes * 1 * 1, W) = matmul(spikes, W).

    The parameters are stored in transformed (unconstrained) space so the constrained values
    are always valid: U_std = sigmoid(logit_U) in (0, 1), tau_rec = exp(log_tau_rec) > 0.
    Hence we set the raw params, not U / tau_rec directly.

    Args:
        model: a loaded model whose synapse submodules carry logit_U / log_tau_rec params.

    Returns:
        the same model, mutated in place (returned for chaining convenience).

    Side effects: overwrites STD parameters in place on every matching submodule.
    """
    with torch.no_grad():  # editing params, not differentiating
        for mod in model.modules():
            # Only chemical-synapse modules expose both STD parameters; skip everything else.
            if hasattr(mod, 'logit_U') and hasattr(mod, 'log_tau_rec'):
                if mod.logit_U is not None:
                    mod.logit_U.fill_(20.0)          # sigmoid(20) ~= 1.0  -> U = 1
                if mod.log_tau_rec is not None:
                    mod.log_tau_rec.fill_(np.log(1e-9))  # tau_rec=1e-9 s -> exp(-dt/tau)->0 -> instant recovery -> x_std ~= 1
    return model


def aggregate(per_seed_metrics):
    """Mean +/- std across a list of per-seed metric dicts (scalar keys only).

    Collapses the per-seed results (one dict per training/eval seed) into a summary suitable
    for the paper's "mean +/- std over N seeds" reporting. Only numeric (int/float) keys are
    aggregated; nested/structured entries (e.g. the 'concentration' sub-dict) are skipped.
    Keys are taken from the first dict, so all dicts are assumed to share the same scalar keys.

    Args:
        per_seed_metrics: non-empty list of metric dicts, e.g. the outputs of
            recompute_metrics for each seed.

    Returns:
        dict mapping each scalar key -> {'mean': float, 'std': float} (population std, ddof=0).
    """
    keys = [k for k in per_seed_metrics[0] if isinstance(per_seed_metrics[0][k], (int, float))]  # scalar keys from first dict
    return {k: {'mean': float(np.mean([m[k] for m in per_seed_metrics])),
                'std': float(np.std([m[k] for m in per_seed_metrics]))} for k in keys}
