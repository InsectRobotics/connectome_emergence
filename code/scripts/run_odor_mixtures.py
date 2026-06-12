"""
run_odor_mixtures.py

C2: Odor Mixture / Superposition Experiments (Reviewer 2).
Presents combinations of 2-3 simultaneous odors to trained canonical models
and analyzes whether KC population codes produce unique, linearly separable
representations distinct from individual component codes.

Post-hoc analysis only — no retraining. Uses canonical models from
results/all_connections_nonad_canonical/.

Key questions:
  1. Do mixture KC codes differ from individual component codes?
  2. Are mixture representations linearly separable from each other?
  3. Do KC codes show non-linear mixture suppression (consistent with
     sparse coding / competitive inhibition via APL)?

Reference: Honegger et al. 2011 (Neuron) — KC ensemble codes for mixtures.

Results saved to: results/odor_mixtures_r5/
  mixture_results_seed{42-46}.json
  mixture_summary.json

Notebook section: Section B — C2 (odor mixture statistics and figure).

------------------------------------------------------------------------------
PIPELINE / SCIENTIFIC CONTEXT
------------------------------------------------------------------------------
This script is a post-hoc *evaluation* driver: it loads already-trained
checkpoints of the connectome-constrained spiking model and probes their KC
(Kenyon cell, mushroom-body principal neuron) representations. It does NOT
train anything and never touches the loss/optimizer.

The forward model it queries is the full larval Drosophila olfactory pathway:
    OR responses (Kreher 2008) -> ORN (LIF) -> LN (LIF) -> PN (LIF)
      -> KC (two-compartment LIF) <- APL (graded divisive inhibition)
      -> linear decoder over 28 odors.
Connectivity is fixed by the Winding 2023 connectome; only ~449 biological
parameters were learned during training. Here we hold all of that fixed and
just read out KC spike codes for single odors vs. odor mixtures.

The "C2" naming refers to a reviewer-requested control/experiment in the
CCN 2026 paper revision (Reviewer 2). Two distinct analyses live in this one
file (merged from two former scripts):
  (A) analyze_mixtures() — geometry of mixture codes (cosine similarity to
      components / linear prediction, sparsity-based suppression ratio) plus
      LinearSVC separability tests.
  (B) run_seed()/run_honegger_all_seeds() — a per-KC sub-additivity metric
      replicating Honegger et al. 2011 (mixture response vs. SUM of component
      responses), reported as a fraction of sub-additive KCs.

Runtime constants here (N_STEPS=30, NOISE_STD=0.3, etc.) are the *canonical
evaluation* values, intentionally duplicated from the training drivers so this
post-hoc script matches the conditions the models were validated under. They
override model.py's legacy in-class defaults (e.g. n_steps_al/kc = 20).
"""
# --- Bootstrap: make the parent package importable regardless of cwd ---
import sys
from pathlib import Path

# Model code lives in code/ (core/, analysis/, scripts/). This script sits in
# code/scripts/, so code/ is two levels up; add it to sys.path for direct subpackage imports.
_pkg_parent = str(Path(__file__).parent.parent)
if _pkg_parent not in sys.path:
    sys.path.insert(0, _pkg_parent)

# Force UTF-8 line-buffered stdout/stderr so the unicode progress glyphs used
# in print() (e.g. '±', '↔', '→') render correctly and stream live to logs.
if hasattr(sys.stdout, 'reconfigure'):  # absent under Jupyter's OutStream
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):  # absent under Jupyter's OutStream
    sys.stderr.reconfigure(encoding='utf-8')

import json
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.svm import LinearSVC            # linear-kernel SVM for separability tests
from sklearn.model_selection import cross_val_score  # k-fold CV accuracy

# The connectome-constrained spiking network and its hyperparameter dataclass.
from core.model import SpikingConnectomeConstrainedModel
from core.layers import SpikingParams

# ============================================================================
# CONFIG
# ============================================================================
DATA_DIR = None  # resolved at runtime  (set in main() once the data folder is located)
# Directory holding the 5 trained canonical checkpoints (model_seed42..46.pt).
# These are the "all connections, non-adaptive, canonical" models — the headline
# configuration that includes all biological features (gap junctions, STD, APL).
CANONICAL_DIR = Path(__file__).parent.parent.parent / 'results' / 'all_connections_nonad_canonical'
# Where this script writes its JSON outputs.
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'results' / 'odor_mixtures_r5'

NOISE_STD = 0.3     # multiplicative input-noise std for repeated trials (dimensionless,
                    # applied to the [0,1] OR-response vector); drives trial-to-trial
                    # variability so averaged KC codes are stable estimates.
N_TRIALS = 20       # noisy trials per stimulus for stable KC codes
N_STEPS = 30        # timesteps for evaluation (matching canonical eval)
                    # (each step = dt = 1 ms of simulated time, so 30 ms of dynamics)
SEEDS = [42, 43, 44, 45, 46]  # all 5 canonical seeds

# How to generate mixture OR responses:
# Element-wise sum (biologically: ORN activation is approximately additive
# for co-presented odorants at moderate concentrations)
# Clamped to [0, 1] after summing.

# Noise configuration used for EVALUATION. These six biological noise sources are
# enabled so the model runs under the same stochastic conditions it was validated
# in (units are SI: volts and amps where applicable):
#   v_noise_std            = 1 mV   membrane-voltage (ion-channel) noise
#   i_noise_std            = 15 pA  background synaptic-bombardment current noise
#   syn_noise_std          = 0.25   multiplicative synaptic-release stochasticity (25% CV)
#   threshold_jitter_std   = 1 mV   spike-threshold jitter
#   orn_receptor_noise_std = 0.10   per-step multiplicative ORN receptor-binding noise
#   circuit_noise_enabled  = True   master switch turning all of the above on
# These values are larger than the SpikingParams defaults — they are the
# canonical "realistic" evaluation noise level for the paper.
REALISTIC_PARAMS = SpikingParams(
    v_noise_std=1.0e-3, i_noise_std=15e-12, syn_noise_std=0.25,
    threshold_jitter_std=1.0e-3, orn_receptor_noise_std=0.10,
    circuit_noise_enabled=True,
)


# ============================================================================
# HELPERS
# ============================================================================
def load_canonical_model(data_dir, seed, n_odors):
    """Load a trained canonical model checkpoint for a given seed.

    Builds the network skeleton from the connectome data directory (which fixes
    all connectivity masks), overrides its simulation step counts to the
    canonical evaluation value, then loads the learned parameters from disk.

    Args:
        data_dir: Path to the connectome data folder (contains kreher2008/ etc.).
            Used to reconstruct connectome-derived masks/shapes.
        seed: Integer training seed; selects CANONICAL_DIR/model_seed{seed}.pt.
        n_odors: Number of odor classes (decoder output dim); here 28.

    Returns:
        A `SpikingConnectomeConstrainedModel` in eval() mode on CPU with weights
        loaded.

    Side effects:
        Reads the checkpoint file from disk; sets model.n_steps_al/kc to N_STEPS.
    """
    # Reconstruct the architecture. n_or_types=21 OR channels, target_sparsity
    # 0.10 = ~10% KC activity target the model was trained toward, include_nonad
    # keeps the non-adaptive (canonical) configuration. REALISTIC_PARAMS injects
    # the evaluation noise level above.
    model = SpikingConnectomeConstrainedModel.from_data_dir(
        data_dir, n_odors=n_odors, n_or_types=21, target_sparsity=0.10,
        params=REALISTIC_PARAMS, include_nonad=True)
    # Override the legacy in-class default of 20 steps with the canonical 30 for
    # BOTH the antennal-lobe (AL) and KC sub-simulations. In unified_simulation
    # the effective horizon is max(n_steps_al, n_steps_kc) = 30 ms.
    model.n_steps_al = N_STEPS
    model.n_steps_kc = N_STEPS
    model_path = CANONICAL_DIR / f'model_seed{seed}.pt'
    # weights_only=False because the checkpoint is a plain state_dict pickle;
    # map_location='cpu' forces CPU load regardless of where it was trained.
    state = torch.load(model_path, weights_only=False, map_location='cpu')
    model.load_state_dict(state)
    model.eval()  # disable any train-only behavior; noise is still applied via params
    return model


def generate_mixture_or(or_responses, odor_indices):
    """Generate mixture OR response by summing component OR responses.

    C2: Element-wise sum of component OR patterns, clamped to [0,1].
    This models the approximate additivity of ORN responses to co-presented
    odorants at moderate concentrations (Kreher et al. 2008).

    Args:
        or_responses: (n_odors, n_or_types) tensor of normalized OR responses in
            [0,1]; row i is the receptor-channel activation pattern for odor i.
        odor_indices: list of odor row-indices to combine into a mixture.

    Returns:
        (n_or_types,) tensor: the per-receptor mixture pattern, summed across the
        selected components and clamped to the valid [0,1] range (clamping caps
        receptor saturation when several odors drive the same channel).
    """
    mixture = torch.zeros_like(or_responses[0])  # (n_or_types,) zero vector
    for idx in odor_indices:
        mixture = mixture + or_responses[idx]    # additive superposition of components
    mixture = mixture.clamp(0, 1)                # saturate to valid receptor range
    return mixture


def get_kc_code(model, or_input, n_trials, noise_std):
    """Get KC population code (spike rate vector) for a single stimulus.

    Returns mean KC spike rates across noisy trials, giving a stable
    population code for the stimulus.

    Args:
        model: a loaded SpikingConnectomeConstrainedModel (eval mode).
        or_input: (n_or_types,) OR-response vector for one stimulus (odor or mix).
        n_trials: number of independent noisy repeats to average over.
        noise_std: std of the multiplicative input noise.

    Returns:
        (n_kc,) tensor of mean KC firing rates (spikes per timestep) averaged
        across the noisy trials. Values lie in [0,1] since they are spike counts
        divided by N_STEPS.
    """
    n_or = or_input.shape[0]
    # Replicate the single stimulus into a batch of n_trials identical rows...
    noisy = or_input.unsqueeze(0).expand(n_trials, -1)            # (n_trials, n_or)
    # ...then perturb each trial with i.i.d. Gaussian *multiplicative* noise so
    # the relative perturbation scales with input magnitude (silent channels stay
    # silent). clamp(min=0) keeps OR responses non-negative.
    noise = torch.randn_like(noisy) * noise_std                  # (n_trials, n_or)
    noisy = (noisy * (1.0 + noise)).clamp(min=0)

    model.eval()
    with torch.no_grad():  # pure inference; no autograd graph
        # return_all=True yields the intermediates dict; 'kc_spikes' is the
        # per-KC spike COUNT over the unified simulation, shape (n_trials, n_kc).
        _, info = model(noisy, return_all=True)
        # Convert spike counts to firing RATES (spikes/step) by dividing by the
        # canonical horizon N_STEPS, matching how the decoder consumed rates.
        kc_rates = info['kc_spikes'].float() / N_STEPS
    return kc_rates.mean(dim=0)  # mean across trials → (n_kc,)


def cosine_sim(a, b):
    """Cosine similarity between two vectors.

    Used to quantify how similar a mixture KC code is to a component code or to
    a linear prediction. Returns a scalar in [-1, 1] (here typically [0, 1] since
    rate vectors are non-negative).

    Args:
        a, b: tensors of any shape; flattened to 1-D before comparison.

    Returns:
        float cosine similarity, or 0.0 if either vector has zero norm
        (avoids a divide-by-zero on a silent / all-zero code).
    """
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0  # define similarity to a zero vector as 0 (undefined otherwise)
    return float(F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)))


# ============================================================================
# MIXTURE ANALYSIS
# ============================================================================
def analyze_mixtures(model, or_responses, odor_names, seed):
    """Run full mixture analysis on a single model.

    C2 analysis:
    1. Get KC codes for all 28 individual odors
    2. Generate all 2-odor and sampled 3-odor mixtures
    3. Compare mixture codes to component codes
    4. Test linear separability of mixture vs individual codes
    5. Measure mixture suppression (sub-additivity from APL)

    Args:
        model: loaded canonical model (eval mode).
        or_responses: (n_odors, n_or_types) tensor of OR responses; row 0 is the
            "sfr" spontaneous-firing-rate / no-odor baseline.
        odor_names: list of n_odors odor name strings (df index).
        seed: integer seed used both to label results and to seed the RNGs that
            sample triplets / SVM mixtures (reproducibility).

    Returns:
        (summary, pair_results, triplet_results) where
          summary: dict of aggregate statistics for this seed,
          pair_results: list of per-2-odor-pair dicts,
          triplet_results: list of per-3-odor-triplet dicts.
    """
    n_odors = or_responses.shape[0]
    print(f"\n  Getting KC codes for {n_odors} individual odors...")

    # 1. Individual odor KC codes
    # Map each odor index -> its (n_kc,) mean rate code; reused as the reference
    # against which mixtures are compared.
    individual_codes = {}
    for i in range(n_odors):
        code = get_kc_code(model, or_responses[i], N_TRIALS, NOISE_STD)
        individual_codes[i] = code

    # 2. Generate 2-odor mixtures (all C(27,2) = 351 pairs; excludes sfr = index 0, the no-odor baseline)
    pairs = list(itertools.combinations(range(1, n_odors), 2))  # start at 1 to skip sfr
    print(f"  Generating {len(pairs)} 2-odor mixture codes...")

    pair_results = []
    for i, j in pairs:
        mix_or = generate_mixture_or(or_responses, [i, j])           # summed OR pattern
        mix_code = get_kc_code(model, mix_or, N_TRIALS, NOISE_STD)    # KC code of the mixture

        # Similarity of mixture to each component
        sim_to_i = cosine_sim(mix_code, individual_codes[i])
        sim_to_j = cosine_sim(mix_code, individual_codes[j])

        # Linear prediction: average of component codes
        # If the KC code were a purely linear function of inputs, the mixture
        # code would match this mean; deviation indicates non-linear processing.
        linear_pred = (individual_codes[i] + individual_codes[j]) / 2.0
        sim_to_linear = cosine_sim(mix_code, linear_pred)

        # Sparsity of mixture vs components
        # Fraction of KCs that fire at all (>0 rate): the hallmark sparse code.
        mix_sparsity = (mix_code > 0).float().mean().item()
        comp_i_sp = (individual_codes[i] > 0).float().mean().item()
        comp_j_sp = (individual_codes[j] > 0).float().mean().item()

        # Suppression ratio: mixture sparsity vs mean component sparsity
        # <1 means mixture is sparser (sub-additive, APL suppression)
        # APL's divisive (shunting) feedback inhibition normalizes total KC
        # activity, so combining odors recruits fewer-than-additive active KCs.
        mean_comp_sp = (comp_i_sp + comp_j_sp) / 2.0
        suppression = mix_sparsity / mean_comp_sp if mean_comp_sp > 0 else 1.0

        pair_results.append({
            'odors': [int(i), int(j)],
            'odor_names': [odor_names[i], odor_names[j]],
            'sim_to_comp1': sim_to_i,
            'sim_to_comp2': sim_to_j,
            'sim_to_linear_pred': sim_to_linear,
            'mix_sparsity': mix_sparsity,
            'comp_sparsities': [comp_i_sp, comp_j_sp],
            'suppression_ratio': suppression,
        })

    # 3. Sample 3-odor mixtures (50 random triplets for speed)
    # Full C(27,3) is large; subsample 50 for tractable runtime. Seed-offset RNG
    # keeps the choice reproducible yet distinct from other RNG uses below.
    rng = np.random.RandomState(seed + 3000)
    n_triplets = min(50, len(list(itertools.combinations(range(1, n_odors), 3))))
    all_triplets = list(itertools.combinations(range(1, n_odors), 3))
    triplet_indices = rng.choice(len(all_triplets), n_triplets, replace=False)
    triplets = [all_triplets[k] for k in triplet_indices]
    print(f"  Generating {len(triplets)} 3-odor mixture codes...")

    triplet_results = []
    for i, j, k in triplets:
        mix_or = generate_mixture_or(or_responses, [i, j, k])
        mix_code = get_kc_code(model, mix_or, N_TRIALS, NOISE_STD)

        # Linear prediction here is the mean of the three component codes.
        linear_pred = (individual_codes[i] + individual_codes[j] + individual_codes[k]) / 3.0
        sim_to_linear = cosine_sim(mix_code, linear_pred)

        mix_sparsity = (mix_code > 0).float().mean().item()
        mean_comp_sp = np.mean([(individual_codes[x] > 0).float().mean().item()
                                for x in [i, j, k]])
        suppression = mix_sparsity / mean_comp_sp if mean_comp_sp > 0 else 1.0

        triplet_results.append({
            'odors': [int(i), int(j), int(k)],
            'sim_to_linear_pred': sim_to_linear,
            'mix_sparsity': mix_sparsity,
            'suppression_ratio': suppression,
        })

    # 4. Linear separability: can an SVM distinguish mixture from individual KC codes?
    print("  Testing linear separability...")

    # Collect multiple noisy trials for SVM
    # Build a labeled dataset of individual-odor KC codes (one feature row per
    # noisy trial) so a LinearSVC can test decodability of odor identity and,
    # later, individual-vs-mixture status.
    n_svm_trials = 10
    X_individual = []
    y_individual = []
    for i in range(1, n_odors):  # exclude sfr (no-odor baseline) for consistency with the pair/triplet sets
        noisy = or_responses[i].unsqueeze(0).expand(n_svm_trials, -1)
        noise = torch.randn_like(noisy) * NOISE_STD
        noisy = (noisy * (1.0 + noise)).clamp(min=0)
        with torch.no_grad():
            _, info = model(noisy, return_all=True)
            kc_rates = info['kc_spikes'].float() / N_STEPS  # (n_svm_trials, n_kc)
        X_individual.append(kc_rates.numpy())
        y_individual.extend([i] * n_svm_trials)             # label = odor index

    X_individual = np.vstack(X_individual)  # (27*n_svm_trials, n_kc)
    y_individual = np.array(y_individual)

    # Individual odor classification accuracy (baseline)
    # Multiclass: how well KC codes separate the 27 individual odors. dual='auto'
    # lets sklearn pick the dual/primal solver; max_iter raised for convergence.
    try:
        svm_indiv = LinearSVC(max_iter=5000, dual='auto')
        indiv_scores = cross_val_score(svm_indiv, X_individual, y_individual, cv=5)
        indiv_svm_acc = float(np.mean(indiv_scores))
    except Exception:
        indiv_svm_acc = float('nan')  # degenerate data (e.g. a class with too few samples)

    # Mixture vs individual: binary classification
    # Sample 50 random 2-odor mixtures for SVM
    # Independent seed offset so the SVM mixture sample differs from the geometry
    # triplet sample above.
    rng2 = np.random.RandomState(seed + 4000)
    mix_sample = rng2.choice(len(pairs), min(50, len(pairs)), replace=False)
    X_mixture = []
    for idx in mix_sample:
        i, j = pairs[idx]
        mix_or = generate_mixture_or(or_responses, [i, j])
        noisy = mix_or.unsqueeze(0).expand(n_svm_trials, -1)
        noise = torch.randn_like(noisy) * NOISE_STD
        noisy = (noisy * (1.0 + noise)).clamp(min=0)
        with torch.no_grad():
            _, info = model(noisy, return_all=True)
            kc_rates = info['kc_spikes'].float() / N_STEPS
        X_mixture.append(kc_rates.numpy())

    X_mixture = np.vstack(X_mixture)  # (50*n_svm_trials, n_kc)

    # Binary: individual (0) vs mixture (1)
    # Tests whether mixture codes occupy a distinguishable region of KC space
    # from individual-odor codes (a non-trivial-geometry signature).
    X_binary = np.vstack([X_individual, X_mixture])
    y_binary = np.array([0] * len(X_individual) + [1] * len(X_mixture))
    try:
        svm_binary = LinearSVC(max_iter=5000, dual='auto')
        binary_scores = cross_val_score(svm_binary, X_binary, y_binary, cv=5)
        binary_svm_acc = float(np.mean(binary_scores))
    except Exception:
        binary_svm_acc = float('nan')

    # Inter-mixture discrimination: can we tell mixtures apart?
    # Multiclass over 20 distinct mixtures: high accuracy => mixtures get unique,
    # linearly separable KC codes (the key claim addressed for Reviewer 2).
    X_mix_disc = []
    y_mix_disc = []
    for mix_idx, pair_idx in enumerate(mix_sample[:20]):  # 20 mixtures
        i, j = pairs[pair_idx]
        mix_or = generate_mixture_or(or_responses, [i, j])
        noisy = mix_or.unsqueeze(0).expand(n_svm_trials, -1)
        noise = torch.randn_like(noisy) * NOISE_STD
        noisy = (noisy * (1.0 + noise)).clamp(min=0)
        with torch.no_grad():
            _, info = model(noisy, return_all=True)
            kc_rates = info['kc_spikes'].float() / N_STEPS
        X_mix_disc.append(kc_rates.numpy())
        y_mix_disc.extend([mix_idx] * n_svm_trials)  # label = which mixture

    X_mix_disc = np.vstack(X_mix_disc)
    y_mix_disc = np.array(y_mix_disc)
    try:
        svm_mix = LinearSVC(max_iter=5000, dual='auto')
        mix_scores = cross_val_score(svm_mix, X_mix_disc, y_mix_disc, cv=5)
        mix_svm_acc = float(np.mean(mix_scores))
    except Exception:
        mix_svm_acc = float('nan')

    # 5. Summary statistics
    # Average the two per-component similarities into a single mix↔component number.
    pair_sims_to_comp = [(r['sim_to_comp1'] + r['sim_to_comp2']) / 2 for r in pair_results]
    pair_sims_to_linear = [r['sim_to_linear_pred'] for r in pair_results]
    pair_suppressions = [r['suppression_ratio'] for r in pair_results]

    triplet_sims = [r['sim_to_linear_pred'] for r in triplet_results]
    triplet_suppressions = [r['suppression_ratio'] for r in triplet_results]

    # Collect all scalar metrics for this seed into a JSON-serializable dict.
    summary = {
        'seed': seed,
        'n_individual_odors': n_odors,
        'n_2odor_mixtures': len(pairs),
        'n_3odor_mixtures': len(triplets),

        # Similarity: mixture code vs components
        'pair_sim_to_components_mean': float(np.mean(pair_sims_to_comp)),
        'pair_sim_to_components_std': float(np.std(pair_sims_to_comp)),
        'pair_sim_to_linear_pred_mean': float(np.mean(pair_sims_to_linear)),
        'pair_sim_to_linear_pred_std': float(np.std(pair_sims_to_linear)),

        # Suppression: mixture sparsity / component sparsity
        'pair_suppression_mean': float(np.mean(pair_suppressions)),
        'pair_suppression_std': float(np.std(pair_suppressions)),
        'triplet_suppression_mean': float(np.mean(triplet_suppressions)),
        'triplet_suppression_std': float(np.std(triplet_suppressions)),

        # Similarity: 3-odor mixtures
        'triplet_sim_to_linear_mean': float(np.mean(triplet_sims)),
        'triplet_sim_to_linear_std': float(np.std(triplet_sims)),

        # SVM classification
        'individual_svm_accuracy': indiv_svm_acc,
        'mixture_vs_individual_svm': binary_svm_acc,
        'inter_mixture_svm_accuracy': mix_svm_acc,
    }

    return summary, pair_results, triplet_results


# ============================================================================
# MAIN
# ============================================================================
# ============================================================================
# HONEGGER PER-KC SUB-ADDITIVITY  (merged from former run_honegger_metric.py)
# ============================================================================
# Replicates Honegger et al. (2011, J Neurosci): for each KC, compare its mixture
# response to the SUM of its individual component responses. A KC is "sub-additive"
# if mixture_response < component1 + component2. Honegger reported 73% sub-additive
# for binary blends. Uses the same canonical checkpoints / noise / n_steps as the
# mixture analysis above. Results -> results/odor_mixtures_r5/honegger_metric.json.
N_TRIPLETS = 50  # random 3-odor triplets per seed (Honegger analysis)


def run_seed(seed):
    """Honegger-style per-KC sub-additivity analysis for one seed (2- and 3-odor mixtures).

    For every mixture, predict each KC's response as the SUM of its responses to
    the individual components (the linear "additive" expectation). A KC counts as
    sub-additive if its actual mixture response falls below that sum — the
    signature of competitive normalization (APL inhibition + KC nonlinearity).

    Args:
        seed: integer seed; selects the checkpoint and seeds the noise/triplet RNGs.

    Returns:
        dict with the fraction (and raw counts) of sub-additive responsive KCs for
        both 2-odor pairs and sampled 3-odor triplets, plus dataset sizes.

    Side effects:
        Loads the seed's checkpoint from disk and reads the Kreher OR-response CSV.
        Designed to run inside a multiprocessing worker (see run_honegger_all_seeds).
    """
    _data = Path(__file__).parent.parent.parent / 'data'
    print(f"[Seed {seed}] Loading model...")
    # Load normalized OR responses: rows = odors (incl. sfr baseline at row 0),
    # columns = 21 OR types; values in [0,1].
    df = pd.read_csv(_data / 'kreher2008' / 'orn_responses_normalized.csv', index_col=0)
    or_responses = torch.from_numpy(df.values).float()  # (n_odors, 21)
    odor_names = list(df.index)
    # Rebuild + load the canonical model for this seed (same recipe as
    # load_canonical_model, inlined here so this analysis is self-contained).
    mdl = SpikingConnectomeConstrainedModel.from_data_dir(
        _data, n_odors=len(odor_names), n_or_types=21,
        target_sparsity=0.10, params=REALISTIC_PARAMS, include_nonad=True)
    mdl.n_steps_al = N_STEPS
    mdl.n_steps_kc = N_STEPS
    state = torch.load(CANONICAL_DIR / f'model_seed{seed}.pt', map_location='cpu', weights_only=False)
    mdl.load_state_dict(state)
    mdl.eval()

    n_odors = len(or_responses) - 1  # skip sfr  (number of real odors = 27)
    odor_or = or_responses[1:]       # (27, 21)   drop the sfr baseline row
    rng = np.random.default_rng(seed)  # NumPy Generator for per-trial input noise

    def get_kc_rates(or_pattern):
        """Mean KC firing-rate vector (144-dim) over N_TRIALS noisy trials.

        Note: this Honegger variant returns mean KC SPIKE COUNTS (info['kc_spikes']
        averaged over trials), not counts/N_STEPS. That is fine here because the
        sub-additivity test only compares responses to each other on a common
        scale, and the >0.01 "responsive" thresholds are calibrated to counts.

        Args:
            or_pattern: (21,) OR-response vector for one stimulus.

        Returns:
            (n_kc,) numpy array of mean per-KC response averaged across trials.
        """
        kc_all = []
        for _ in range(N_TRIALS):
            # Multiplicative Gaussian input noise, one draw per trial.
            noise = torch.from_numpy(
                rng.normal(0, NOISE_STD, or_pattern.shape).astype(np.float32))
            x = (or_pattern * (1.0 + noise)).clamp(0)  # keep non-negative
            with torch.no_grad():
                _, info = mdl(x.unsqueeze(0), return_all=True)  # batch dim of 1
            kc_all.append(info['kc_spikes'].float().squeeze().numpy())
        return np.mean(kc_all, axis=0)  # average out trial noise → stable code

    print(f"[Seed {seed}] Running {n_odors} individual odors...")
    # Cache each odor's KC response so component sums can be assembled cheaply.
    individual_kc = {i: get_kc_rates(odor_or[i]) for i in range(n_odors)}

    # === 2-ODOR PAIRS ===
    pairs = list(itertools.combinations(range(n_odors), 2))  # all C(27,2)=351 pairs
    print(f"[Seed {seed}] Running {len(pairs)} 2-odor pairs...")
    pair_sub, pair_resp = [], []  # per-pair counts of sub-additive KCs / responsive KCs
    for idx, (i, j) in enumerate(pairs):
        mix_or = (odor_or[i] + odor_or[j]).clamp(0, 1)  # additive OR mixture, saturated
        mix_kc = get_kc_rates(mix_or)
        linear_pred = individual_kc[i] + individual_kc[j]  # additive expectation (SUM, not mean)
        # A KC is "responsive" if EITHER the additive prediction or the actual
        # mixture exceeds a small floor (0.01), excluding silent KCs from the stat.
        responsive = (linear_pred > 0.01) | (mix_kc > 0.01)
        n_resp = responsive.sum()
        if n_resp > 0:
            # Among responsive KCs, count how many fire LESS than the additive sum.
            pair_sub.append((mix_kc[responsive] < linear_pred[responsive]).sum())
            pair_resp.append(n_resp)
        if (idx + 1) % 100 == 0:
            print(f"[Seed {seed}] pairs: {idx+1}/{len(pairs)}")
    # Pooled fraction of sub-additive responsive KCs across all pairs.
    pair_frac = np.sum(pair_sub) / np.sum(pair_resp)

    # === 3-ODOR TRIPLETS ===
    all_triplets = list(itertools.combinations(range(n_odors), 3))
    rng_trip = np.random.default_rng(seed + 1000)  # separate RNG for triplet sampling
    # Randomly subsample N_TRIPLETS triplets (capped at the total available).
    triplet_indices = rng_trip.choice(len(all_triplets), size=min(N_TRIPLETS, len(all_triplets)), replace=False)
    triplets = [all_triplets[idx] for idx in triplet_indices]
    print(f"[Seed {seed}] Running {len(triplets)} 3-odor triplets...")
    trip_sub, trip_resp = [], []
    for (i, j, k) in triplets:
        mix_or = (odor_or[i] + odor_or[j] + odor_or[k]).clamp(0, 1)
        mix_kc = get_kc_rates(mix_or)
        linear_pred = individual_kc[i] + individual_kc[j] + individual_kc[k]  # 3-way additive sum
        responsive = (linear_pred > 0.01) | (mix_kc > 0.01)
        n_resp = responsive.sum()
        if n_resp > 0:
            trip_sub.append((mix_kc[responsive] < linear_pred[responsive]).sum())
            trip_resp.append(n_resp)
    trip_frac = np.sum(trip_sub) / np.sum(trip_resp)

    result = {
        'seed': seed,
        'pair_frac_sub_additive': float(pair_frac),
        'pair_total_sub': int(np.sum(pair_sub)),
        'pair_total_resp': int(np.sum(pair_resp)),
        'triplet_frac_sub_additive': float(trip_frac),
        'triplet_total_sub': int(np.sum(trip_sub)),
        'triplet_total_resp': int(np.sum(trip_resp)),
        'n_pairs': len(pairs),
        'n_triplets': len(triplets),
    }
    # Print sub-additive fractions as percentages for quick comparison to the 73%
    # Honegger benchmark.
    print(f"[Seed {seed}] DONE: pairs={pair_frac:.1%}, triplets={trip_frac:.1%}")
    return result


def run_honegger_all_seeds():
    """Run Honegger sub-additivity across all 5 seeds (parallel) and save the aggregate.

    Spawns a 5-worker process pool (one seed per worker) to run run_seed in
    parallel, aggregates the per-seed sub-additive fractions into mean ± std for
    2- and 3-odor mixtures, prints a comparison to the Honegger (2011) 73%
    benchmark, and writes everything to honegger_metric.json.

    Side effects:
        Creates OUTPUT_DIR if needed; writes honegger_metric.json.
    """
    import multiprocessing as mp
    with mp.Pool(5) as pool:
        results = pool.map(run_seed, SEEDS)  # blocks until all 5 seeds finish
    pair_fracs = [r['pair_frac_sub_additive'] for r in results]
    trip_fracs = [r['triplet_frac_sub_additive'] for r in results]
    print("\n" + "=" * 60)
    print("HONEGGER AGGREGATE (5 seeds)")
    print("=" * 60)
    print(f"2-odor: {np.mean(pair_fracs):.1%} ± {np.std(pair_fracs):.1%} sub-additive KCs")
    print(f"3-odor: {np.mean(trip_fracs):.1%} ± {np.std(trip_fracs):.1%} sub-additive KCs")
    print(f"Honegger (2011): 73% (2-odor only)")  # literature reference point
    output = {
        'per_seed': results,
        'aggregate': {
            'pair_frac_mean': float(np.mean(pair_fracs)),
            'pair_frac_std': float(np.std(pair_fracs)),
            'triplet_frac_mean': float(np.mean(trip_fracs)),
            'triplet_frac_std': float(np.std(trip_fracs)),
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'honegger_metric.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")



def honegger_for_model(mdl, or_responses, seed, quiet=True):
    """Honegger per-KC sub-additivity for an ALREADY-LOADED model (2- and 3-odor mixtures).

    Model-based twin of run_seed (which loads the canonical checkpoint), so the same
    sub-additivity metric can run on ABLATION checkpoints. ``or_responses`` is
    (n_odors, 21) with the sfr baseline at row 0.
    """
    mdl.eval()
    n_odors = len(or_responses) - 1
    odor_or = or_responses[1:]
    rng = np.random.default_rng(seed)
    def get_kc_rates(or_pattern):
        kc_all = []
        for _ in range(N_TRIALS):
            noise = torch.from_numpy(rng.normal(0, NOISE_STD, or_pattern.shape).astype(np.float32))
            x = (or_pattern * (1.0 + noise)).clamp(0)
            with torch.no_grad():
                _, info = mdl(x.unsqueeze(0), return_all=True)
            kc_all.append(info['kc_spikes'].float().squeeze().numpy())
        return np.mean(kc_all, axis=0)
    individual_kc = {i: get_kc_rates(odor_or[i]) for i in range(n_odors)}
    pairs = list(itertools.combinations(range(n_odors), 2))
    pair_sub, pair_resp = [], []
    for (i, j) in pairs:
        mix_kc = get_kc_rates((odor_or[i] + odor_or[j]).clamp(0, 1))
        linear_pred = individual_kc[i] + individual_kc[j]
        responsive = (linear_pred > 0.01) | (mix_kc > 0.01)
        nr = responsive.sum()
        if nr > 0:
            pair_sub.append((mix_kc[responsive] < linear_pred[responsive]).sum()); pair_resp.append(nr)
    pair_frac = np.sum(pair_sub) / np.sum(pair_resp)
    all_triplets = list(itertools.combinations(range(n_odors), 3))
    rng_trip = np.random.default_rng(seed + 1000)
    ti = rng_trip.choice(len(all_triplets), size=min(N_TRIPLETS, len(all_triplets)), replace=False)
    triplets = [all_triplets[k] for k in ti]
    trip_sub, trip_resp = [], []
    for (i, j, k) in triplets:
        mix_kc = get_kc_rates((odor_or[i] + odor_or[j] + odor_or[k]).clamp(0, 1))
        linear_pred = individual_kc[i] + individual_kc[j] + individual_kc[k]
        responsive = (linear_pred > 0.01) | (mix_kc > 0.01)
        nr = responsive.sum()
        if nr > 0:
            trip_sub.append((mix_kc[responsive] < linear_pred[responsive]).sum()); trip_resp.append(nr)
    trip_frac = np.sum(trip_sub) / np.sum(trip_resp)
    if not quiet:
        print(f"[Seed {seed}] DONE: pairs={pair_frac:.1%}, triplets={trip_frac:.1%}")
    return {'seed': seed, 'pair_frac_sub_additive': float(pair_frac),
            'triplet_frac_sub_additive': float(trip_frac)}


F4_EXTRA_DIR = Path(__file__).resolve().parent.parent.parent / 'results' / 'f4_extra'


def f4extra_for_ckpt(label, seed, ckpt_path):
    """One ablation checkpoint -> APL-mediated KC suppression (%) + Honegger 2-odor
    sub-additivity (%). Writes results/f4_extra/{label}_s{seed}.json. Dispatched one
    subprocess per (label, seed) by the Figure-4 'F4-extra' notebook cell."""
    from analysis.compute import run_mancini_test
    _data = Path(__file__).resolve().parent.parent.parent / 'data'
    df = pd.read_csv(_data / 'kreher2008' / 'orn_responses_normalized.csv', index_col=0)
    or_responses = torch.from_numpy(df.values).float()
    mdl = SpikingConnectomeConstrainedModel.from_data_dir(
        _data, n_odors=len(df.index), n_or_types=21, target_sparsity=0.10,
        params=REALISTIC_PARAMS, include_nonad=True)
    mdl.n_steps_al = N_STEPS; mdl.n_steps_kc = N_STEPS
    mdl.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=False), strict=False)
    mdl.eval()
    h = honegger_for_model(mdl, or_responses, seed, quiet=True)
    ratio = float(run_mancini_test(mdl, n_trials=20)['ratio'])
    suppression = max(0.0, (1.0 - 1.0 / ratio) * 100.0)   # % KC suppression by APL activation
    out = {'label': label, 'seed': seed, 'mancini_ratio': ratio,
           'suppression_pct': suppression, 'subadd_pct': h['pair_frac_sub_additive'] * 100.0}
    F4_EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    with open(F4_EXTRA_DIR / f'{label}_s{seed}.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[F4x {label} s{seed}] suppression={suppression:.1f}%  sub-add={out['subadd_pct']:.1f}%")
    return out


def main():
    """CLI entry point: dispatch to either the Honegger metric or the mixture analysis.

    Flags:
        --honegger    Run the per-KC sub-additivity analysis (run_honegger_all_seeds)
                      and exit, instead of the geometry/SVM mixture analysis.
        --seed N      Run only a single seed (otherwise all 5).
        --n-seeds K   Run the first K canonical seeds.

    Default path: locate the data directory, load OR responses, then for each
    seed run analyze_mixtures and write per-seed JSON plus an aggregated
    mixture_summary.json.
    """
    import argparse
    parser = argparse.ArgumentParser(description='C2: Odor mixture analysis')
    parser.add_argument('--seed', type=int, default=None,
                        help='Single seed (default: run all 5)')
    parser.add_argument('--n-seeds', type=int, default=None,
                        help='Number of seeds to use (default: all 5)')
    parser.add_argument('--honegger', action='store_true',
                        help='Run the Honegger per-KC sub-additivity analysis instead of the mixture geometry/SVM analysis')
    parser.add_argument('--f4extra', action='store_true',
                        help='Figure-4 extra: APL suppression + sub-additivity for ONE ablation checkpoint (needs --label/--seed/--ckpt)')
    parser.add_argument('--label', type=str, default=None, help='condition label for --f4extra output filename')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint path for --f4extra')
    args = parser.parse_args()

    # Branch F4-extra: one ablation checkpoint -> suppression + sub-additivity JSON.
    if args.f4extra:
        f4extra_for_ckpt(args.label, args.seed, args.ckpt)
        return

    # Branch A: Honegger sub-additivity metric (self-contained; returns early).
    if args.honegger:
        run_honegger_all_seeds()
        return

    # Resolve data directory
    # Locate the dataset directory.
    _parent = Path(__file__).resolve().parent.parent.parent
    _data_candidates = [
        Path(__file__).resolve().parent.parent.parent / 'data',
    ]
    global DATA_DIR
    DATA_DIR = next((p for p in _data_candidates if (p / 'kreher2008').is_dir()), None)
    if DATA_DIR is None:
        raise FileNotFoundError('Cannot find connectome data.')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load OR responses and odor names
    kreher_dir = DATA_DIR / 'kreher2008'
    df = pd.read_csv(kreher_dir / 'orn_responses_normalized.csv', index_col=0)
    or_responses = torch.from_numpy(df.values).float()  # (n_odors, 21) in [0,1]
    odor_names = df.index.tolist()
    n_odors = len(odor_names)
    print(f"Loaded {n_odors} odors x {or_responses.shape[1]} OR types")

    # Determine seeds
    # Priority: explicit --seed, else first --n-seeds of SEEDS, else all 5.
    if args.seed is not None:
        seeds = [args.seed]
    elif args.n_seeds is not None:
        seeds = SEEDS[:args.n_seeds]
    else:
        seeds = SEEDS

    all_summaries = []  # collect each seed's summary dict for the aggregate report

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}: MIXTURE ANALYSIS")
        print(f"{'='*70}")

        model = load_canonical_model(DATA_DIR, seed, n_odors)
        summary, pairs, triplets = analyze_mixtures(model, or_responses, odor_names, seed)
        del model  # free the model before loading the next seed (memory hygiene)

        all_summaries.append(summary)

        # Save per-seed results
        # Persist the full per-pair / per-triplet detail alongside the summary so
        # downstream figures can re-derive distributions without recomputation.
        per_seed_path = OUTPUT_DIR / f'mixture_results_seed{seed}.json'
        with open(per_seed_path, 'w') as f:
            json.dump({
                'summary': summary,
                'pair_results': pairs,
                'triplet_results': triplets,
            }, f, indent=2)
        print(f"  Saved: {per_seed_path}")

        # Human-readable console summary for this seed.
        print(f"\n  --- Seed {seed} Summary ---")
        print(f"  2-odor mix sim to components:   {summary['pair_sim_to_components_mean']:.3f} ± {summary['pair_sim_to_components_std']:.3f}")
        print(f"  2-odor mix sim to linear pred:  {summary['pair_sim_to_linear_pred_mean']:.3f} ± {summary['pair_sim_to_linear_pred_std']:.3f}")
        print(f"  2-odor suppression ratio:       {summary['pair_suppression_mean']:.3f} ± {summary['pair_suppression_std']:.3f}")
        print(f"  3-odor suppression ratio:       {summary['triplet_suppression_mean']:.3f} ± {summary['triplet_suppression_std']:.3f}")
        print(f"  Individual SVM accuracy:        {summary['individual_svm_accuracy']:.1%}")
        print(f"  Mixture vs individual SVM:      {summary['mixture_vs_individual_svm']:.1%}")
        print(f"  Inter-mixture SVM:              {summary['inter_mixture_svm_accuracy']:.1%}")

    # ---- AGGREGATE SUMMARY ----
    # Only meaningful with >1 seed; print mean ± std across seeds for each metric.
    if len(all_summaries) > 1:
        print(f"\n{'='*80}")
        print("C2 ODOR MIXTURE AGGREGATE SUMMARY")
        print(f"{'='*80}")

        # (summary-key, human-readable label) pairs to report across seeds.
        metrics = [
            ('pair_sim_to_components_mean', 'Mix↔Component similarity'),
            ('pair_sim_to_linear_pred_mean', 'Mix↔Linear prediction sim'),
            ('pair_suppression_mean', '2-odor suppression ratio'),
            ('triplet_suppression_mean', '3-odor suppression ratio'),
            ('individual_svm_accuracy', 'Individual SVM accuracy'),
            ('mixture_vs_individual_svm', 'Mix vs Individual SVM'),
            ('inter_mixture_svm_accuracy', 'Inter-mixture SVM'),
        ]

        for key, label in metrics:
            vals = [s[key] for s in all_summaries]
            print(f"  {label:<35} {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # Save combined results
    # Write the cross-seed summary. The 'aggregate' block (mean/std per metric) is
    # only populated when >1 seed was run, else it is an empty dict.
    combined_path = OUTPUT_DIR / 'mixture_summary.json'
    with open(combined_path, 'w') as f:
        json.dump({
            'n_seeds': len(all_summaries),
            'seeds': seeds,
            'per_seed': all_summaries,
            'aggregate': {
                key: {
                    'mean': float(np.mean([s[key] for s in all_summaries])),
                    'std': float(np.std([s[key] for s in all_summaries])),
                }
                # Iterate the same metric keys (labels unused here, hence '').
                for key, _ in [
                    ('pair_sim_to_components_mean', ''),
                    ('pair_sim_to_linear_pred_mean', ''),
                    ('pair_suppression_mean', ''),
                    ('triplet_suppression_mean', ''),
                    ('individual_svm_accuracy', ''),
                    ('mixture_vs_individual_svm', ''),
                    ('inter_mixture_svm_accuracy', ''),
                ]
            } if len(all_summaries) > 1 else {},
        }, f, indent=2)
    print(f"\nCombined summary saved: {combined_path}")


if __name__ == '__main__':
    main()
