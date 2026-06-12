"""Analysis subpackage for connectome model paper figures.

This package is the post-hoc analysis / figure-generation layer that sits on
TOP of the spiking network defined in ``model.py``. It does NOT train or
simulate anything new on import; instead it re-exports the helper functions
used by the paper notebook (``code/paper_figures.ipynb``) and the
``recompute.py`` drivers to turn trained checkpoints into the numbers and
panels that appear in the CCN 2026 paper.

Pipeline recap (for context on what these analyses consume):
    OR responses (Kreher 2008) -> ORN (LIF) -> LN (LIF) -> PN (LIF)
        -> KC (two-compartment) <- APL (graded divisive inhibition)
        -> linear decoder over 28 odors.
The functions below operate on the OUTPUTS of that pipeline (spike/rate
patterns at the OR, PN and KC stages, the learned biological parameters, and
the decoder), computing representational-similarity / decorrelation metrics,
classification accuracy, parameter statistics, and the corresponding plots.

This module itself contains NO executable logic beyond the imports: it is a
flat namespace facade. The actual implementations live in three sibling
modules, grouped by responsibility:

    * ``utils``    -- low-level, stateless numeric helpers (cosine similarity,
                      noisy forward passes, Hill concentration scaling,
                      parameter extraction).
    * ``compute``  -- higher-level metric computations that run models on data
                      and aggregate results across seeds (load_models,
                      evaluate_model, decorrelation, Mancini test,
                      cross-model/KC consistency, concentration invariance,
                      coefficient-of-variation of few-shot parameters).
    * ``plotting`` -- matplotlib figure builders, one per paper panel.

By re-exporting everything here, downstream code can pull any analysis
function from a single, stable import path regardless of which sibling module
actually defines it.

Re-exports all functions from utils, compute, and plotting modules.

Usage:
    from analysis import (
        load_models, evaluate_model, compute_per_pair_decorrelation,
        plot_training_curves, plot_summary, ...
    )
"""

# --- Stateless numeric / extraction helpers (analysis/utils.py) -------------
# These are pure-ish utilities: similarity math, noise-injecting forward
# passes, the Hill-equation concentration transform, and routines that pull the
# ~449 learned biological parameters out of a trained ``model`` instance into
# plain dicts/arrays for downstream statistics.
from .utils import (
    cosine_sim_pair,            # cosine similarity between two 1-D pattern vectors (a.b / (|a||b|))
    cosine_sim_matrix,          # full pairwise cosine-similarity matrix for a stack of patterns
    noisy_forward_pass,         # run model over OR responses for n_trials with injected noise (seeded), returns per-stage patterns
    per_pair_similarity_ratios, # per odor-pair similarity at OR/PN/KC stages -> quantifies stage-wise decorrelation
    centroid_classify,          # nearest-centroid classification of a test trial against per-odor centroids
    hill_effective_concentration,  # Hill-equation transform c -> c^n / (ec50^n + c^n); models OR ligand binding vs. concentration
    extract_all_parameters,     # unpack every learned biological parameter from the model into a labelled dict (handles log-space exp/softplus storage)
    compute_pairwise_correlations,  # correlations of parameter vectors across the seed ensemble of models
    analyze_biological_parameters,  # summary statistics over a single model's biological parameters
)

# --- Model-running metric computations (analysis/compute.py) ----------------
# These functions instantiate / load trained checkpoints and execute the
# spiking network to produce the paper's quantitative results, aggregating
# across the seed ensemble. NOTE: load_models defaults to n_steps=30, the
# CANONICAL runtime value from the run_*.py drivers (model.py's former n_steps=20 and
# in-class compute_loss are LEGACY defaults that the scripts override).
from .compute import (
    load_models,                    # load per-seed checkpoints into model instances (n_steps=30 canonical; include_nonad toggles non-adapting connections)
    evaluate_model,                 # decoder classification accuracy of one model on a test DataLoader
    compute_per_pair_decorrelation, # per odor-pair decorrelation across stages under noisy trials (n_trials, noise_std injected at the OR input)
    compute_mean_sim_decorrelation, # mean within-/between-odor similarity decorrelation summary
    run_mancini_test,               # in-silico carbachol + APL-injection perturbation (Mancini-style) validation experiment
    evaluate_per_odor_all_models,   # per-odor accuracy across every model in the seed ensemble
    compute_cross_model_consistency,# representational consistency of the same odor ACROSS independently trained seeds
    compute_kc_consistency_per_odor,# per-odor KC-layer response consistency across models
    centroid_accuracy,              # nearest-centroid accuracy under noise (n_trials, noise_std) as an alternative to the linear decoder
    extract_gap_junction_info,      # pull learned gap-junction conductances (LN-LN, PN-PN sister, eLN-PN) from a model
    extract_nonad_strengths,        # pull non-adapting connection strengths from a model
    run_concentration_invariance,   # sweep odor concentrations (Hill-scaled) and measure representation stability (concentration invariance)
    compute_few_param_cv,           # coefficient of variation of the few-shot / sparse learned parameters across seeds (min_params guard)
)

# --- Figure builders (analysis/plotting.py) ---------------------------------
# One function per paper/supplementary panel. Each takes precomputed metrics
# (from the compute/utils functions above) and renders a matplotlib figure,
# optionally saving to output_path and/or showing it. They produce the PNG/PDF
# panels under figures/ referenced by the paper.
from .plotting import (
    plot_training_curves,           # student vs. teacher accuracy training curves per seed (Fig: training)
    plot_per_odor_breakdown,        # per-odor accuracy and KC sparsity breakdown bars
    plot_kc_sparsity_distribution,  # distribution of KC activity sparsity per odor (KC coding sparseness)
    plot_biological_parameters,     # summary plot of the learned biological parameter values
    plot_correlation_bars,          # bar chart of cross-model parameter correlations
    plot_few_param_cv,              # bar chart of the few-parameter coefficient of variation
    plot_mancini_validation,        # visualization of the Mancini carbachol/APL perturbation results across seeds
    plot_gap_junction_conductances, # learned gap-junction conductance values across the seed ensemble
    plot_ln_pn_split,               # LN vs. PN contribution split visualization across seeds
    plot_core_figure,               # the main "core" figure: OR/PN/KC similarity + optional KC activity (paper Fig 2_core)
    plot_kc_heatmap,                # heatmap of KC activity across odors (paper Fig S1_kc_heat)
    plot_concentration,             # concentration-invariance / Hill-curve figure (paper Fig 3_concentration; hill_ec50 sets the EC50)
)
