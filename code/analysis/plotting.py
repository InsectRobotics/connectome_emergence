"""Plotting functions for connectome model analysis.

This module is the *visualization* layer of the larval-Drosophila olfactory
connectome SNN project. It consumes already-computed summary statistics and
metrics (produced by ``analysis/compute.py`` / ``analysis/recompute.py`` and
the ``run_*.py`` drivers, or loaded from the cached ``results/*.json``
checkpoints) and turns them into the publication figures (Figures 2-3) and
supplementary/diagnostic panels (S1-S3) of the CCN 2026 paper.

It does NOT run the spiking model or recompute anything biophysical itself;
every numeric input (accuracies, sparsities, similarity matrices, learned
biological parameters, APL/Mancini ratios, gap-junction conductances, etc.)
is passed in by the caller as plain arrays/dicts. The pipeline it visualizes
is:

    OR responses (Kreher 2008) -> ORN (LIF) -> LN (LIF) -> PN (LIF)
        -> KC (two-compartment) <- APL (graded divisive inhibition)
        -> linear decoder over 28 odors.

Each function in this file follows the same convention so it can be driven
uniformly from a notebook or a headless script:
  - Returns the matplotlib Figure object
  - Takes show=False kwarg (True keeps figure open for Jupyter display)
  - Takes output_path=None kwarg (saves to file if provided)
  - Does NOT set matplotlib.use('Agg') (caller's responsibility)
  - Accepts constants (seeds, concentrations, etc.) as arguments

The "show / close" pattern (``if not show: plt.close(fig)``) means that when a
figure is being saved in a batch/headless run it is closed to free memory,
but when displayed interactively in Jupyter it is left open so the notebook
front-end can render it.
"""

# numpy: array math for means/stds/normalization used throughout.
import numpy as np
# torch: only needed because some inputs (e.g. KC activity tensors) may arrive
# as torch tensors that must be detached/moved to CPU before plotting.
import torch
import matplotlib.pyplot as plt
# mcolors: provides LogNorm / PowerNorm color normalizations for heatmaps.
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import SymLogNorm
# FixedLocator/FixedFormatter: place explicit, human-readable colorbar ticks on
# a log-scaled axis (otherwise log ticks read as e.g. 10^-0.3).
from matplotlib.ticker import FixedLocator, FixedFormatter

# hill_effective_concentration: Hill dose-response transfer (saturating curve);
# used as a reference overlay in the concentration-invariance figure to show
# what pure receptor-level (no circuitry) gain compression would look like.
from .utils import hill_effective_concentration


# ---------------------------------------------------------------------------
# Training curves (S1)
# ---------------------------------------------------------------------------

def plot_training_curves(histories, teacher_accs, seeds=None,
                         output_path=None, show=False):
    """Plot training accuracy and sparsity curves for all spiking models (Fig S1).

    Visualizes the ANN-to-SNN transfer: each spiking "student" is trained to
    match a rate "teacher", and we track (left) decoder test accuracy and
    (right) Kenyon-cell (KC) sparsity over training epochs.

    Parameters
    ----------
    histories : list[dict]
        One entry per seed/model. Each dict has keys:
          'epochs'   -> list of epoch indices,
          'test_acc' -> list of test accuracies as fractions in [0, 1],
          'sparsity' -> list of KC sparsity values as fractions in [0, 1]
                        (fraction of KCs active per odor).
    teacher_accs : sequence[float]
        Rate-teacher accuracy (fraction in [0, 1]) for each model; drawn as a
        dashed reference line (the target the spiking students aim at).
    seeds : sequence, optional
        Random seeds (unused here; accepted for a uniform call signature).
    output_path : str or None
        If given, the figure is written there at 150 dpi.
    show : bool
        If False, the figure is closed after saving (headless/batch mode).

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel figure (accuracy | sparsity).
    """
    # Two side-by-side panels: [0] accuracy, [1] sparsity.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left panel: test accuracy vs epoch, one curve per model. ---
    ax = axes[0]
    for i, hist in enumerate(histories):
        epochs = hist['epochs']
        # Convert stored fractions to percentages for display.
        accs = [a * 100 for a in hist['test_acc']]
        ax.plot(epochs, accs, label=f'Spiking {i+1} (teacher: {teacher_accs[i]:.1%})', alpha=0.8)
    # Horizontal reference: mean rate-teacher accuracy (the asymptotic target).
    ax.axhline(y=np.mean(teacher_accs) * 100, color='red', linestyle='--',
               label=f'Mean teacher ({np.mean(teacher_accs):.1%})', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training Accuracy'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # --- Right panel: KC sparsity vs epoch, one curve per model. ---
    ax = axes[1]
    for i, hist in enumerate(histories):
        epochs = hist['epochs']
        # Sparsity fractions -> percent of KCs active.
        sps = [s * 100 for s in hist['sparsity']]
        ax.plot(epochs, sps, label=f'Model {i+1}', alpha=0.8)
    # Target sparsity line: ~10% active KCs, the biologically-motivated set point
    # the sparsity loss drives the network toward.
    ax.axhline(y=10, color='green', linestyle='--', label='Target (10%)', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('KC Sparsity (%)')
    ax.set_title('KC Sparsity'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    # In batch mode (show=False) free the figure to avoid memory growth.
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Per-odor breakdown (diagnostic)
# ---------------------------------------------------------------------------

def plot_per_odor_breakdown(per_odor_acc, per_odor_sparsity, odor_names,
                            output_path=None, show=False):
    """Plot per-odor decoder accuracy and KC sparsity as horizontal bars.

    Diagnostic that breaks the aggregate metrics down by individual odor so
    one can see which of the 28 odors are easy/hard to classify and which
    evoke the right amount of KC activity.

    Parameters
    ----------
    per_odor_acc : sequence of (mean, std)
        Per-odor classification accuracy as (mean, std) fractions in [0, 1],
        aggregated across the n=5 seeds/models.
    per_odor_sparsity : sequence of (mean, std)
        Per-odor KC sparsity as (mean, std) fractions in [0, 1].
    odor_names : sequence[str]
        Human-readable odor labels, indexed identically to the metric lists.
    output_path : str or None
        If given, figure written there at 150 dpi.
    show : bool
        If False, close after saving.

    Returns
    -------
    matplotlib.figure.Figure
        Two-panel figure (accuracy bars | sparsity bars), odors sorted by
        accuracy descending.
    """
    # Unpack (mean, std) tuples into parallel lists.
    acc_means = [c[0] for c in per_odor_acc]
    acc_stds = [c[1] for c in per_odor_acc]
    sp_means = [c[0] for c in per_odor_sparsity]
    sp_stds = [c[1] for c in per_odor_sparsity]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    # Sort odors by accuracy, descending, so the best classified appear on top.
    sorted_idx = np.argsort(acc_means)[::-1]
    # Apply the sort order and rescale fractions -> percent for both metrics.
    sorted_acc = [acc_means[i] * 100 for i in sorted_idx]
    sorted_acc_std = [acc_stds[i] * 100 for i in sorted_idx]
    sorted_sp = [sp_means[i] * 100 for i in sorted_idx]
    sorted_sp_std = [sp_stds[i] * 100 for i in sorted_idx]
    sorted_names = [odor_names[i] for i in sorted_idx]
    y_pos = np.arange(len(sorted_names))

    # --- Accuracy bars: color-code by performance band (green/orange/red). ---
    colors = ['#2ecc71' if a > 70 else '#f39c12' if a > 50 else '#e74c3c' for a in sorted_acc]
    ax1.barh(y_pos, sorted_acc, xerr=sorted_acc_std, color=colors, edgecolor='black',
             alpha=0.8, capsize=2, error_kw={'linewidth': 1})
    # Vertical line at the mean accuracy across odors.
    ax1.axvline(x=np.mean(acc_means) * 100, color='blue', linestyle='--',
                label=f'Mean: {np.mean(acc_means)*100:.1f}%')
    ax1.set_yticks(y_pos); ax1.set_yticklabels(sorted_names, fontsize=9)
    ax1.set_xlabel('Accuracy (%)'); ax1.set_title('Per-Odor Accuracy (n=5 models)'); ax1.legend()

    # --- Sparsity bars: blue if within the healthy 5-20% band, else red. ---
    colors = ['#3498db' if 5 <= s <= 20 else '#e74c3c' for s in sorted_sp]
    ax2.barh(y_pos, sorted_sp, xerr=sorted_sp_std, color=colors, edgecolor='black',
             alpha=0.8, capsize=2, error_kw={'linewidth': 1})
    # Target sparsity (10%) and the across-odor mean.
    ax2.axvline(x=10, color='green', linestyle='--', label='Target: 10%')
    ax2.axvline(x=np.mean(sp_means) * 100, color='blue', linestyle='--',
                label=f'Mean: {np.mean(sp_means)*100:.1f}%')
    # Hide y tick labels here since the left panel already shows odor names.
    ax2.set_yticks(y_pos); ax2.set_yticklabels([]); ax2.set_xlabel('KC Sparsity (%)')
    ax2.set_title('Per-Odor KC Sparsity (n=5 models)'); ax2.legend()

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# KC sparsity distribution (diagnostic)
# ---------------------------------------------------------------------------

def plot_kc_sparsity_distribution(per_odor_kc, output_path=None, show=False):
    """Plot KC activity heatmap (odor x KC) and its marginal histogram.

    Diagnostic showing the raw KC firing pattern: a log-scaled heatmap of mean
    spike counts (rows = odors, columns = KCs) plus a histogram of the per-cell
    spike counts. Used to confirm sparse, near-binary KC coding.

    Parameters
    ----------
    per_odor_kc : torch.Tensor (n_odors, n_kcs) or list of such rows
        Mean KC spike counts per odor (averaged over trials). Accepts either a
        single tensor or a list of per-odor tensors to be stacked.
    output_path : str or None
        If given, figure written there at 150 dpi.
    show : bool
        If False, close after saving.

    Returns
    -------
    matplotlib.figure.Figure
        Two-panel figure (log-scale heatmap | activity histogram).
    """
    # Normalize input to a numpy (n_odors, n_kcs) array; detach from autograd
    # and move off any GPU before handing to matplotlib.
    if isinstance(per_odor_kc, list):
        kc_activity = torch.stack(per_odor_kc).detach().cpu().numpy()
    else:
        kc_activity = per_odor_kc.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # Add a small floor (0.1) so true zeros are representable on a log color
    # scale (log(0) is undefined); the +0.1 offset maps "silent" to the floor.
    kc_activity_for_plot = kc_activity + 0.1
    vmin = 0.1
    # Clamp the color ceiling to [1, 10] spikes so a few hot cells don't wash
    # out the dynamic range of the rest.
    vmax = min(max(kc_activity.max(), 1.0), 10.0)
    im = ax1.imshow(kc_activity_for_plot, aspect='auto', cmap='viridis',
                    norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
    ax1.set_xlabel('KC Index'); ax1.set_ylabel('Odor Index')
    ax1.set_title('KC Activity Pattern per Odor (Best Model, log scale)')
    cbar = plt.colorbar(im, ax=ax1, label='Mean Spike Count (n=20 trials)')
    # Re-label the log colorbar with human-readable spike counts; note the 0.1
    # floor is displayed as "0" since it represents silent cells.
    cbar.locator = FixedLocator([0.1, 0.5, 1, 2, 5, 10])
    cbar.formatter = FixedFormatter(['0', '0.5', '1', '2', '5', '10'])
    cbar.update_ticks()

    # Right panel: histogram of all per-cell spike counts (the un-offset data).
    ax2.hist(kc_activity.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('KC Activity (Spike Count)'); ax2.set_ylabel('Count')
    ax2.set_title('Distribution of KC Activity')
    # Mark the "silent" (zero spikes) location; sparse coding piles mass here.
    ax2.axvline(x=0, color='red', linestyle='--', label='Silent'); ax2.legend()

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Biological parameters (diagnostic)
# ---------------------------------------------------------------------------

def plot_biological_parameters(bio_params, output_path=None, show=False):
    """Plot learned biophysical parameters against biological reference ranges.

    Two panels:
      (left)  Per-population mean spike threshold v_th (mV) with min/max
              whiskers, overlaid on the biological band [-55 mV, -30 mV] used
              to constrain the learnable thresholds.
      (right) The single learned somatic coupling conductance g_soma (nS) of
              the two-compartment KC model, with its clamp bounds shown
              (canonically [1, 20] nS).

    Parameters
    ----------
    bio_params : dict
        Summary of learned parameters. Expected structure:
          bio_params['v_th']['populations'][pop] -> {'mean_mV','min_mV','max_mV'}
          bio_params['v_th']['pct_in_bounds']    -> float (percent in band)
          bio_params['g_soma'] (optional)        ->
              {'value_nS', 'std_nS'(opt), 'bounds_nS', 'in_bounds',
               'all_values_nS'(opt)}
    output_path : str or None
        If given, figure written there at 150 dpi.
    show : bool
        If False, close after saving.

    Returns
    -------
    matplotlib.figure.Figure
        Two-panel figure (v_th by population | g_soma).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: spike threshold v_th by neuron population. ---
    ax = axes[0]
    populations = bio_params['v_th']['populations']
    pop_names = list(populations.keys())
    means = [populations[p]['mean_mV'] for p in pop_names]
    mins = [populations[p]['min_mV'] for p in pop_names]
    maxs = [populations[p]['max_mV'] for p in pop_names]
    x = np.arange(len(pop_names))
    ax.bar(x, means, color='steelblue', edgecolor='black', alpha=0.7)
    # Asymmetric whiskers: distance from mean down to min and up to max.
    ax.errorbar(x, means, yerr=[np.array(means) - np.array(mins),
                                 np.array(maxs) - np.array(means)],
                fmt='none', color='black', capsize=5)
    # Biological threshold band: roughly -55 mV (floor) to -30 mV (ceiling).
    ax.axhline(y=-55, color='red', linestyle='--', linewidth=2, label='Bio min (-55mV)')
    ax.axhline(y=-30, color='red', linestyle='--', linewidth=2, label='Bio max (-30mV)')
    ax.axhspan(-55, -30, alpha=0.1, color='green', label='Biological range')
    ax.set_xticks(x); ax.set_xticklabels(pop_names); ax.set_ylabel('Threshold (mV)')
    ax.set_title(f"v_th Distribution by Population\n"
                 f"({bio_params['v_th']['pct_in_bounds']:.0f}% in biological bounds)")
    ax.legend(loc='upper right', fontsize=8)

    # --- Right: learned somatic coupling conductance g_soma (if present). ---
    ax = axes[1]
    if 'g_soma' in bio_params:
        g_val = bio_params['g_soma']['value_nS']
        # std across seeds may be absent; default 0 (single value / no whisker).
        g_std = bio_params['g_soma'].get('std_nS', 0)
        bounds = bio_params['g_soma']['bounds_nS']
        # Color the bar by whether the learned value stayed inside the clamp.
        ax.bar(['g_soma'], [g_val], yerr=[g_std] if g_std > 0 else None, capsize=5,
               color='coral' if bio_params['g_soma']['in_bounds'] else 'red', edgecolor='black')
        # g_soma clamp bounds (canonical [1, 20] nS): the soma<->dendrite
        # coupling conductance the matrix-exponential KC integrator uses.
        ax.axhline(y=bounds[0], color='green', linestyle='--', linewidth=2,
                   label=f'Min ({bounds[0]:.0f} nS)')
        ax.axhline(y=bounds[1], color='green', linestyle='--', linewidth=2,
                   label=f'Max ({bounds[1]:.0f} nS)')
        ax.axhspan(bounds[0], bounds[1], alpha=0.1, color='green')
        ax.set_ylabel('Conductance (nS)')
        # n = number of seeds contributing to the mean/std, if recorded.
        n_models = len(bio_params['g_soma'].get('all_values_nS', [1]))
        ax.set_title(f"Learned g_soma (n={n_models})\n({g_val:.1f}+-{g_std:.1f} nS)")
        ax.legend(loc='upper right')
        # Annotate the bar with mean +- std.
        ax.text(0, g_val + g_std + 2, f'{g_val:.1f}+-{g_std:.1f} nS',
                ha='center', fontweight='bold')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Parameter correlation bars (S2a)
# ---------------------------------------------------------------------------

def plot_correlation_bars(correlations, title, all_params=None,
                          min_params_for_correlation=10,
                          output_path=None, show=False):
    """Plot Pairwise Pearson correlation bars per parameter category (Fig S2a).

    Used for the seed-consistency analysis: across independently-trained seeds,
    how reproducible is each category of learned parameter? A correlation near
    1.0 means different seeds converge to nearly identical parameter vectors.

    Groups with < min_params_for_correlation are excluded (use plot_few_param_cv).
    (A Pearson correlation over very few parameters is statistically
    unreliable, so small groups are reported separately via the CV plot.)

    Parameters
    ----------
    correlations : dict[str, sequence[float]]
        Maps category name -> list of pairwise Pearson r values (one per pair
        of seeds). May contain a special 'Overall' key spanning all params.
    title : str
        Figure title.
    all_params : list[dict] or None
        If provided, all_params[0] (a reference seed's category -> param tensor
        mapping) is inspected so categories with too few parameters can be
        excluded.
    min_params_for_correlation : int
        Threshold below which a category is dropped from this plot.
    output_path : str or None
        If given, figure written there at 150 dpi (white facecolor).
    show : bool
        If False, close after saving.

    Returns
    -------
    matplotlib.figure.Figure
        Bar chart of mean pairwise correlation (+- std) per category.
    """
    # Determine which categories are too small to correlate meaningfully.
    exclude_cats = set()
    if all_params:
        ref_params = all_params[0]
        for cat in ref_params:
            # 'Total'/'Overall' are aggregate buckets, never individual groups.
            if cat in ('Total', 'Overall'):
                continue
            if len(ref_params[cat]) < min_params_for_correlation:
                exclude_cats.add(cat)

    # Assemble bar data; put 'Overall' first so it anchors the left edge.
    categories, means, stds = [], [], []
    if 'Overall' in correlations:
        categories.append('Overall')
        means.append(np.mean(correlations['Overall']))
        stds.append(np.std(correlations['Overall']))
    for cat in sorted(correlations.keys()):
        if cat != 'Overall' and cat not in exclude_cats:
            categories.append(cat)
            means.append(np.mean(correlations[cat]))
            stds.append(np.std(correlations[cat]))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(categories))
    # Highlight the 'Overall' bar in green; per-category bars in blue.
    colors = ['#2ecc71' if cat == 'Overall' else '#3498db' for cat in categories]
    ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Pairwise Pearson Correlation', fontsize=12)
    ax.set_xlabel('Parameter Category', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
    # Correlations live in [-1, 1]; cap slightly above 1 to leave room for text.
    ax.set_ylim(0, 1.1)
    # Reference line at perfect correlation r=1.0.
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    # Numeric annotation above each bar (placed above its error cap).
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.03, f'{m:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Few-parameter CV (S2b)
# ---------------------------------------------------------------------------

def plot_few_param_cv(cv_results, title, output_path=None, show=False):
    """Plot coefficient of variation for small parameter groups (Fig S2b).

    Companion to ``plot_correlation_bars``: for categories with too few
    parameters to correlate reliably, consistency across seeds is summarized
    instead by the coefficient of variation CV = std / |mean| (lower = more
    reproducible). Returns None if there are no such groups.

    Parameters
    ----------
    cv_results : dict[str, dict]
        Maps category name -> {'mean_cv': float, 'n_params': int}.
    title : str
        Figure title.
    output_path : str or None
        If given, figure written there at 150 dpi (white facecolor).
    show : bool
        If False, close after saving.

    Returns
    -------
    matplotlib.figure.Figure or None
        Bar chart of per-group CV, or None when cv_results is empty.
    """
    # Nothing to plot if no small groups were collected.
    if not cv_results:
        return None

    categories = sorted(cv_results.keys())
    cvs = [cv_results[cat]['mean_cv'] for cat in categories]
    # Track parameter count per group to annotate "(Np)" on each bar.
    n_params = [cv_results[cat]['n_params'] for cat in categories]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(categories))
    bars = ax.bar(x, cvs, color='#e67e22', edgecolor='black', alpha=0.8)
    # Annotate each bar with its CV value and the number of parameters.
    for i, (bar, cv, n) in enumerate(zip(bars, cvs, n_params)):
        ax.text(bar.get_x() + bar.get_width() / 2., cv + 0.005,
                f'{cv:.3f}\n({n}p)', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Coefficient of Variation (CV = std/|mean|)', fontsize=12)
    ax.set_xlabel('Parameter Group', fontsize=12); ax.set_title(title, fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
    # Headroom above the tallest bar for the text annotations.
    ax.set_ylim(0, max(cvs) * 1.4 if cvs else 1.0); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    # Reference bands: CV<=0.05 "very consistent", CV~0.20 "moderate variation".
    ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.4,
               label='CV=0.05 (very consistent)')
    ax.axhline(y=0.20, color='orange', linestyle='--', alpha=0.4,
               label='CV=0.20 (moderate variation)')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Mancini validation (S3)
# ---------------------------------------------------------------------------

def plot_mancini_validation(mancini_results, seeds, output_path=None, show=False):
    """Plot per-seed APL-inhibition ratios vs the Mancini et al. 2023 benchmark.

    Validates the divisive (shunting) APL feedback: Mancini et al. 2023 report
    that activating the APL neuron roughly halves KC calcium/activity. The
    "Mancini ratio" here = (KC spikes WITHOUT APL) / (KC spikes WITH APL); a
    ~2x ratio reproduces that halving. Each bar is one trained seed; green if
    it passes the acceptance criterion, red otherwise.

    Parameters
    ----------
    mancini_results : list[dict]
        One dict per seed with keys 'ratio' (float) and 'passes' (bool).
    seeds : sequence
        Seed identifiers, used for x-axis tick labels.
    output_path : str or None
        If given, figure written there at 150 dpi.
    show : bool
        If False, close after saving.

    Returns
    -------
    matplotlib.figure.Figure
        Single-axis bar chart of ratios with the target band annotated.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ratios = [r['ratio'] for r in mancini_results]
    passes = [r['passes'] for r in mancini_results]
    # Green = passes the biological criterion, red = fails.
    colors = ['#2ecc71' if p else '#e74c3c' for p in passes]
    x = np.arange(len(seeds))
    ax.bar(x, ratios, color=colors, edgecolor='black', alpha=0.8)
    # Target ratio 2.0 (APL halves KC activity) and acceptable band [1.5, 2.5].
    ax.axhline(y=2.0, color='blue', linestyle='--', linewidth=2, label='Target (2.0)')
    ax.axhspan(1.5, 2.5, alpha=0.1, color='green', label='Acceptable range (1.5-2.5)')
    ax.axhline(y=1.5, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=2.5, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f'Seed {s}' for s in seeds])
    ax.set_ylabel('Mancini Ratio (baseline/boosted)', fontsize=12)
    ax.set_title(f'APL Inhibition Validation (Mancini et al. 2023)\n'
                 f'Mean: {np.mean(ratios):.2f} +- {np.std(ratios):.2f}', fontsize=14)
    ax.legend(loc='upper right'); ax.set_ylim(0, 3.0)
    # Annotate each bar with its ratio.
    for i, r in enumerate(ratios):
        ax.text(i, r + 0.1, f'{r:.2f}', ha='center', fontsize=10)
    # In-axes caption restating the experimental definition for the reader.
    ax.text(0.02, 0.02,
            'Mancini et al. 2023: APL activation reduces KC calcium to ~50%\n'
            'Ratio = KC spikes without APL / KC spikes with APL',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Gap junction conductances (diagnostic)
# ---------------------------------------------------------------------------

def plot_gap_junction_conductances(all_gap_info, seeds, output_path=None, show=False):
    """Plot learned electrical (gap-junction) coupling conductances per seed.

    The model includes electrical synapses (gap junctions) that pass current
    I = g * (V_pre - V_post) between coupled neurons in three pools:
    LN-LN, PN-PN (sister PNs), and eLN-PN. This diagnostic shows the learned
    coupling conductance g for each pool across seeds, in picosiemens (pS).

    Parameters
    ----------
    all_gap_info : list[dict]
        One dict per seed mapping each gap key ('ln_ln','pn_pn','eln_pn') to a
        conductance in SIEMENS (S), or None if that coupling is absent.
    seeds : sequence
        Seed identifiers for x-axis tick labels.
    output_path : str or None
        If given, figure written there at 150 dpi.
    show : bool
        If False, close after saving.

    Returns
    -------
    matplotlib.figure.Figure
        Three-panel figure, one per gap-junction type.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # (dict key, display title, bar color) for each gap-junction pool.
    gap_types = [
        ('ln_ln', 'LN-LN', '#8e44ad'),
        ('pn_pn', 'PN-PN', '#2980b9'),
        ('eln_pn', 'eLN-PN', '#27ae60'),
    ]
    for ax, (key, title, color) in zip(axes, gap_types):
        # Collect non-None conductances for this pool across seeds (in S).
        vals = [g[key] for g in all_gap_info if g[key] is not None]
        if not vals:
            # This coupling type is not present in the model; skip its panel.
            ax.set_title(f'{title}\n(not present)')
            continue
        # Convert siemens -> picosiemens (1 S = 1e12 pS) for readable axes.
        vals_ps = [v * 1e12 for v in vals]
        x = np.arange(len(vals))
        ax.bar(x, vals_ps, color=color, edgecolor='black', alpha=0.8)
        ax.axhline(y=np.mean(vals_ps), color='red', linestyle='--',
                   label=f'Mean: {np.mean(vals_ps):.1f} pS', linewidth=2)
        ax.set_xticks(x); ax.set_xticklabels([f'Seed {s}' for s in seeds[:len(vals)]])
        ax.set_ylabel('Conductance (pS)')
        ax.set_title(f'{title}\n({np.mean(vals_ps):.1f} +- {np.std(vals_ps):.1f} pS)')
        ax.legend(fontsize=9)
        # Annotate each seed's bar with its value.
        for i, v in enumerate(vals_ps):
            ax.text(i, v + 1, f'{v:.1f}', ha='center', fontsize=9)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# LN->PN split (diagnostic)
# ---------------------------------------------------------------------------

def plot_ln_pn_split(all_ln_pn_split, seeds, output_path=None, show=False):
    """Plot the learned inhibitory vs excitatory LN->PN synaptic strengths.

    Local neurons (LNs) project to projection neurons (PNs). Different LN
    subtypes are inhibitory (Broad/Choosy) or excitatory (Picky). This
    diagnostic shows, per seed, the aggregate inhibitory and excitatory LN->PN
    chemical synaptic strengths (in nA) and their ratio, confirming that the
    AL's lateral interactions are inhibition-dominated.

    Parameters
    ----------
    all_ln_pn_split : list[dict]
        One dict per seed with keys:
          'inhibitory_strength', 'excitatory_strength' -> strengths in AMPERES,
          (optional) 'n_excitatory_ln', 'n_total_ln'   -> LN subtype counts.
    seeds : sequence
        Seed identifiers for x-axis tick labels.
    output_path : str or None
        If given, figure written there at 150 dpi.
    show : bool
        If False, close after saving.

    Returns
    -------
    matplotlib.figure.Figure
        Two-panel figure (grouped strength bars | inhibition/excitation ratio).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Convert amperes -> nanoamperes (1 A = 1e9 nA) for both strength lists.
    inhib_vals = [s['inhibitory_strength'] * 1e9 for s in all_ln_pn_split]
    excit_vals = [s['excitatory_strength'] * 1e9 for s in all_ln_pn_split]
    x = np.arange(len(seeds)); width = 0.35
    # Grouped bars per seed: inhibitory (red, left) vs excitatory (green, right).
    ax1.bar(x - width/2, inhib_vals, width, color='#c0392b', edgecolor='black',
            alpha=0.8, label='Inhibitory')
    ax1.bar(x + width/2, excit_vals, width, color='#27ae60', edgecolor='black',
            alpha=0.8, label='Excitatory')
    ax1.set_xticks(x); ax1.set_xticklabels([f'Seed {s}' for s in seeds])
    ax1.set_ylabel('Synaptic Strength (nA)')
    # Title reports inhib in nA and excit in pA (excit*1e3 converts nA->pA).
    ax1.set_title(f'LN->PN Pathway Strengths\nInhib: {np.mean(inhib_vals):.3f} nA, '
                  f'Excit: {np.mean(excit_vals)*1e3:.3f} pA')
    # Log scale because inhibitory and excitatory magnitudes differ by orders.
    ax1.legend(); ax1.set_yscale('log')

    # Inhibition/excitation ratio per seed; max(e, 1e-15) avoids divide-by-zero.
    ratios = [i / max(e, 1e-15) for i, e in zip(inhib_vals, excit_vals)]
    ax2.bar(x, ratios, color='#f39c12', edgecolor='black', alpha=0.8)
    ax2.axhline(y=np.mean(ratios), color='red', linestyle='--',
               label=f'Mean ratio: {np.mean(ratios):.0f}x', linewidth=2)
    ax2.set_xticks(x); ax2.set_xticklabels([f'Seed {s}' for s in seeds])
    ax2.set_ylabel('Inhibitory / Excitatory Ratio')
    ax2.set_title(f'LN->PN Inhibition Dominance\n(Ratio: {np.mean(ratios):.0f}x)')
    ax2.legend()

    # Optional footnote describing the LN-subtype composition (counts).
    if 'n_excitatory_ln' in all_ln_pn_split[0]:
        n_excit = all_ln_pn_split[0]['n_excitatory_ln']
        n_total = all_ln_pn_split[0]['n_total_ln']
        fig.text(0.5, 0.01, f'LN subtypes: {n_excit}/{n_total} excitatory (Picky), '
                f'{n_total - n_excit}/{n_total} inhibitory (Broad/Choosy)',
                ha='center', fontsize=10, style='italic')

    # Reserve the bottom 4% of the figure for the footnote text.
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Core figure: similarity matrices + KC heatmap (Figure 2)
# ---------------------------------------------------------------------------

def plot_core_figure(or_sim, pn_sim, kc_sim, kc_activity=None,
                     output_path=None, show=False):
    """Figure 2: 1x3 — A (OR sim), B (PN sim), C (KC sim).

    The paper's core "decorrelation" figure: three pairwise cosine-similarity
    matrices over the 28 odors, one at each stage of the pathway (receptor OR
    input, antennal-lobe PN output, mushroom-body KC output). Reading A->B->C
    shows how the circuit progressively decorrelates odor representations
    (off-diagonal similarity shrinks), the substrate for sparse KC coding.

    The bulk of the function is manual axes placement (in figure fractions)
    computed from a fixed pixel-per-cell scale so all three matrices render at
    identical physical size with consistent gaps and a shared colorbar.

    Parameters
    ----------
    or_sim, pn_sim, kc_sim : ndarray (n_odors, n_odors)
        Pairwise cosine similarity matrices.
    kc_activity : ignored (kept for backward compatibility).
    output_path : str or None
        If given, saved as PNG (200 dpi) and, when the path ends in .png, also
        as a sibling PDF (300 dpi) for vector publication.
    show : bool
        If False, close after saving.

    Returns
    -------
    matplotlib.figure.Figure
        The three-panel similarity figure with a shared colorbar.
    """
    n_odors = or_sim.shape[0]

    # Layout is described in abstract "cells", then scaled to inches via `pix`.
    mat_size = n_odors           # 28
    gap_top = 9                  # horizontal gap between panels A/B/C (was 6)
    cb_gap = 3                   # gap (cells) before the colorbar
    cb_w = 2                     # colorbar width (cells)

    # Physical size: 0.12 inches per matrix cell, plus margins/title padding.
    pix = 0.12
    # Total layout width in cells: 3 matrices + 2 inter-panel gaps + colorbar.
    top_w = 3 * mat_size + 2 * gap_top + cb_gap + cb_w
    fig_w = top_w * pix + 3.0
    fig_h = mat_size * pix + 3.0

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Convert all layout quantities from inches to figure fractions (0-1), as
    # required by fig.add_axes([left, bottom, width, height]).
    margin_l = 1.4 / fig_w
    margin_b = 1.4 / fig_h
    mat_w_frac = mat_size * pix / fig_w
    mat_h_frac = mat_size * pix / fig_h
    gap_top_frac = gap_top * pix / fig_w
    cb_gap_frac = cb_gap * pix / fig_w
    cb_w_frac = cb_w * pix / fig_w

    # Place the three square matrix axes left-to-right, then the colorbar axis,
    # advancing the running x cursor by panel width + the appropriate gap.
    x = margin_l
    ax_a = fig.add_axes([x, margin_b, mat_w_frac, mat_h_frac])
    x += mat_w_frac + gap_top_frac
    ax_b = fig.add_axes([x, margin_b, mat_w_frac, mat_h_frac])
    x += mat_w_frac + gap_top_frac
    ax_c = fig.add_axes([x, margin_b, mat_w_frac, mat_h_frac])
    x += mat_w_frac + cb_gap_frac
    ax_cb = fig.add_axes([x, margin_b, cb_w_frac, mat_h_frac])

    # Font sizes (pt) for title / axis labels / ticks / panel letters / colorbar.
    title_fs, label_fs, tick_fs, letter_fs, cb_fs = 20, 18, 16, 20, 16

    # Odor axes are 1-indexed labels (1..n_odors), ticked every 5th odor.
    odor_ticks = list(range(n_odors))
    odor_labels = [str(i + 1) for i in range(n_odors)]

    # Shared color range: cosine similarity in [-0.2, 1.0]; the diverging
    # RdYlBu_r map puts high similarity (1.0) red and dissimilar/negative blue.
    vmin, vmax = -0.2, 1.0
    for ax, mat, title in [(ax_a, or_sim, 'OR Similarity'),
                            (ax_b, pn_sim, 'PN Similarity'),
                            (ax_c, kc_sim, 'KC Similarity')]:
        im = ax.imshow(mat, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(title, fontsize=title_fs, fontweight='bold', pad=10)
        ax.set_xlabel('Odor', fontsize=label_fs)
        ax.set_ylabel('Odor', fontsize=label_fs)
        ax.set_xticks(odor_ticks[::5])
        ax.set_xticklabels([odor_labels[i] for i in odor_ticks[::5]], fontsize=tick_fs)
        ax.set_yticks(odor_ticks[::5])
        ax.set_yticklabels([odor_labels[i] for i in odor_ticks[::5]], fontsize=tick_fs)

    # One shared colorbar (uses the last `im`; all share vmin/vmax so it's valid).
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label('Cosine Similarity', fontsize=cb_fs)
    cb.ax.tick_params(labelsize=tick_fs)

    # Panel letters A/B/C placed just above each matrix's top-left corner.
    _letter_kw = dict(fontsize=letter_fs, fontweight='bold', va='bottom', ha='left')
    for ax, letter in [(ax_a, 'A'), (ax_b, 'B'), (ax_c, 'C')]:
        ax.text(-0.06, 1.05, letter, transform=ax.transAxes, **_letter_kw)

    if output_path:
        # Raster PNG for previews; also emit a vector PDF for the manuscript.
        fig.savefig(str(output_path), dpi=200, bbox_inches='tight')
        pdf_path = str(output_path).replace('.png', '.pdf')
        if pdf_path != str(output_path):
            fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig


def plot_kc_heatmap(kc_activity, output_path=None, show=False):
    """KC activity heatmap (separate from Figure 2).

    Renders the mushroom-body KC population code as an odor x KC heatmap of
    mean spike counts, using a power-law color norm so the many low-but-nonzero
    cells remain visible against the few highly active ones. Kept as its own
    function (rather than a 4th panel of Figure 2) for flexible layout.

    Parameters
    ----------
    kc_activity : ndarray (n_odors, n_kcs)
        Mean KC firing rates per odor.
    output_path : str or None
        If given, saved as PNG (200 dpi) and, when the path ends in .png, also
        as a sibling PDF (300 dpi).
    show : bool
        If False, close after saving.

    Returns
    -------
    matplotlib.figure.Figure
        Single-axis KC heatmap with a power-norm colorbar.
    """
    # Accept torch tensors (detach/CPU) or array-likes; normalize to numpy.
    kc_data = kc_activity.detach().cpu().numpy() if hasattr(kc_activity, 'detach') else np.array(kc_activity)
    n_odors = kc_data.shape[0]

    fig, ax = plt.subplots(figsize=(10, 4))

    title_fs, label_fs, tick_fs, cb_fs = 20, 18, 16, 16
    odor_ticks = list(range(n_odors))
    odor_labels = [str(i + 1) for i in range(n_odors)]

    vmax_kc = kc_data.max()
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax_kc)  # power-law compression (gamma: 1=linear, lower=harsher, pulls low-value ticks up toward vmax)
    im = ax.imshow(kc_data, aspect='auto', cmap='viridis', norm=norm,
                   interpolation='nearest')
    ax.set_xlabel('KC Index', fontsize=label_fs)
    ax.set_ylabel('Odor', fontsize=label_fs)
    ax.set_yticks(odor_ticks[::5])
    ax.set_yticklabels([odor_labels[i] for i in odor_ticks[::5]], fontsize=tick_fs)
    ax.tick_params(axis='x', labelsize=tick_fs)

    cb = fig.colorbar(im, ax=ax)
    # Explicit spike-count ticks; under the gamma=0.5 norm these are unevenly
    # spaced in color space but read as familiar integer-ish spike counts.
    cb.set_ticks([0, 0.5, 1, 2, 5, 10])
    cb.set_ticklabels(['0', '0.5', '1', '2', '5', '10'])
    cb.set_label('Mean Spike Count', fontsize=cb_fs)
    cb.ax.tick_params(labelsize=tick_fs)

    plt.tight_layout()
    if output_path:
        fig.savefig(str(output_path), dpi=200, bbox_inches='tight')
        pdf_path = str(output_path).replace('.png', '.pdf')
        if pdf_path != str(output_path):
            fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Concentration invariance (Figure 3)
# ---------------------------------------------------------------------------

def plot_concentration(conc_data, concentrations=None, hill_ec50=1.0,
                       output_path=None, show=False):
    """Figure 3: 1x3 — (A) gain control, (B) accuracy, (C) similarity.

    The concentration-invariance figure. Odors are presented at a range of
    relative concentrations (the network is trained only at c=1.0) and three
    properties are tracked:
      A. Gain control: stage-wise activity (OR/PN/KC) normalized to its own
         c=1.0 value, against a Hill dose-response reference. Shows that LN
         inhibition (OR->PN) and APL inhibition (PN->KC) progressively flatten
         the concentration-dependence of activity.
      B. Classification accuracy vs concentration, for the trained linear
         decoder and a 0-parameter KC-centroid (nearest-centroid) baseline.
      C. Cosine similarity of each stage's representation back to its own
         c=1.0 baseline, i.e. how concentration-stable the code is.

    Parameters
    ----------
    conc_data : dict
        Maps str(concentration) -> dict of metric lists (one entry per seed):
          'mean_ors','mean_pns','mean_kcs'        -> mean activities,
          'decoder_accs','kc_centroid_accs'       -> accuracies (fractions),
          'or_sims','pn_sims','kc_sims'            -> similarity-to-baseline.
    concentrations : sequence[float] or None
        Relative concentrations to plot on the (log) x-axis. Defaults to
        [0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0] when None.
    hill_ec50 : float
        EC50 of the reference Hill transfer overlaid in panel A.
    output_path : str or None
        If given, saved as PNG (300 dpi) and, when the path ends in .png, also
        as a sibling PDF (300 dpi).
    show : bool
        If False, close after saving.

    Returns
    -------
    matplotlib.figure.Figure
        Three-panel concentration-invariance figure.
    """
    # Default concentration sweep (relative units; 1.0 == training concentration).
    if concentrations is None:
        concentrations = [0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Stable per-stage colors reused across panels: OR (red), PN (blue),
    # KC (green), decoder (near-black).
    c_or = '#922b21'
    c_pn = '#1a5276'
    c_kc = '#196f3d'
    c_dec = '#1b2631'

    label_fs, tick_fs, letter_fs, legend_fs = 14, 12, 20, 12

    def _format_conc_ax(ax):
        """Apply the shared concentration-axis styling to one subplot.

        Sets a log x-scale (concentrations span ~2 decades), places explicit
        ticks at the sampled concentrations, and removes the top/right spines.
        Mutates the passed Axes in place; returns nothing.
        """
        ax.set_xscale('log')
        ax.set_xticks(concentrations)
        ax.set_xticklabels([f'{c}' for c in concentrations])
        ax.tick_params(labelsize=tick_fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # --- A. Gain Control ---
    ax = axes[0]
    # Reference activities at the training concentration (c=1.0) per stage.
    ref_or = np.mean(conc_data['1.0']['mean_ors'])
    ref_pn = np.mean(conc_data['1.0']['mean_pns'])
    ref_kc = np.mean(conc_data['1.0']['mean_kcs'])
    # Mean activity at each concentration, normalized to the c=1.0 reference.
    # max(ref, 1e-8) guards against divide-by-zero when a stage is silent.
    or_norm = [np.mean(conc_data[str(c)]['mean_ors']) / max(ref_or, 1e-8) for c in concentrations]
    pn_norm = [np.mean(conc_data[str(c)]['mean_pns']) / max(ref_pn, 1e-8) for c in concentrations]
    kc_norm = [np.mean(conc_data[str(c)]['mean_kcs']) / max(ref_kc, 1e-8) for c in concentrations]
    # Across-seed SD of the normalized activity, for error bars.
    or_norm_sd = [np.std([v / max(ref_or, 1e-8) for v in conc_data[str(c)]['mean_ors']]) for c in concentrations]
    pn_norm_sd = [np.std([v / max(ref_pn, 1e-8) for v in conc_data[str(c)]['mean_pns']]) for c in concentrations]
    kc_norm_sd = [np.std([v / max(ref_kc, 1e-8) for v in conc_data[str(c)]['mean_kcs']]) for c in concentrations]
    # Pure receptor-level Hill transfer reference (no circuit), normalized so
    # it equals 1.0 at c=1.0 (see hill_effective_concentration).
    eff_concs = [hill_effective_concentration(c, ec50=hill_ec50) for c in concentrations]

    # Shared errorbar styling for the three stage curves.
    _eb = dict(linewidth=2, markersize=7, capsize=4, elinewidth=1.2, capthick=1.2)
    ax.errorbar(concentrations, or_norm, yerr=or_norm_sd, fmt='s-', color=c_or, label='OR input', **_eb)
    ax.errorbar(concentrations, pn_norm, yerr=pn_norm_sd, fmt='D-', color=c_pn,
                label='PN (after LN inhibition)', **_eb)
    ax.errorbar(concentrations, kc_norm, yerr=kc_norm_sd, fmt='o-', color=c_kc,
                label='KC (after APL inhibition)', **_eb)
    ax.plot(concentrations, eff_concs, '--', color='gray', alpha=0.5, linewidth=1.5,
            label=f'Hill transfer (EC50={hill_ec50})')
    # Vertical marker at the training concentration.
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Concentration (relative)', fontsize=label_fs)
    ax.set_ylabel('Activity (normalized to c=1.0)', fontsize=label_fs)
    ax.legend(fontsize=legend_fs, loc='upper left')
    _format_conc_ax(ax)
    # Panel letter "A".
    ax.text(-0.1, 1.02, 'A', transform=ax.transAxes, fontsize=letter_fs,
            fontweight='bold', va='bottom', ha='left')

    # --- B. Classification Accuracy ---
    ax = axes[1]
    # Trained-decoder accuracy (mean/SD over seeds), as percentages.
    dec_mean = [np.mean(conc_data[str(c)]['decoder_accs']) * 100 for c in concentrations]
    dec_std = [np.std(conc_data[str(c)]['decoder_accs']) * 100 for c in concentrations]
    # 0-parameter KC-centroid baseline accuracy for comparison.
    kc_cent_mean = [np.mean(conc_data[str(c)]['kc_centroid_accs']) * 100 for c in concentrations]
    kc_cent_std = [np.std(conc_data[str(c)]['kc_centroid_accs']) * 100 for c in concentrations]

    ax.errorbar(concentrations, dec_mean, yerr=dec_std, fmt='o-', color=c_dec,
                label='Trained decoder', **_eb)
    ax.errorbar(concentrations, kc_cent_mean, yerr=kc_cent_std, fmt='s--', color=c_kc,
                linewidth=1.5, markersize=6, capsize=4, elinewidth=1.2, capthick=1.2,
                label='KC centroid (0 params)')
    # Chance accuracy for the 28-way odor classification (100/28 %); matches the body text's 3.6%.
    ax.axhline(y=100/28, color='red', linestyle=':', alpha=0.4, label=f'Chance ({100/28:.1f}%)')
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, label='Training conc.')
    ax.set_xlabel('Concentration (relative)', fontsize=label_fs)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=label_fs)
    ax.legend(fontsize=legend_fs, loc='upper left')
    _format_conc_ax(ax)
    # Panel letter "B".
    ax.text(-0.1, 1.02, 'B', transform=ax.transAxes, fontsize=letter_fs,
            fontweight='bold', va='bottom', ha='left')

    # --- C. Representation Similarity ---
    ax = axes[2]
    # Mean cosine similarity of each stage's code to its own c=1.0 baseline.
    or_s = [np.mean(conc_data[str(c)]['or_sims']) for c in concentrations]
    pn_s = [np.mean(conc_data[str(c)]['pn_sims']) for c in concentrations]
    kc_s = [np.mean(conc_data[str(c)]['kc_sims']) for c in concentrations]
    # Across-seed SD for error bars.
    or_s_sd = [np.std(conc_data[str(c)]['or_sims']) for c in concentrations]
    pn_s_sd = [np.std(conc_data[str(c)]['pn_sims']) for c in concentrations]
    kc_s_sd = [np.std(conc_data[str(c)]['kc_sims']) for c in concentrations]

    ax.errorbar(concentrations, or_s, yerr=or_s_sd, fmt='s-', color=c_or, label='OR', **_eb)
    ax.errorbar(concentrations, pn_s, yerr=pn_s_sd, fmt='D-', color=c_pn, label='PN (after AL)', **_eb)
    ax.errorbar(concentrations, kc_s, yerr=kc_s_sd, fmt='o-', color=c_kc, label='KC (after MB)', **_eb)
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, label='Training conc.')
    ax.set_xlabel('Concentration (relative)', fontsize=label_fs)
    ax.set_ylabel('Cosine Similarity to Baseline (c=1.0)', fontsize=label_fs)
    ax.legend(fontsize=legend_fs)
    # Similarity is in [0, 1]; small headroom above 1.0.
    ax.set_ylim(0, 1.05)
    _format_conc_ax(ax)
    # Panel letter "C".
    ax.text(-0.1, 1.02, 'C', transform=ax.transAxes, fontsize=letter_fs,
            fontweight='bold', va='bottom', ha='left')

    plt.tight_layout()
    if output_path:
        # Publication-quality raster (300 dpi) plus a sibling vector PDF.
        fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
        pdf_path = str(output_path).replace('.png', '.pdf')
        if pdf_path != str(output_path):
            fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig

