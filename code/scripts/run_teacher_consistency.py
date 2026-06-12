"""
run_teacher_consistency.py

C5: Teacher Parameter Consistency Analysis (Reviewer 3, Point 5).
Post-hoc analysis — no training. Loads the 5 cached teacher models and
5 trained student (spiking) models, then computes:

1. Teacher-teacher parameter correlations across 5 seeds
   → How reproducible is the rate-based solution?
2. Teacher-to-student parameter drift during Phase 2 fine-tuning
   → How much do shared parameters change when converting to spiking?
3. Student-student parameter correlations across 5 seeds
   → Does the connectome constrain the spiking solution more or less?

Key shared parameters between teacher and student:
  - or_to_orn.or_gains (21 OR receptor gains) — copied then fine-tuned
  - decoder.weight (28×144 readout matrix) — copied then fine-tuned
  - decoder.bias (28 readout biases) — copied then fine-tuned
  - kc_layer.apl.apl_gain (scalar APL gain) — copied with 4× boost, then fine-tuned

Student-only parameters (analyze consistency across seeds):
  - kc_layer.kc_neurons.v_th (144 per-KC thresholds) — init from scratch
  - antennal_lobe.log_g_gap_* (3 gap conductances) — init from log(1e-10)
  - antennal_lobe.*.log_strength (synaptic pathway strengths)
  - kc_layer.kc_neurons.log_g_soma (somatic conductance)

Cross-seed scalar CVs are computed in PHYSICAL units (conductances/strengths/
time constants are stored as log(theta); we exponentiate before taking CV) so
the student CVs are unit-invariant and directly comparable to the teacher's
real-valued scalars. (Computing CV on log(theta) is invalid and unit-dependent.)

Results saved to: results/teacher_consistency_r2/
  teacher_consistency_results.json
  c5_convergence_results.json

Notebook section: Section B — C5 (teacher/student consistency figure and table).

----------------------------------------------------------------------------
PIPELINE CONTEXT
----------------------------------------------------------------------------
This file sits at the very END of the modelling pipeline. The biology
(OR responses -> ORN -> LN -> PN -> KC <- APL -> linear decoder over 28 odors)
is trained elsewhere in two phases:
  Phase 1: a *rate* ("teacher") model is fit per seed (ANN-style, fast).
  Phase 2: a *spiking* ("student") model is initialised from the teacher's
           shared parameters and fine-tuned with full LIF / two-compartment
           dynamics (the ANN-to-SNN transfer).
This script never instantiates either network. It loads the saved
``state_dict``-style checkpoints (plain tensors keyed by parameter path),
pulls out the learned biological parameters, and quantifies how *reproducible*
(across the 5 random seeds) and how *transferable* (teacher -> student) those
learned solutions are. It is pure NumPy/Torch tensor bookkeeping — no
forward pass, no gradient, no simulation.
"""
import sys
from pathlib import Path
import json
import numpy as np
import torch

# Force UTF-8 line-buffered stdout/stderr so the progress prints below (which
# contain non-ASCII glyphs like '±', '→', '×') render correctly and stream live
# when this script is launched as a subprocess / piped to a log file.
if hasattr(sys.stdout, 'reconfigure'):  # absent under Jupyter's OutStream
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):  # absent under Jupyter's OutStream
    sys.stderr.reconfigure(encoding='utf-8')

# ============================================================================
# CONFIG
# ============================================================================
# Directory holding the per-seed rate-based TEACHER checkpoints
# (teacher_seed{42..46}.pt), written by the canonical run (run_training.py
# train_single_model) ALONGSIDE the matching student, in the SAME run. So
# teacher_seed{s} is exactly the rate teacher that initialized student
# model_seed{s} -- the drift analysis uses the real teacher<->student pair, not a
# cross-run proxy. Same directory as STUDENT_DIR below.
TEACHER_DIR = Path(__file__).parent.parent.parent / 'results' / 'all_connections_nonad_canonical'
# Directory holding the 5 trained spiking STUDENT checkpoints
# (model_seed{42..46}.pt) — the canonical "all connections enabled" run.
STUDENT_DIR = Path(__file__).parent.parent.parent / 'results' / 'all_connections_nonad_canonical'
# Where the JSON consistency report is written.
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'results' / 'teacher_consistency_r2'

# The 5 random seeds used for both teacher and student training; consistency is
# measured as agreement of learned parameters across these independent runs.
SEEDS = [42, 43, 44, 45, 46]
# When a teacher is converted to a student, its scalar APL gain is multiplied by
# this factor at initialisation. The graded APL provides divisive (shunting)
# inhibition onto KCs; spiking KCs need stronger inhibition than the rate model,
# hence the 4x boost. Used below to align teacher vs student APL gain in drift.
APL_BOOST = 4.0  # boost factor applied when copying teacher APL gain to student


# ============================================================================
# LOAD MODELS
# ============================================================================
def load_teachers():
    """Load all 5 teacher state dicts.

    Reads ``teacher_seed{seed}.pt`` for every seed in ``SEEDS`` from
    ``TEACHER_DIR``. Each checkpoint is loaded onto CPU (``map_location='cpu'``)
    with ``weights_only=False`` (the file may contain full pickled objects, not
    just a bare tensor dict). Missing files are skipped with a warning rather
    than raising, so the analysis can still run on a partial seed set.

    Returns:
        dict[int, dict]: seed -> teacher state dict (parameter-path keys mapping
        to CPU tensors). Side effect: prints load/warning lines to stdout.
    """
    teachers = {}
    for seed in SEEDS:
        path = TEACHER_DIR / f'teacher_seed{seed}.pt'
        if not path.exists():
            # Tolerate a missing seed: warn and continue with whatever loads.
            print(f"  WARNING: teacher seed {seed} not found at {path}")
            continue
        # weights_only=False: checkpoint may be a pickled object graph, not a
        # bare tensor dict. map_location='cpu': analysis is CPU-only, no GPU.
        teachers[seed] = torch.load(path, weights_only=False, map_location='cpu')
        print(f"  Loaded teacher seed {seed}")
    return teachers


def load_students():
    """Load all 5 student state dicts.

    Mirror of ``load_teachers`` but for the spiking student checkpoints
    ``model_seed{seed}.pt`` in ``STUDENT_DIR``. Same CPU / full-pickle load
    semantics and same tolerant skip-on-missing behaviour.

    Returns:
        dict[int, dict]: seed -> student state dict (CPU tensors keyed by
        parameter path). Side effect: prints to stdout.
    """
    students = {}
    for seed in SEEDS:
        path = STUDENT_DIR / f'model_seed{seed}.pt'
        if not path.exists():
            # Tolerate a missing seed: warn and continue.
            print(f"  WARNING: student seed {seed} not found at {path}")
            continue
        students[seed] = torch.load(path, weights_only=False, map_location='cpu')
        print(f"  Loaded student seed {seed}")
    return students


# ============================================================================
# PARAMETER EXTRACTION
# ============================================================================
def extract_teacher_params(state):
    """Extract key parameter vectors from teacher state dict.

    Teacher uses rate-based model (ConnectomeConstrainedModel), which has
    different param names than the spiking student.

    Args:
        state (dict): one teacher state dict (CPU tensors keyed by the
            rate-model's parameter paths).

    Returns:
        dict: named NumPy arrays / Python scalars for the parameters of
        interest. Shapes/units are annotated inline below. The teacher's
        ``strengths``/``apl_gain`` are stored as plain real values (NOT in
        log space), so no exponentiation is needed here — unlike the student.
    """
    return {
        'or_gains': state['or_to_orn.or_gains'].numpy(),                  # (21,) per-OR receptor gains (dimensionless)
        'decoder_weight': state['decoder.weight'].numpy(),                 # (28, 144) odor x KC linear readout matrix
        'decoder_bias': state['decoder.bias'].numpy(),                     # (28,) per-odor readout bias
        'apl_gain': state['kc_layer.apl.apl_gain'].numpy().item(),        # scalar APL divisive-inhibition gain (real-valued)
        'kc_threshold': state['kc_layer.kc_threshold'].numpy(),           # (144,) per-KC firing threshold (rate-model form)
        'orn_pn_strength': state['antennal_lobe.orn_pn.strengths'].numpy().item(),  # scalar ORN->PN synaptic strength (real-valued)
        'ln_pn_strength': state['antennal_lobe.ln_pn.strengths'].numpy().item(),    # scalar LN->PN synaptic strength (real-valued)
    }


def extract_student_params(state):
    """Extract key parameter vectors from student state dict.

    Student uses spiking model (SpikingConnectomeConstrainedModel) with
    different naming: log_strength instead of strengths, v_th instead of
    kc_threshold, etc.

    IMPORTANT (cross-seed CV must be computed in physical units): the spiking
    model stores conductances, synaptic strengths and time constants as
    log(theta) parameters. A coefficient of variation taken on log(theta)
    is invalid -- log(theta) is interval-scale, so its mean (and hence
    std/|mean|) shifts when the unit of theta changes (nS<->S, ms<->s), and
    it is not comparable to the teacher's real-valued scalars. We therefore
    exponentiate every log parameter back to physical units (S, A, s) here,
    so scalar_consistency() reports a valid, unit-invariant relative spread
    on the same footing as the teacher (orn_pn/ln_pn strengths, apl_gain).

    The model parameterises strictly-positive biophysical quantities in log
    space precisely so that gradient descent stays positive and spans several
    orders of magnitude; this function inverts that storage with exp().

    Args:
        state (dict): one student state dict (CPU tensors keyed by the spiking
            model's parameter paths).

    Returns:
        dict: named NumPy arrays (vector params) and Python floats (scalar
        params in physical SI units). Shapes/units annotated inline below.
    """
    # Helper: pull a stored scalar log-parameter and map it back to physical
    # units. ``exp(log_theta) = theta`` undoes the log-space storage; the
    # downstream CV/std then live in real units (S, A, s) and are unit-invariant.
    exp = lambda k: float(np.exp(state[k].numpy().item()))
    return {
        'or_gains': state['or_to_orn.or_gains'].numpy(),                  # (21,) per-OR receptor gains (shared w/ teacher)
        'decoder_weight': state['decoder.weight'].numpy(),                 # (28, 144) odor x KC readout (shared w/ teacher)
        'decoder_bias': state['decoder.bias'].numpy(),                     # (28,) per-odor readout bias (shared w/ teacher)
        'apl_gain': state['kc_layer.apl.apl_gain'].numpy().item(),        # scalar, real-valued (shared w/ teacher)
        'kc_v_th': state['kc_layer.kc_neurons.v_th'].numpy(),             # (144,) per-KC LIF threshold (V), student-only
        'ln_v_th': state['antennal_lobe.ln_neurons.v_th'].numpy(),        # (108,) per-LN LIF threshold (V), student-only
        'orn_v_th': state['antennal_lobe.orn_neurons.v_th'].numpy(),      # (42,) per-ORN LIF threshold (V), student-only
        'pn_v_th': state['antennal_lobe.pn_neurons.v_th'].numpy(),        # (72,) per-PN LIF threshold (V), student-only
        # Scalar biophysical params -> physical units (exp of stored log_theta)
        'g_gap_ln': exp('antennal_lobe.log_g_gap_ln'),                    # S — LN-LN gap-junction conductance
        'g_gap_pn': exp('antennal_lobe.log_g_gap_pn'),                    # S — PN-PN (sister) gap-junction conductance
        'g_gap_eln_pn': exp('antennal_lobe.log_g_gap_eln_pn'),           # S — eLN-PN gap-junction conductance
        'g_soma': exp('kc_layer.kc_neurons.log_g_soma'),                 # S — KC dendrite<->soma coupling conductance
        'orn_pn_strength': exp('antennal_lobe.orn_pn.log_strength'),     # A — ORN->PN synaptic strength
        'ln_pn_strength': exp('antennal_lobe.ln_pn.log_strength'),       # A — LN->PN (inhibitory) synaptic strength
        'ln_pn_excit_strength': exp('antennal_lobe.ln_pn_excit.log_strength'),  # A — excitatory LN->PN synaptic strength
        'kc_kc_aa_strength': exp('kc_layer.kc_kc_aa.log_strength'),      # A — KC->KC axo-axonal synaptic strength
        'pn_kc_strength': exp('kc_layer.pn_kc.log_strength'),            # A — PN->KC synaptic strength
        'tau_apl': exp('kc_layer.apl.log_tau_apl'),                      # s — APL membrane/inhibition time constant
    }


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================
def pairwise_correlations(param_dict_by_seed, param_name):
    """Compute all pairwise Pearson correlations for a vector parameter.

    C5: Measures how similar the same parameter is across different seeds.
    High correlation = connectome strongly constrains this parameter.

    For an N-seed set there are N*(N-1)/2 unordered seed pairs; we correlate
    the (flattened) parameter vector of each seed against every other seed and
    summarise the off-diagonal (upper-triangular) correlations.

    Args:
        param_dict_by_seed (dict[int, dict]): seed -> extracted-params dict.
        param_name (str): key naming the vector parameter to correlate
            (e.g. 'decoder_weight'); it is ``.flatten()``-ed before correlation
            so 2-D matrices like the 28x144 decoder are treated as 1-D vectors.

    Returns:
        dict: summary stats over the unique seed pairs — mean/std/min/max
        Pearson r and the number of pairs.
    """
    seeds = sorted(param_dict_by_seed.keys())
    # Flatten each seed's parameter to a 1-D vector so 2-D matrices (e.g. the
    # 28x144 decoder) can be Pearson-correlated element-for-element.
    vectors = {s: param_dict_by_seed[s][param_name].flatten() for s in seeds}

    n = len(seeds)
    # Symmetric correlation matrix; diagonal stays 1 (a seed vs itself).
    corr_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Pearson r between seed i and seed j; np.corrcoef returns the 2x2
            # correlation matrix, [0,1] is the cross-correlation entry.
            r = np.corrcoef(vectors[seeds[i]], vectors[seeds[j]])[0, 1]
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r  # mirror to keep the matrix symmetric

    # Mean of upper triangle
    # Upper-triangular indices (k=1 excludes the diagonal) select exactly the
    # unique seed pairs, avoiding double-counting and the trivial r=1 diagonal.
    triu_idx = np.triu_indices(n, k=1)
    mean_corr = float(np.mean(corr_matrix[triu_idx]))
    std_corr = float(np.std(corr_matrix[triu_idx]))

    return {
        'mean_correlation': mean_corr,
        'std_correlation': std_corr,
        'min_correlation': float(np.min(corr_matrix[triu_idx])),
        'max_correlation': float(np.max(corr_matrix[triu_idx])),
        'n_pairs': int(len(triu_idx[0])),
    }


def scalar_consistency(param_dict_by_seed, param_name):
    """Compute mean and CV for a scalar parameter across seeds.

    C5: For scalar parameters (strengths, gains), correlation isn't meaningful.
    Instead, report coefficient of variation (CV = std/mean) in PHYSICAL units
    (extract_*_params already exponentiates the student's log parameters), so
    teacher and student CVs are on the same, unit-invariant footing.

    Also reports std_log = std(log theta) for strictly-positive scalars: this
    is the unit-invariant "geometric" relative spread and, for small spreads,
    std_log ~= the real-space CV. It is reported alongside CV for transparency.

    Args:
        param_dict_by_seed (dict[int, dict]): seed -> extracted-params dict.
        param_name (str): key naming the scalar parameter (already in physical
            units for the student).

    Returns:
        dict: mean, std, cv (real-space relative spread), std_log (geometric
        spread; NaN if any value is non-positive), and the per-seed values.
    """
    seeds = sorted(param_dict_by_seed.keys())
    # Collect the scalar value per seed into one array for population stats.
    values = np.array([float(param_dict_by_seed[s][param_name]) for s in seeds])
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    # CV = std/|mean|, the unit-invariant relative spread. Guard against a
    # near-zero mean (e.g. a parameter centred on 0) to avoid blow-up -> NaN.
    cv = std_val / abs(mean_val) if abs(mean_val) > 1e-30 else float('nan')
    # std of log(values) = geometric/relative spread; only defined for strictly
    # positive values (log undefined otherwise) -> NaN guard for sign-changing
    # quantities like apl_gain that may go negative.
    std_log = float(np.std(np.log(values))) if np.all(values > 0) else float('nan')

    return {
        'mean': mean_val,
        'std': std_val,
        'cv': cv,            # real-space (physical-unit) coefficient of variation
        'std_log': std_log,  # unit-invariant geometric spread (~ cv for small spreads)
        'values': {s: float(v) for s, v in zip(seeds, values)},
    }


# ============================================================================
# DRIFT ANALYSIS
# ============================================================================
def compute_drift(teacher_params, student_params, seeds):
    """Compute teacher-to-student parameter drift for shared parameters.

    C5: For each seed, measures how much shared parameters change during
    Phase 2 (spiking fine-tuning). Large drift = spiking dynamics reshape
    the solution significantly. Small drift = rate-based solution transfers.

    Note: APL gain is boosted 4x when copied to student, so we compare
    teacher_gain * 4 vs student_gain.

    Drift is measured per shared parameter with three complementary metrics:
      - correlation: does the *pattern* (relative profile) survive fine-tuning?
      - norm_ratio: did the *overall magnitude* grow/shrink (student/teacher)?
      - mean_rel_change: average per-element relative change (magnitude AND sign).

    Args:
        teacher_params (dict[int, dict]): seed -> teacher extracted params.
        student_params (dict[int, dict]): seed -> student extracted params.
        seeds (list[int]): seeds present in BOTH model sets.

    Returns:
        dict: param_name -> {per_seed metrics, averages, qualitative
        interpretation} for the vector params, plus an 'apl_gain' entry handling
        the 4x boost for the scalar APL gain.
    """
    results = {}

    for param_name in ['or_gains', 'decoder_weight', 'decoder_bias']:
        drifts = []
        for s in seeds:
            # Flatten both to compare element-for-element (e.g. the 28x144 decoder).
            t_vec = teacher_params[s][param_name].flatten()
            s_vec = student_params[s][param_name].flatten()

            # Correlation: how much does the pattern change?
            # r near 1 -> the relative shape of the parameter is preserved even
            # if its absolute scale shifted.
            corr = float(np.corrcoef(t_vec, s_vec)[0, 1])

            # Relative magnitude change
            # L2 norms capture overall scale; the ratio >1 means fine-tuning
            # grew the parameter, <1 means it shrank.
            t_norm = float(np.linalg.norm(t_vec))
            s_norm = float(np.linalg.norm(s_vec))
            norm_ratio = s_norm / t_norm if t_norm > 1e-10 else float('nan')  # guard zero teacher norm

            # Mean absolute relative change per element
            # Denominator floored at 1e-8 so near-zero teacher entries don't
            # produce divide-by-zero / exploding relative changes.
            denom = np.maximum(np.abs(t_vec), 1e-8)
            rel_change = float(np.mean(np.abs(s_vec - t_vec) / denom))

            drifts.append({
                'seed': s,
                'correlation': corr,
                'norm_ratio': norm_ratio,
                'mean_rel_change': rel_change,
            })

        # Average the per-seed pattern-correlation and relative change.
        avg_corr = float(np.mean([d['correlation'] for d in drifts]))
        avg_rel = float(np.mean([d['mean_rel_change'] for d in drifts]))

        results[param_name] = {
            'per_seed': drifts,
            'avg_correlation': avg_corr,
            'avg_rel_change': avg_rel,
            # Qualitative label from the average teacher->student correlation:
            # >0.9 the rate solution transfers almost intact; >0.5 partial; else
            # the spiking dynamics substantially reshape this parameter.
            'interpretation': 'preserved' if avg_corr > 0.9 else
                            'partially preserved' if avg_corr > 0.5 else
                            'substantially reshaped',
        }

    # APL gain (scalar, boosted 4x at init)
    apl_drifts = []
    for s in seeds:
        # Align units: the student was initialised from teacher_gain * APL_BOOST,
        # so the meaningful drift is student vs that boosted reference.
        t_gain = teacher_params[s]['apl_gain'] * APL_BOOST
        s_gain = student_params[s]['apl_gain']
        rel_change = abs(s_gain - t_gain) / abs(t_gain) if abs(t_gain) > 1e-8 else float('nan')  # guard zero
        apl_drifts.append({
            'seed': s,
            'teacher_gain_x4': float(t_gain),
            'student_gain': float(s_gain),
            'relative_change': float(rel_change),
        })
    avg_apl_rel = float(np.mean([d['relative_change'] for d in apl_drifts]))
    results['apl_gain'] = {
        'per_seed': apl_drifts,
        'avg_rel_change': avg_apl_rel,
        'note': f'Teacher gain multiplied by APL_BOOST={APL_BOOST} at student init',
    }

    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Run the full C5 consistency/drift analysis and write the JSON report.

    Orchestrates the four reported sections:
      1. Teacher-teacher consistency (vector correlations + scalar CVs).
      2. Student-student consistency (same, plus the student-only biophysics).
      3. Teacher-to-student drift on the shared parameters.
      4. Side-by-side teacher vs student consistency comparison.

    Side effects: creates ``OUTPUT_DIR``, prints a formatted report to stdout,
    and serialises all results to ``teacher_consistency_results.json`` (with a
    NumPy-aware JSON encoder). Returns None.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("C5: TEACHER PARAMETER CONSISTENCY ANALYSIS")
    print("="*70)

    # Load models
    print("\nLoading teacher models...")
    teachers = load_teachers()
    print("\nLoading student models...")
    students = load_students()

    # Only analyse seeds that have BOTH a teacher and a student checkpoint, so
    # the drift comparison is well-defined for every seed used.
    available_seeds = sorted(set(teachers.keys()) & set(students.keys()))
    print(f"\nAvailable seeds: {available_seeds}")

    # Extract parameters
    # Pull the parameters of interest once per seed into compact dicts; all
    # downstream analysis operates on these (the raw state dicts aren't reused).
    teacher_params = {s: extract_teacher_params(teachers[s]) for s in available_seeds}
    student_params = {s: extract_student_params(students[s]) for s in available_seeds}

    # ====================================================================
    # 1. TEACHER-TEACHER CONSISTENCY
    # ====================================================================
    print(f"\n{'='*70}")
    print("1. TEACHER-TEACHER PARAMETER CONSISTENCY")
    print(f"{'='*70}")
    print("   (High correlation = connectome constrains rate-based solution)")

    # Vector teacher params: summarised by cross-seed pairwise Pearson r.
    teacher_vector_params = ['or_gains', 'decoder_weight', 'decoder_bias', 'kc_threshold']
    teacher_vector_results = {}
    for param in teacher_vector_params:
        result = pairwise_correlations(teacher_params, param)
        teacher_vector_results[param] = result
        print(f"\n  {param}:")
        print(f"    Mean pairwise r = {result['mean_correlation']:.4f} +/- {result['std_correlation']:.4f}")
        print(f"    Range: [{result['min_correlation']:.4f}, {result['max_correlation']:.4f}]")

    # Scalar teacher params: real-valued already, so summarised by mean + CV.
    teacher_scalar_params = ['apl_gain', 'orn_pn_strength', 'ln_pn_strength']
    teacher_scalar_results = {}
    for param in teacher_scalar_params:
        result = scalar_consistency(teacher_params, param)
        teacher_scalar_results[param] = result
        print(f"\n  {param}:")
        print(f"    Mean = {result['mean']:.6f}, CV = {result['cv']:.4f}")
        print(f"    Values: {result['values']}")

    # ====================================================================
    # 2. STUDENT-STUDENT CONSISTENCY
    # ====================================================================
    print(f"\n{'='*70}")
    print("2. STUDENT-STUDENT PARAMETER CONSISTENCY")
    print(f"{'='*70}")
    print("   (Compare to teacher consistency — more or less constrained?)")

    # Vector student params: shared readout params plus the per-neuron LIF
    # thresholds (v_th) for every spiking population (KC/LN/ORN/PN).
    student_vector_params = ['or_gains', 'decoder_weight', 'decoder_bias', 'kc_v_th',
                             'ln_v_th', 'orn_v_th', 'pn_v_th']
    student_vector_results = {}
    for param in student_vector_params:
        result = pairwise_correlations(student_params, param)
        student_vector_results[param] = result
        print(f"\n  {param}:")
        print(f"    Mean pairwise r = {result['mean_correlation']:.4f} +/- {result['std_correlation']:.4f}")

    # Physical-unit scalar params (extract_student_params exponentiates log_theta -> theta)
    # Conductances (S), synaptic strengths (A) and the APL time constant (s);
    # all already in physical units, so CV is a valid cross-seed spread here.
    student_scalar_params = ['apl_gain', 'g_gap_ln', 'g_gap_pn', 'g_gap_eln_pn',
                             'g_soma', 'orn_pn_strength', 'ln_pn_strength',
                             'ln_pn_excit_strength', 'kc_kc_aa_strength', 'pn_kc_strength', 'tau_apl']
    student_scalar_results = {}
    for param in student_scalar_params:
        result = scalar_consistency(student_params, param)
        student_scalar_results[param] = result
        print(f"\n  {param}:")
        print(f"    Mean = {result['mean']:.6g}, CV = {result['cv']*100:.2f}%  (std_log = {result['std_log']:.4f})")

    # Summary: teacher vs student scalar-CV, both in physical units (apples-to-apples)
    # Aggregate the per-parameter CVs into one teacher number and one student
    # number so the report can state whether spiking fine-tuning tightened or
    # loosened the scalar solution overall.
    t_cvs = [teacher_scalar_results[p]['cv'] for p in teacher_scalar_params]
    s_cvs = [student_scalar_results[p]['cv'] for p in student_scalar_params]
    print(f"\n  Scalar CV (physical units): teacher mean {np.mean(t_cvs)*100:.1f}% "
          f"(range {min(t_cvs)*100:.1f}-{max(t_cvs)*100:.1f}%)  ->  "
          f"student mean {np.mean(s_cvs)*100:.1f}% (range {min(s_cvs)*100:.1f}-{max(s_cvs)*100:.1f}%)")
    # apl_gain is the one scalar stored real-valued in BOTH models, so its
    # teacher->student CV is the cleanest like-for-like comparison.
    print(f"  apl_gain (real-valued in BOTH): teacher {teacher_scalar_results['apl_gain']['cv']*100:.1f}% "
          f"-> student {student_scalar_results['apl_gain']['cv']*100:.1f}%")

    # ====================================================================
    # 3. TEACHER-TO-STUDENT DRIFT
    # ====================================================================
    print(f"\n{'='*70}")
    print("3. TEACHER-TO-STUDENT PARAMETER DRIFT (Phase 2)")
    print(f"{'='*70}")
    print("   (Low drift = rate-based solution transfers to spiking)")

    drift_results = compute_drift(teacher_params, student_params, available_seeds)

    # Two print branches: vector params expose 'avg_correlation', whereas the
    # scalar APL-gain entry exposes only 'avg_rel_change' + per-seed pairs.
    for param, result in drift_results.items():
        if 'avg_correlation' in result:
            print(f"\n  {param}:")
            print(f"    Avg T-S correlation: {result['avg_correlation']:.4f}")
            print(f"    Avg relative change: {result['avg_rel_change']:.4f}")
            print(f"    Interpretation: {result['interpretation']}")
        elif 'avg_rel_change' in result:
            print(f"\n  {param}:")
            print(f"    Avg relative change: {result['avg_rel_change']:.4f}")
            for d in result['per_seed']:
                print(f"      seed {d['seed']}: teacher*4={d['teacher_gain_x4']:.4f} -> student={d['student_gain']:.4f}")

    # ====================================================================
    # 4. COMPARISON: Teacher vs Student consistency
    # ====================================================================
    print(f"\n{'='*70}")
    print("4. COMPARISON: DOES SPIKING FINE-TUNING INCREASE OR DECREASE CONSISTENCY?")
    print(f"{'='*70}")

    # Only the parameters that exist (and were correlated) in both models can be
    # compared directly. kc_threshold/v_th differ in form, so they're excluded.
    shared_params = ['or_gains', 'decoder_weight', 'decoder_bias']
    print(f"\n  {'Parameter':<25} {'Teacher r':>12} {'Student r':>12} {'Delta':>8} {'Change':>15}")
    print(f"  {'-'*72}")
    comparison_results = {}
    for param in shared_params:
        t_r = teacher_vector_results[param]['mean_correlation']
        s_r = student_vector_results[param]['mean_correlation']
        delta = s_r - t_r  # positive -> student more consistent than teacher
        # +/-0.01 dead-band so tiny numerical wobble is reported as STABLE.
        change = 'MORE consistent' if delta > 0.01 else 'LESS consistent' if delta < -0.01 else 'STABLE'
        comparison_results[param] = {'teacher_r': t_r, 'student_r': s_r, 'delta': delta, 'change': change}
        print(f"  {param:<25} {t_r:>12.4f} {s_r:>12.4f} {delta:>+8.4f} {change:>15}")

    # ====================================================================
    # SAVE RESULTS
    # ====================================================================
    # Bundle every section into one nested dict mirroring the printed report.
    full_results = {
        'seeds': available_seeds,
        'teacher_teacher': {
            'vector_params': teacher_vector_results,
            'scalar_params': teacher_scalar_results,
        },
        'student_student': {
            'vector_params': student_vector_results,
            'scalar_params': student_scalar_results,
        },
        'teacher_to_student_drift': drift_results,
        'comparison': comparison_results,
    }

    results_path = OUTPUT_DIR / 'teacher_consistency_results.json'
    # Custom JSON encoder: NumPy scalar/array types aren't natively
    # serialisable, so coerce them to native Python (int/float/bool/list);
    # anything else falls back to its string form.
    def _json_default(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return str(obj)

    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2, default=_json_default)
    print(f"\nResults saved: {results_path}")


if __name__ == '__main__':
    main()
