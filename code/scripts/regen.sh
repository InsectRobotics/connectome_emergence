#!/bin/bash
# =============================================================================
# Regenerate experiment results. Two modes:
#   bash code/scripts/regen.sh fast    # eval-only analyses (~50 min): teacher, posthoc, honegger, mixtures
#   bash code/scripts/regen.sh heavy   # ALL retraining from scratch (~18-20 h); run overnight
#
# Heavy mode SKIPS conditions whose output already exists; to force a true
# from-scratch run, delete the result dirs first (see TRUE_FROM_SCRATCH below).
# =============================================================================
set -e
MODE="${1:-fast}"
# cd into code/ so the `python scripts/run_*.py` calls below resolve. Each script
# self-bootstraps sys.path (adds code/ to it) and resolves data/ + results/ relative
# to its own __file__, so no PYTHONPATH or machine-specific paths are needed.
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE_DIR"
SEEDS="42 43 44 45 46"

if [ "$MODE" = "fast" ]; then
  echo "=== [1/4] teacher_consistency (~1min) ===";  python scripts/run_teacher_consistency.py 2>&1 | tail -3
  echo "=== [2/4] posthoc_ablation (~15min) ===";     python scripts/run_ablation.py posthoc 2>&1 | tail -3
  echo "=== [3/4] honegger_metric (~10min) ===";      python scripts/run_odor_mixtures.py --honegger 2>&1 | tail -5
  echo "=== [4/4] odor_mixtures (~25min) ===";        python scripts/run_odor_mixtures.py 2>&1 | tail -5
  echo "=== REGEN_FAST_DONE ==="

elif [ "$MODE" = "heavy" ]; then
  # --- TRUE_FROM_SCRATCH: uncomment to wipe and fully retrain (else cached runs are kept) ---
  # rm -rf results/ablations_r1 results/std_ablation results/task_complexity_r6 results/energy_only_r1

  echo "################ R3 + C3 ablations (run_ablation.py) ~6h ################"
  for s in $SEEDS; do
    python scripts/run_ablation.py --train-teacher --seed $s
    python scripts/run_ablation.py --no-gap   --kc-sparsity --seed $s --label r3i_no_gap_s$s
    python scripts/run_ablation.py --no-apl   --kc-sparsity --seed $s --label r3ii_no_apl_fix_s$s
    python scripts/run_ablation.py --shuffle  --kc-sparsity --seed $s --label r3iii_shuffle_s$s
    python scripts/run_ablation.py --kc-sparsity --ln-quantile 0.20  --seed $s --label r4_fix_q0.20_s$s
    python scripts/run_ablation.py --kc-sparsity --ln-quantile 0.417 --seed $s --label r4_fix_q0.417_s$s
    python scripts/run_ablation.py --kc-sparsity --ln-quantile 0.50  --seed $s --label r4_fix_q0.50_s$s
  done

  echo "################ STD ablation (run_std_ablation.py) ~1h ################"
  python scripts/run_ablation.py std --condition both

  echo "################ Task complexity (run_task_complexity.py) ~2.5h ################"
  for n in 7 14 56; do for s in $SEEDS; do
    python scripts/run_task_complexity.py --n-odors $n --seed $s
  done; done
  python scripts/run_task_complexity.py --canonical-only

  echo "################ Energy variants (run_training_energy_only.py) ~7-8h ################"
  for s in $SEEDS; do
    python scripts/run_training.py --train-teacher --seed $s
    python scripts/run_training.py --kc-sparsity                   --label canonical        --seed $s
    python scripts/run_training.py                                 --label ce_only          --seed $s
    python scripts/run_training.py --energy-weight 1               --label energy_1         --seed $s
    python scripts/run_training.py --energy-weight 3               --label energy_conserv   --seed $s
    python scripts/run_training.py --energy-weight 15              --label energy_aggress   --seed $s
    python scripts/run_training.py --energy-weight 50              --label energy_50        --seed $s
    python scripts/run_training.py --kc-energy-only --energy-weight 3   --label kc_energy_conserv --seed $s
    python scripts/run_training.py --kc-energy-only --energy-weight 15  --label kc_energy_aggress --seed $s
    python scripts/run_training.py --kc-energy-only --energy-weight 50  --label kc_energy_50      --seed $s
  done

  echo "################ REGEN_HEAVY_DONE ################"
  echo "Now re-run the notebook (Run All EXCEPT the canonical train cell) to load the fresh results."

else
  echo "Usage: bash code/scripts/regen.sh [fast|heavy]"; exit 1
fi
