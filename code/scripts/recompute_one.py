"""Recompute the deterministic eval metrics for ONE checkpoint and cache to JSON.

This is the parallel/cached backend for the notebook's R1 and R3 recompute cells.
Each (label, seed) is an independent subprocess so they fan out across cores via
run_parallel, and the JSON it writes is a permanent cache (the recompute is fully
deterministic at eval seed 42, so the cached result is bit-for-bit reproducible).

The output JSON is exactly the dict returned by recompute_metrics (accuracy,
sparsity, centroid, al_decorr, mb_decorr, total_decorr, [mancini]) plus 'seed'
and 'label'. The R3 cell reshapes it into its row format; the R1 cell feeds the
raw dicts straight into aggregate() (which ignores the non-numeric 'label').

Usage:
    python scripts/recompute_one.py --ckpt PATH --label NAME --seed N --output OUT.json
                                    [--ablate gap|apl] [--no-mancini]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

# Make `analysis` and `scripts` importable regardless of CWD.
_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))

from analysis.recompute import load_checkpoint, recompute_metrics, disable_std_params  # noqa: E402
from scripts.run_ablation import ablate_gap_junctions, ablate_apl          # noqa: E402


def _find_data_dir():
    """Locate the connectome data dir (the one containing kreher2008/), matching the notebook."""
    root = _CODE.parent  # repo root
    for p in (root / 'data', root, Path.cwd() / 'data', Path.cwd()):
        if (p / 'kreher2008').is_dir():
            return p
    raise FileNotFoundError("could not locate a data dir containing kreher2008/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='checkpoint .pt path')
    ap.add_argument('--label', required=True, help='cache label (also written into the JSON)')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--output', required=True, help='cache JSON path to write')
    ap.add_argument('--ablate', choices=['gap', 'apl'], default=None,
                    help='post-hoc ablation applied before recompute (R3 post-hoc rows)')
    ap.add_argument('--disable-std', action='store_true',
                    help='disable short-term depression before recompute (R5/STD ablation)')
    ap.add_argument('--with-concentration', action='store_true',
                    help='also run the concentration sweep and cache the PN/KC dynamic ranges')
    ap.add_argument('--no-mancini', action='store_true', help='skip the Mancini APL test')
    args = ap.parse_args()

    data_dir = _find_data_dir()
    or_responses = torch.from_numpy(
        pd.read_csv(data_dir / 'kreher2008' / 'orn_responses_normalized.csv', index_col=0).values).float()

    model = load_checkpoint(data_dir, Path(args.ckpt))
    if args.ablate == 'gap':
        ablate_gap_junctions(model)
    elif args.ablate == 'apl':
        ablate_apl(model)
    if args.disable_std:
        disable_std_params(model)

    m = recompute_metrics(model, data_dir, or_responses, with_mancini=not args.no_mancini,
                          with_concentration=args.with_concentration)
    if args.with_concentration and 'concentration' in m:
        # flatten the PN/KC dynamic ranges the STD ablation cell needs, drop the bulky sub-dict
        _tests = m['concentration'].get('tests', {})
        m['pn_range'] = _tests.get('pn_range')
        m['kc_range'] = _tests.get('kc_range')
        m.pop('concentration', None)
    m['seed'] = args.seed
    m['label'] = args.label

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(m, open(out, 'w'), indent=1)
    print(f"[recompute] {args.label} s{args.seed}: acc={m['accuracy']:.3f} "
          f"AL={m['al_decorr']:.1f} MB={m['mb_decorr']:.1f}"
          + (f" manc={m['mancini']:.2f}" if 'mancini' in m else ""))


if __name__ == '__main__':
    main()
