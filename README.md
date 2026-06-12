# Connectome-Constrained Spiking Olfactory Pathway Model

Spiking neural network model of the larval *Drosophila* olfactory pathway,
constrained by the complete Winding et al. (2023) connectome. All connectivity
(368 neurons, ~50,000 chemical synaptic contacts, ~12,000 synaptic edges) is fixed
to the connectome; only 449 scalar biophysical parameters are learned.

**Paper**: "Connectome-Constrained Spiking Neural Networks Reproduce Emergent
Computations in the Larval *Drosophila* Olfactory Pathway" (Cognitive Computation Neuroscience 2026).
See [`CCN_Final_2026.pdf`](CCN_Final_2026.pdf).

## Installation

```bash
git clone https://github.com/InsectRobotics/connectome_emergence.git
cd connectome_emergence
pip install -r requirements.txt
pip install .
```

This installs the model code as the importable `core`, `analysis`, and `scripts` packages
(source root `code/`). Dependencies (`requirements.txt` / `pyproject.toml`): Python ≥3.10,
PyTorch ≥2.0, NumPy, SciPy, Pandas, **scikit-learn** (odor-mixture SVM metric), Matplotlib,
JupyterLab, and psutil (auto-detects parallel-training worker count, falls back to
`os.cpu_count()`).

> **Running the training / figure scripts in place?** Use an editable install — `pip install -e .`
> — so they resolve `data/` and `results/` relative to the cloned repo.

**Conda alternative:** `conda env create -f environment.yml && conda activate connectome-emergence`.

## Architecture

```
OR (rate, 21 types) -> ORN (LIF, 42) -> LN (LIF, 108) -> PN (LIF, 72) -> KC (2-comp, 144) <- APL (graded, 2) -> Decoder (28 odors)
                                         ^___v gap junctions              ^___v KC-KC recurrent
```

- **ORN, LN, PN**: Leaky integrate-and-fire with surrogate gradients
- **KC**: Two-compartment (dendrite + axon) with learnable soma conductance
- **APL**: Graded (non-spiking) global inhibition
- **Gap junctions**: LN-LN (354 pairs), PN-PN sister (929), eLN-PN (103)
- **Non-AD contacts**: 6 pathways, weakly initialized, low learning rate
- **STD**: Tsodyks-Markram short-term depression on all chemical synapses
- **Noise**: 6 biologically motivated sources (OR 30% CV, membrane 1 mV, background 15 pA, synaptic 25%, threshold 1 mV, receptor 10%)

## Repository layout

Everything lives inside `code/`. Data, results, figures, and the paper sit at the repo root.

```
connectome_emergence/
|-- code/                                  # source root — installs as the `core`, `analysis`, `scripts` packages
|   |-- core/                              # model files (spiking + rate-based teacher)
|   |   |-- model.py                       #   SpikingConnectomeConstrainedModel — full OR->ORN->AL->KC<-APL->decoder pipeline
|   |   |-- layers.py                      #   spiking neuron/synapse layers (LIF, TwoCompartmentKC, APL, gap junctions, STD, non-AD)
|   |   |-- dataset.py                     #   Kreher-2008 OR-response loading + noisy DataLoaders
|   |   |-- rate_model.py                  #   ConnectomeConstrainedModel — rate-based ANN teacher for ANN->SNN transfer
|   |   +-- rate_layers.py                 #   rate-based layers (ConnectomeLinear, AntennalLobe, KenyonCellLayer)
|   |-- analysis/                          # metric / recompute / plot helpers (no training)
|   |   |-- recompute.py                   #   deterministic engine: checkpoint -> metrics (eval seed 42)
|   |   |-- compute.py                     #   decorrelation, Mancini APL test, concentration invariance
|   |   |-- utils.py                       #   shared primitives (cosine similarity, noisy forward pass, v_th bounds)
|   |   +-- plotting.py                    #   figure functions (each returns a fig, accepts show=)
|   |-- scripts/                           # experiment drivers (each self-bootstraps sys.path)
|   |   |-- run_training.py                #   canonical 5-seed training + R1 energy/sparsity variants
|   |   |-- run_ablation.py                #   R3 retrained ablations + posthoc + std subcommands
|   |   |-- run_odor_mixtures.py           #   R5: odor-mixture KC coding + Honegger sub-additivity
|   |   |-- run_teacher_consistency.py     #   R2: teacher/student parameter consistency
|   |   |-- run_task_complexity.py         #   R6: task complexity / KC-threshold scaling
|   |   |-- recompute_one.py               #   per-checkpoint recompute -> results/recompute_cache/ JSON (R1/R3/R4/STD cells fan it out, cached)
|   |   +-- regen.sh                       #   regenerate all results: bash code/scripts/regen.sh fast|heavy
|   +-- paper_figures.ipynb                # MAIN entry point: loads checkpoints, recomputes metrics, makes figures
|
|-- data/                                  # bundled inputs
|   |-- kreher2008/                        #   ORN response matrices (Kreher et al. 2008)
|   |-- winding2023/                       #   collapsed connectivity matrices (Winding et al. 2023)
|   +-- winding2023_compartments/          #   compartment-resolved contacts (aa, ad, da, dd)
|-- results/                               # per-experiment outputs + trained checkpoints (.pt)
|   |-- all_connections_nonad_canonical/   #   canonical 5-seed models + results.json
|   |-- energy_only_r1/  ablations_r1/  posthoc_ablations/  std_ablation/
|   |-- odor_mixtures_r5/  teacher_consistency_r2/  task_complexity_r6/
|   +-- _shuffled_connectomes/  recompute_cache/  f4_extra/
|-- figures/                               # notebook outputs: PNGs + vector PDFs + paper_values.json
|-- CCN_Final_2026.pdf                     # the paper (CCN 2026)
+-- environment.yml   pyproject.toml   requirements.txt   .gitignore   LICENSE   README.md
```

## Setup

> Install dependencies first — see [Installation](#installation).

### 1. Reproduce the paper (no training required)

```bash
jupyter lab code/paper_figures.ipynb
```

Run all cells. The notebook loads pre-trained checkpoints and pre-computed results from
`results/`, recomputes every metric at runtime via `analysis/recompute.py` (deterministic,
eval seed 42), generates all figures into `figures/`, and prints every value cited in the
paper — also persisted to `figures/paper_values.json`.

> **Parallel training:** the notebook's "Option A" cells auto-detect the machine (CPU cores +
> free RAM) and dispatch seeds in parallel (~30–60 min on a modern multi-core machine).

### 2. Train / regenerate from scratch (optional)

Run scripts directly from the repo root — they self-bootstrap `sys.path`:

```bash
python code/scripts/run_training.py                        # canonical models (5 seeds, ~2 h)
python code/scripts/run_training.py --label ce_only        # R1 energy/sparsity variants (~8 h)
python code/scripts/run_ablation.py --no-gap --kc-sparsity # R3 retrained ablations (~4 h)
python code/scripts/run_ablation.py posthoc                # R3 post-hoc ablations (~30 min)
python code/scripts/run_ablation.py std --condition both   # STD ablation (~2 h)
python code/scripts/run_odor_mixtures.py                   # R5 mixture coding (~25 min)
python code/scripts/run_odor_mixtures.py --honegger        # R5 Honegger sub-additivity (~10 min)
python code/scripts/run_teacher_consistency.py             # R2 parameter consistency (~5 min)
python code/scripts/run_task_complexity.py --n-odors 14 --seed 42   # R6 task complexity
```

Or in one go: `bash code/scripts/regen.sh fast` (eval-only, ~50 min) /
`bash code/scripts/regen.sh heavy` (all retraining, ~18–20 h).

## Where outputs go

| Output | Location | Written by |
|--------|----------|-----------|
| Figures (PNG) | `figures/` | notebook figure cells |
| **All paper values** (JSON) | `figures/paper_values.json` | notebook final cell |
| Per-experiment results (JSON) | `results/<subdir>/*.json` | `code/scripts/run_*.py` |
| Trained checkpoints (`.pt`) | `results/<subdir>/*.pt` | training drivers |

## Generated figures

| Figure | File | Description |
|--------|------|-------------|
| Figure 2 | `figures/2_core.png` | Pairwise cosine-similarity matrices (OR→PN→KC decorrelation) |
| Figure 3 | `figures/3_concentration.png` | Concentration invariance (gain control, accuracy, similarity) |
| Figure 4 | `figures/4_abl.png` | Architecture / ablation bars |
| Figure S2 | `figures/S2_confusion_matrix.png` | Classification confusion matrix |
| Figure S3 | `figures/S3_within_odor_consistency.png` | Within-odor KC consistency |
| Figure S4 | `figures/S4_threshold_scaling.png` | KC threshold saturation vs task difficulty |
| KC heatmap | `figures/S1_kc_heat.png` | KC activity heatmap across odors |

## Canonical results (5-seed ensemble)

| Metric | Value | Biological comparison |
|--------|-------|-----------------------|
| Test accuracy | 70.3% ± 1.1% | ~20× chance (3.6%) |
| Centroid accuracy | 65.9% ± 0.5% | decoder-free odor discrimination |
| KC sparsity | 12.5% ± 0.2% | sparse expansion coding |
| AL decorrelation | 4.4% | modest, as expected |
| MB decorrelation | 35.8% | expansion coding (Bhandawat et al., 2007) |
| APL suppression | 1.85 ± 0.05 (~46%) | ~50% (Mancini et al., 2023) |
| PN dynamic range | 1.23 ± 0.02× | sublinear gain (Olsen et al., 2010) |
| KC pattern similarity | 0.942 | cross-model convergence |
| Parameter correlation | 0.909 | connectome constrains the solution |

## Emergent computations (not trained for)

- **MB-localized decorrelation**: 35.8% in MB vs 4.4% in AL
- **APL suppression**: KC activity reduced by ~46%, matching optogenetic data
- **Concentration invariance**: gain control compresses the OR→PN→KC dynamic range
- **Sub-additive mixture coding**: 69.4% of KCs sub-additive (Honegger 2011: 73%)

## Data

Connectome from Winding et al. (2023); ORN responses from Kreher et al. (2008); bundled in `data/`:
- `kreher2008/` — ORN response matrices (normalized + raw)
- `winding2023/` — collapsed connectivity matrices (`.pt`)
- `winding2023_compartments/` — compartment-resolved contacts (aa, ad, da, dd)

---

This code was written by Jordan Watts, in Barbara Webb's Insect Robotics Group at the University of Edinburgh, School of Informatics.
