# Kreher et al. 2008 Larval ORN Response Data

## Source

Kreher, S.A., Mathew, D., Kim, J., & Carlson, J.R. (2008).
Translation of sensory input into behavioral output via an olfactory system.
*Neuron*, 59(1), 110-124.

- PubMed ID: 18614033
- PMC ID: PMC2496968
- DOI: 10.1016/j.neuron.2008.06.010

## Data Description

This dataset contains electrophysiological recordings from larval Drosophila
olfactory receptor neurons (ORNs) across 28 stimulus conditions: a
spontaneous-firing baseline (`sfr`, no odor) plus 27 odorants (including CO2).
The model uses all 28 as the 28-way classification task.

### Receptors (21 functional ORs)
Or1a, Or2a, Or7a, Or13a, Or22c, Or24a, Or30a, Or33b, Or35a, Or42a, Or42b,
Or45a, Or45b, Or47a, Or49a, Or59a, Or67b, Or74a, Or82a, Or83a, Or85c

### Odors (28 conditions: sfr baseline + 27 odorants)
From diverse chemical classes: ketones, aromatics, alcohols, esters,
aldehydes, terpenes, and organic acids.

## Data Format

- `orn_responses.csv`: Response matrix (21 ORs × 28 conditions)
- Values: Change in spike rate (spikes/sec) relative to solvent control
- Positive values: excitation
- Negative values: inhibition

## Data Extraction

Values extracted from Figure 1B of Kreher et al. 2008 (Table S1 in supplementary).
Response magnitudes normalized to [0, 1] range for model input.

## Usage Note

This is the **gold standard** dataset for larval Drosophila ORN responses.
Unlike the DoOR database (adult fly data), this was measured directly from
larval ORNs and is appropriate for modeling the larval olfactory system.
