"""Dataset classes for connectome-constrained olfactory model training.

Uses ORN response data from Kreher et al. (2008) larval electrophysiology.
Contains 28 odors x 21 OR types.

This is a self-contained copy of the dataset utilities needed by the paper
package. The bundled data lives in data/kreher2008/ (relative to the package root).

Pipeline role
-------------
This module sits at the very front of the model pipeline. It turns the raw,
experimentally measured odor-receptor (OR) response matrix from Kreher et al.
(2008) into PyTorch datasets/dataloaders that feed the rest of the network:

    OR responses (this file)  ->  ORN (LIF)  ->  LN  ->  PN  ->  KC  <- APL
                                                                  |
                                                          linear decoder (28 odors)

Concretely, each sample produced here is a 21-dimensional vector of normalized
OR activations (one entry per olfactory receptor / glomerular channel) together
with the integer index of the odor that evoked it. Downstream, that 21-vector
is used to drive the input ORN layer of the spiking network, and the odor index
is the classification target for the linear decoder. The only stochasticity
introduced at this stage is sensory/input noise added on top of the fixed,
deterministic Kreher response template (see RepeatedOdorDataset).

Units / conventions
-------------------
The OR responses are dimensionless normalized activations (the on-disk file is
named ``orn_responses_normalized``), not raw firing rates in Hz, so ``noise_std``
is expressed in those same normalized units. Responses are clamped to be
non-negative because they represent rectified activation / rate-like quantities
that cannot be negative.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from typing import Tuple, List

# --- build-banner logging: silenced by default; set this module's _BUILD_VERBOSE = True to restore ---
_BUILD_VERBOSE = False
def _blog(*args, **kwargs):
    if _BUILD_VERBOSE:
        print(*args, **kwargs)



class RepeatedOdorDataset(Dataset):
    """Dataset that repeats each odor with independent noise samples.

    Each sample is (noisy_or_pattern, odor_index). Noise is drawn fresh
    on every __getitem__ call, so different epochs see different noise.

    The dataset has no stored "examples" per se: it holds a single clean
    response template per odor (``base_responses``) and synthesizes a virtual
    dataset of ``n_odors * repeats`` samples by re-noising those templates on the
    fly. Because the noise is sampled inside ``__getitem__`` (not precomputed),
    every epoch and every DataLoader pass sees a fresh, independent noise draw
    for the same underlying odor template. This is the model of sensory/trial
    variability used during training and evaluation.

    Args (constructor):
        orn_responses: Tensor of shape (n_odors, n_or_types) holding the clean,
            normalized OR response template for each odor. For the bundled
            Kreher 2008 data this is (28, 21). Cast to float in __init__.
        odor_names: List of length n_odors giving the human-readable odor label
            for each row of ``orn_responses`` (the label order defines the
            integer odor_index used as the classification target).
        repeats_per_odor: Number of noisy samples to expose per odor; this scales
            the apparent dataset length (see __len__).
        noise_std: Standard deviation of the injected input noise, in the same
            normalized units as the responses. If <= 0, samples are returned
            clean (no noise, no clamping).
        noise_type: Either 'additive' (noise magnitude is constant, independent
            of signal) or 'multiplicative' (noise scales with each channel's
            response magnitude; treated as the default for any non-'additive'
            value).
    """

    def __init__(self, orn_responses: torch.Tensor, odor_names: List[str],
                 repeats_per_odor: int = 10, noise_std: float = 0.1,
                 noise_type: str = 'additive'):
        # Clean per-odor OR templates, shape (n_odors, n_or_types); forced to
        # float32 so downstream tensor math and noise draws are well-typed.
        self.base_responses = orn_responses.float()
        # Row-aligned odor labels; index into this list == the classification target.
        self.odor_names = odor_names
        # How many noisy realizations to expose per odor (inflates dataset length).
        self.repeats = repeats_per_odor
        # Input-noise std in normalized OR units; 0 disables noise entirely.
        self.noise_std = noise_std
        # 'additive' vs anything-else ('multiplicative') selects the noise model.
        self.noise_type = noise_type
        # Number of distinct odors == number of decoder classes (28 for Kreher).
        self.n_odors = len(odor_names)

    def __len__(self) -> int:
        """Return the virtual dataset size: n_odors * repeats_per_odor.

        Each odor contributes ``repeats`` synthetic (re-noised) samples, so the
        DataLoader iterates this many times per epoch.
        """
        return self.n_odors * self.repeats

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Materialize one noisy odor sample for a flat dataset index.

        Args:
            idx: Flat sample index in [0, len(self)). Samples for a given odor
                occupy a contiguous block of ``repeats`` indices, so the odor is
                recovered by integer division.

        Returns:
            (pattern, odor_idx) where
              - pattern is a fresh noisy OR vector of shape (n_or_types,)
                (21 for Kreher), in normalized units, clamped to be non-negative.
              - odor_idx is the integer class label / row index of the odor.

        Side effects:
            Draws random noise via torch.randn_like, advancing the global RNG
            state; the source ``base_responses`` template is never mutated
            (it is cloned before noising).
        """
        # Map the flat index to its odor: indices are grouped in blocks of
        # ``repeats`` per odor, so floor-division yields the odor row.
        odor_idx = idx // self.repeats
        # Clone the clean template so we never mutate the shared base tensor.
        pattern = self.base_responses[odor_idx].clone()
        if self.noise_std > 0:
            # Standard-normal noise with the same shape as the OR vector;
            # each OR channel gets an independent draw.
            noise = torch.randn_like(pattern)
            if self.noise_type == 'multiplicative':
                # Multiplicative noise: scales with signal strength (biologically realistic)
                # Each OR channel gets independent noise proportional to its response
                # pattern *= (1 + N(0, noise_std)): a channel with zero response
                # stays zero, while strongly responding channels carry proportionally
                # larger absolute variability.
                pattern = pattern * (1.0 + noise * self.noise_std)
            else:
                # Additive noise: constant magnitude regardless of signal
                # pattern += N(0, noise_std), same absolute jitter on every channel.
                pattern = pattern + noise * self.noise_std
            # Clamp to non-negative (firing rates can't be negative)
            # Both noise models can push a channel below zero; rectify to keep
            # inputs in the physically meaningful (rate-like) range.
            pattern = torch.clamp(pattern, min=0)
        return pattern, odor_idx


def load_kreher2008_all_odors(
    data_dir: Path,
    train_repeats: int = 10,
    test_repeats: int = 5,
    noise_std: float = 0.1,
    noise_type: str = 'additive',
) -> Tuple[Dataset, Dataset, List[str]]:
    """Load Kreher 2008 data with ALL 28 odors for both train and test.

    Train and test sets differ only in their noise draws.

    Both returned datasets wrap the *same* clean OR response matrix; they are
    distinguished only by how many noisy repeats they expose per odor
    (``train_repeats`` vs ``test_repeats``) and by the fresh per-access noise
    sampling inside RepeatedOdorDataset. There is no held-out subset of odors:
    every one of the 28 odors appears in both splits (this is a noise-robustness
    /generalization-to-noise setup, not a disjoint-class split).

    Args:
        data_dir: Path to the directory containing kreher2008/ subfolder
        train_repeats: Noise repeats per odor for training
        test_repeats: Noise repeats per odor for testing
        noise_std: Noise standard deviation
        noise_type: 'additive' or 'multiplicative'

    Returns:
        (train_dataset, test_dataset, odor_names)
        - train_dataset / test_dataset: RepeatedOdorDataset instances over all
          28 odors.
        - odor_names: list of the 28 odor labels, row-aligned with the response
          matrix and with the integer class indices used as targets.

    Side effects:
        Reads the Kreher data files from disk and prints summary lines reporting
        the data dimensions and per-split sample counts.
    """
    # Resolve the bundled data location: <data_dir>/kreher2008/ holds both a
    # CSV (for labels + human-readable values) and an optional pre-tensorized .pt.
    kreher_dir = Path(data_dir) / 'kreher2008'
    csv_path = kreher_dir / 'orn_responses_normalized.csv'
    pt_path = kreher_dir / 'orn_responses_normalized.pt'

    if pt_path.exists():
        # Fast path: load the pre-saved tensor (weights_only=True for safe
        # unpickling), and still read the CSV purely to recover the odor labels
        # (the .pt stores only the numeric matrix, not the row index).
        or_responses = torch.load(pt_path, weights_only=True)
        df = pd.read_csv(csv_path, index_col=0)
        odor_names = df.index.tolist()
    else:
        # Fallback path: build the response tensor directly from the CSV values.
        # The CSV is indexed by odor name (column 0), so df.values is the
        # (n_odors, n_or_types) numeric block and df.index gives the labels.
        df = pd.read_csv(csv_path, index_col=0)
        or_responses = torch.from_numpy(df.values).float()
        odor_names = df.index.tolist()

    # Report dataset dimensions: rows == odors (28), columns == OR types (21).
    _blog(f'Kreher 2008 data: {len(odor_names)} odors x {or_responses.shape[1]} OR types')

    # Train split: same clean templates, exposing ``train_repeats`` noisy samples per odor.
    train_dataset = RepeatedOdorDataset(
        or_responses, odor_names,
        repeats_per_odor=train_repeats, noise_std=noise_std, noise_type=noise_type)
    # Test split: identical templates/noise model but typically fewer repeats.
    test_dataset = RepeatedOdorDataset(
        or_responses, odor_names,
        repeats_per_odor=test_repeats, noise_std=noise_std, noise_type=noise_type)

    # Echo the resulting virtual sample counts (n_odors * repeats) for each split.
    _blog(f'Train: {len(train_dataset)} samples ({len(odor_names)} odors x {train_repeats} repeats)')
    _blog(f'Test: {len(test_dataset)} samples ({len(odor_names)} odors x {test_repeats} repeats)')
    return train_dataset, test_dataset, odor_names


def create_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders.

    Thin wrapper that pairs each dataset with the appropriate shuffling policy.

    Args:
        train_dataset: Dataset for training (will be shuffled each epoch).
        test_dataset: Dataset for evaluation (kept in fixed order).
        batch_size: Number of (pattern, odor_idx) samples per minibatch.
        num_workers: DataLoader worker processes (0 == load in the main process;
            keeps the fresh-noise RNG behavior simple and reproducible).

    Returns:
        (train_loader, test_loader) DataLoaders. The train loader shuffles for
        stochastic optimization; the test loader does not, so evaluation order is
        deterministic and comparable across runs.
    """
    # Shuffle training batches so gradient steps see odors in random order.
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    # No shuffle for test: stable, repeatable ordering for metric computation.
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
