"""
Connectome-Constrained Olfactory Pathway Model.

Full pipeline: Real Odor → ORN → Antennal Lobe → PN → KC ← APL → Decoder

Following Lappalainen et al. 2024 methodology applied to larval Drosophila.
Uses real connectivity from Winding et al. 2023 and ORN responses from
Kreher et al. 2008 (larval electrophysiology data).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

# --- build-banner logging: silenced by default; set this module's _BUILD_VERBOSE = True to restore ---
_BUILD_VERBOSE = False
def _blog(*args, **kwargs):
    if _BUILD_VERBOSE:
        print(*args, **kwargs)


from .rate_layers import AntennalLobe, KenyonCellLayer, ConnectomeLinear


class ORtoORNMapping(nn.Module):
    """
    Fixed biological mapping from OR type responses to ORN neuron activations.

    In larval Drosophila:
    - Each OR type is expressed in exactly ONE ORN type (1:1 mapping)
    - Each ORN type exists as a bilateral pair (left and right neurons)
    - 21 larval OR types total (Kreher 2008 + Or83a)
    - Winding 2023 connectome has 42 ORN neurons (21 types × 2 hemispheres)

    This creates a PERFECT 1:1 mapping:
    - 21 OR types × 2 hemispheres = 42 ORN neurons

    This is a FIXED mapping (not learned) based on larval olfactory biology.

    References:
    - Kreher et al. (2005, 2008) Neuron: Larval OR functional responses
    - Kreher et al. (2011) PLOS One: Or83a spontaneous activity (~8 Hz)
    - Fishilevich et al. (2005) Current Biology: 1:1 OR-to-ORN mapping
    - Winding et al. (2023) Science: 42 ORN neurons (21 types × 2 hemispheres)

    OR types (21 total):
        Or1a, Or2a, Or7a, Or13a, Or22c, Or24a, Or30a, Or33b, Or35a, Or42a,
        Or42b, Or45a, Or45b, Or47a, Or49a, Or59a, Or67b, Or74a, Or82a, Or83a, Or85c

    Note on Or83a:
        Or83a was classified as "non-functional" in the Kreher 2008 Canton-S strain,
        meaning it showed no odor-evoked responses. However, Kreher et al. 2011 found
        it has spontaneous activity (~8 Hz). We include it with baseline firing only.
    """

    # Standard larval OR order (21 types, alphabetical by OR number)
    KREHER_OR_ORDER = [
        'Or1a', 'Or2a', 'Or7a', 'Or13a', 'Or22c', 'Or24a', 'Or30a', 'Or33b',
        'Or35a', 'Or42a', 'Or42b', 'Or45a', 'Or45b', 'Or47a', 'Or49a', 'Or59a',
        'Or67b', 'Or74a', 'Or82a', 'Or83a', 'Or85c'
    ]

    def __init__(self, n_or_types: int = 21, n_orn_neurons: int = 42):
        """
        Args:
            n_or_types: Number of OR types (21 larval ORs)
            n_orn_neurons: Number of ORN neurons in connectome (Winding 2023: 42)
        """
        super().__init__()
        self.n_or_types = n_or_types
        self.n_orn_neurons = n_orn_neurons

        # Create fixed binary mapping matrix (not learned)
        # Each OR type maps to exactly 2 ORN neurons (left and right)
        # 21 ORs × 2 = 42 ORN neurons (perfect 1:1 mapping)
        mapping = torch.zeros(n_or_types, n_orn_neurons)
        for i in range(n_or_types):
            left_idx = 2 * i
            right_idx = 2 * i + 1
            if left_idx < n_orn_neurons:
                mapping[i, left_idx] = 1.0
            if right_idx < n_orn_neurons:
                mapping[i, right_idx] = 1.0

        # Register as buffer (not a parameter - fixed, not learned)
        self.register_buffer('mapping', mapping)

        # Optional learnable gain per OR type (for response scaling)
        self.or_gains = nn.Parameter(torch.ones(n_or_types))

        # Log dimension info
        expected_orn = n_or_types * 2
        _blog(f"ORtoORNMapping: {n_or_types} ORs × 2 → {expected_orn} ORNs")
        if n_orn_neurons != expected_orn:
            print(f"  Warning: {n_orn_neurons} ORNs in connectome, expected {expected_orn}")

    def forward(self, or_responses: torch.Tensor) -> torch.Tensor:
        """
        Map OR responses to ORN activations using fixed biological mapping.

        Each OR response is duplicated to its left and right ORN neurons.

        Args:
            or_responses: (batch, n_or_types) OR receptor responses

        Returns:
            orn_activations: (batch, n_orn_neurons) ORN neuron activations
        """
        # Apply per-OR gain (learnable scaling)
        scaled_responses = or_responses * F.softplus(self.or_gains)

        # Apply fixed mapping: each OR → its left and right ORN
        # This is equivalent to: orn[2*i] = orn[2*i+1] = or[i]
        orn_activations = torch.matmul(scaled_responses, self.mapping)

        return orn_activations


class ConnectomeConstrainedModel(nn.Module):
    """
    Complete connectome-constrained olfactory pathway model.

    Architecture:
        OR (input) → ORN (fixed L/R duplication) → Antennal Lobe → PN → KC ← APL → Decoder

    All connectivity is constrained by the Winding et al. 2023 connectome.

    OR→ORN mapping is FIXED (not learned):
    - 21 OR types from Kreher et al. 2008 larval electrophysiology + Or83a
    - Each OR duplicated to left and right ORN neurons
    - 21 OR types × 2 = 42 ORN neurons (perfect 1:1 mapping)

    Learnable parameters:
    - OR gain scaling (per-OR response magnitude)
    - Unitary synapse strengths (scaled by synapse counts)
    - Neural thresholds
    - Decoder weights (for training objective)
    """

    def __init__(
        self,
        connectome: Dict[str, torch.Tensor],
        n_odors: int,
        n_or_types: int = 21,
        target_sparsity: float = 0.05,
        apl_iterations: int = 3,
    ):
        """
        Args:
            connectome: Dictionary with connectivity matrices:
                - orn_to_pn: (n_orn, n_pn) synapse counts
                - ln_to_pn: (n_ln, n_pn) synapse counts
                - pn_to_kc: (n_pn, n_kc) synapse counts
                - kc_to_apl: (n_kc, n_apl) synapse counts
                - apl_to_kc: (n_apl, n_kc) synapse counts
            n_odors: Number of odor classes for decoder
            n_or_types: Number of OR types in input (21 larval ORs)
            target_sparsity: Target KC sparsity (~5-10%)
            apl_iterations: Number of KC-APL feedback iterations
        """
        super().__init__()

        # Store dimensions
        self.n_or_types = n_or_types
        self.n_orn = connectome['orn_to_pn'].shape[0]
        self.n_pn = connectome['orn_to_pn'].shape[1]
        self.n_ln = connectome['ln_to_pn'].shape[0]
        self.n_kc = connectome['pn_to_kc'].shape[1]
        self.n_apl = connectome['kc_to_apl'].shape[1]
        self.n_odors = n_odors
        self.target_sparsity = target_sparsity

        _blog(f"Building model with:")
        _blog(f"  OR types (input): {self.n_or_types} (21 larval ORs)")
        _blog(f"  ORN: {self.n_orn} ({self.n_or_types} × 2 L/R hemispheres)")
        _blog(f"  LN: {self.n_ln}, PN: {self.n_pn}")
        _blog(f"  KC: {self.n_kc}, APL: {self.n_apl}")
        _blog(f"  Target sparsity: {target_sparsity:.1%}")

        # OR → ORN mapping (fixed biological duplication for L/R neurons)
        self.or_to_orn = ORtoORNMapping(n_or_types, self.n_orn)

        # Antennal Lobe: ORN → PN with LN lateral inhibition
        # LN receives input from ORN (orn_to_ln) and inhibits PN (ln_to_pn)
        self.antennal_lobe = AntennalLobe(
            orn_to_pn=connectome['orn_to_pn'],
            ln_to_pn=connectome['ln_to_pn'],
            orn_to_ln=connectome.get('orn_to_ln', None),  # Real LN input from connectome
        )

        # Kenyon Cells: PN → KC with APL feedback
        self.kc_layer = KenyonCellLayer(
            pn_to_kc=connectome['pn_to_kc'],
            kc_to_apl=connectome['kc_to_apl'],
            apl_to_kc=connectome['apl_to_kc'],
            target_sparsity=target_sparsity,
            n_iterations=apl_iterations,
        )

        # Linear decoder: KC → odor classification
        self.decoder = nn.Linear(self.n_kc, n_odors)

    def forward(
        self,
        or_input: torch.Tensor,
        return_all: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the olfactory pathway.

        Args:
            or_input: (batch, n_or_types) OR response pattern for each odor
            return_all: If True, return intermediate representations

        Returns:
            logits: (batch, n_odors) classification logits
            (optional) dict with orn_output, pn_output, kc_output, sparsity
        """
        # Map OR responses to ORN activations
        orn_output = self.or_to_orn(or_input)

        # Antennal lobe processing
        pn_output = self.antennal_lobe(orn_output)

        # KC encoding with APL feedback
        kc_output = self.kc_layer(pn_output)

        # Decode to odor class
        logits = self.decoder(kc_output)

        if return_all:
            sparsity = self.kc_layer.compute_sparsity(kc_output)
            diff_sparsity = self.kc_layer.compute_differentiable_sparsity(kc_output)
            return logits, {
                'orn_output': orn_output,
                'pn_output': pn_output,
                'kc_output': kc_output,
                'sparsity': sparsity,
                'diff_sparsity': diff_sparsity,
            }
        return logits

    def compute_loss(
        self,
        or_input: torch.Tensor,
        odor_labels: torch.Tensor,
        sparsity_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss with task and sparsity terms.

        Args:
            or_input: (batch, n_or_types) OR responses
            odor_labels: (batch,) odor class labels
            sparsity_weight: Weight for sparsity regularization

        Returns:
            total_loss: Combined loss
            metrics: Dictionary with individual loss terms
        """
        # Forward pass
        logits, intermediates = self.forward(or_input, return_all=True)

        # Task loss: cross-entropy for odor classification
        task_loss = F.cross_entropy(logits, odor_labels)

        # Sparsity loss: use differentiable proxy so gradients flow
        # Penalize deviation from target sparsity
        diff_sparsity = intermediates['diff_sparsity']
        sparsity_loss = (diff_sparsity - self.target_sparsity) ** 2

        # Combined loss
        total_loss = task_loss + sparsity_weight * sparsity_loss

        # Get actual sparsity for logging (non-differentiable)
        sparsity = intermediates['sparsity']

        # Metrics
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == odor_labels).float().mean().item()

        metrics = {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'sparsity_loss': sparsity_loss,
            'sparsity': sparsity,
            'accuracy': accuracy,
        }

        return total_loss, metrics

    @classmethod
    def from_data_dir(
        cls,
        data_dir: Path,
        n_odors: int,
        n_or_types: int = 21,
        **kwargs
    ) -> 'ConnectomeConstrainedModel':
        """
        Load model from saved connectome tensors.

        Args:
            data_dir: Path to data/ directory
            n_odors: Number of odor classes
            n_or_types: Number of OR types in input (21 larval ORs)
            **kwargs: Additional arguments for __init__

        Returns:
            Initialized model
        """
        winding_dir = data_dir / "winding2023"

        # Load connectivity matrices
        connectome = {
            'orn_to_pn': torch.load(winding_dir / "orn_to_pn.pt"),
            'ln_to_pn': torch.load(winding_dir / "ln_to_pn.pt"),
            'orn_to_ln': torch.load(winding_dir / "orn_to_ln.pt"),  # LN input from ORN
            'pn_to_kc': torch.load(winding_dir / "pn_to_kc.pt"),
            'kc_to_apl': torch.load(winding_dir / "kc_to_apl.pt"),
            'apl_to_kc': torch.load(winding_dir / "apl_to_kc.pt"),
        }

        return cls(connectome, n_odors, n_or_types=n_or_types, **kwargs)


class OdorRepresentationModel(nn.Module):
    """
    Simplified model for analyzing odor representations.

    Maps ORN responses to KC patterns without decoder.
    Useful for analyzing learned representations.
    """

    def __init__(
        self,
        connectome: Dict[str, torch.Tensor],
        target_sparsity: float = 0.05,
    ):
        super().__init__()

        self.n_orn = connectome['orn_to_pn'].shape[0]
        self.n_pn = connectome['orn_to_pn'].shape[1]
        self.n_kc = connectome['pn_to_kc'].shape[1]

        # Antennal Lobe (with full ORN→LN→PN pathway)
        self.antennal_lobe = AntennalLobe(
            orn_to_pn=connectome['orn_to_pn'],
            ln_to_pn=connectome['ln_to_pn'],
            orn_to_ln=connectome.get('orn_to_ln', None),
        )

        # KC layer
        self.kc_layer = KenyonCellLayer(
            pn_to_kc=connectome['pn_to_kc'],
            kc_to_apl=connectome['kc_to_apl'],
            apl_to_kc=connectome['apl_to_kc'],
            target_sparsity=target_sparsity,
        )

    def forward(self, orn_input: torch.Tensor) -> torch.Tensor:
        """Return KC representation for ORN input."""
        pn_output = self.antennal_lobe(orn_input)
        kc_output = self.kc_layer(pn_output)
        return kc_output

    def get_all_representations(self, orn_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return all intermediate representations."""
        pn_output, ln_activity = self.antennal_lobe(orn_input, return_intermediates=True)
        kc_output = self.kc_layer(pn_output)

        return {
            'orn': orn_input,
            'ln': ln_activity,
            'pn': pn_output,
            'kc': kc_output,
        }


def load_kreher2008_data(
    data_dir: Path,
    normalized: bool = True
) -> Tuple[torch.Tensor, list, list]:
    """
    Load Kreher et al. 2008 larval ORN response data.

    This is the gold standard for larval Drosophila OR responses.
    Data extracted from DoOR database's Kreher.2008.EN columns,
    plus Or83a (non-functional, spontaneous rate only from Kreher 2011).

    Contains 28 odors × 21 OR types.
    - Raw values: -31 to 297 spikes/sec (electrophysiology)
    - Or83a: 8 Hz spontaneous, no odor responses (non-functional in Canton-S)
    - Normalized: 0-1 range (clipped, shifted, scaled)

    Args:
        data_dir: Path to data/ directory
        normalized: If True, return normalized [0,1] values (recommended for model input)
                   If False, return raw spike rates

    Returns:
        or_responses: (n_odors, n_or_types) OR response matrix
        odor_names: List of odor names
        or_names: List of OR type names
    """
    import pandas as pd

    kreher_dir = data_dir / "kreher2008"

    if normalized:
        # Try loading pre-computed tensor first (faster)
        tensor_path = kreher_dir / "orn_responses_normalized.pt"
        csv_path = kreher_dir / "orn_responses_normalized.csv"

        if tensor_path.exists():
            or_responses = torch.load(tensor_path, weights_only=True)
            # Load CSV just for names
            df = pd.read_csv(csv_path, index_col=0)
            odor_names = df.index.tolist()
            or_names = df.columns.tolist()
        elif csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0)
            or_responses = torch.from_numpy(df.values).float()
            odor_names = df.index.tolist()
            or_names = df.columns.tolist()
        else:
            raise FileNotFoundError(
                f"Kreher 2008 normalized data not found at {kreher_dir}. "
                "Expected: orn_responses_normalized.pt or .csv"
            )
    else:
        # Load raw spike rates
        csv_path = kreher_dir / "orn_responses.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Kreher 2008 data not found at {csv_path}. "
                "Please ensure the data file exists."
            )
        df = pd.read_csv(csv_path, index_col=0)
        or_responses = torch.from_numpy(df.values).float()
        odor_names = df.index.tolist()
        or_names = df.columns.tolist()

    return or_responses, odor_names, or_names


def load_model_and_data(
    data_dir: Path,
    use_kreher: bool = True
) -> Tuple[ConnectomeConstrainedModel, torch.Tensor, list]:
    """
    Convenience function to load model with OR response data.

    Args:
        data_dir: Path to data/ directory
        use_kreher: If True, use Kreher 2008 larval data (recommended).
                   If False, fall back to DoOR data.

    Returns:
        model: Initialized model
        or_responses: (n_odors, n_or_types) OR response matrix
        odor_names: List of odor names
    """
    import pandas as pd

    if use_kreher:
        # Load Kreher 2008 larval electrophysiology data (gold standard)
        or_responses, odor_names, or_names = load_kreher2008_data(data_dir)
        n_or_types = len(or_names)
        _blog(f"Loaded Kreher 2008: {len(odor_names)} odors × {n_or_types} OR types")
        _blog(f"  OR types: {', '.join(or_names[:5])}...")
    else:
        # Fall back to DoOR data (adult fly, less appropriate for larvae)
        print("Warning: Using DoOR (adult fly) data. Kreher 2008 (larval) recommended.")
        or_responses = torch.load(data_dir / "larval_orn" / "orn_responses.pt")
        odor_metadata = pd.read_csv(data_dir / "larval_orn" / "odor_metadata.csv")
        odor_names = odor_metadata['odor_name'].tolist()
        n_or_types = or_responses.shape[1]

    n_odors = len(odor_names)
    _blog(f"OR response matrix shape: {or_responses.shape}")

    # Create model
    model = ConnectomeConstrainedModel.from_data_dir(
        data_dir, n_odors=n_odors, n_or_types=n_or_types
    )

    return model, or_responses, odor_names


if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("Testing ConnectomeConstrainedModel")
    print("=" * 60)

    # Create dummy connectome (42 ORNs, 21 PNs, 72 KCs from Winding 2023)
    connectome = {
        'orn_to_pn': torch.randint(0, 10, (42, 21)).float(),
        'ln_to_pn': torch.randint(0, 5, (10, 21)).float(),
        'pn_to_kc': torch.randint(0, 5, (21, 72)).float(),
        'kc_to_apl': torch.randint(0, 3, (72, 2)).float(),
        'apl_to_kc': torch.randint(0, 3, (2, 72)).float(),
    }

    # n_or_types=21 (21 larval ORs including Or83a)
    model = ConnectomeConstrainedModel(connectome, n_odors=28, n_or_types=21)

    # Test forward pass with 21 OR inputs
    batch_size = 4
    or_input = torch.randn(batch_size, 21).abs()  # 21 OR types
    logits, info = model.forward(or_input, return_all=True)

    print(f"\nForward pass:")
    print(f"  OR input shape: {or_input.shape} (21 larval ORs)")
    print(f"  ORN output shape: {info['orn_output'].shape} (42 ORNs = 21 × 2)")
    print(f"  PN output shape: {info['pn_output'].shape}")
    print(f"  KC output shape: {info['kc_output'].shape}")
    print(f"  Logits shape: {logits.shape} (28 odors)")
    print(f"  KC sparsity: {info['sparsity']:.2%}")

    # Test loss computation
    labels = torch.randint(0, 28, (batch_size,))
    loss, metrics = model.compute_loss(or_input, labels)

    print(f"\nLoss computation:")
    print(f"  Total loss: {metrics['total_loss']:.4f}")
    print(f"  Task loss: {metrics['task_loss']:.4f}")
    print(f"  Sparsity: {metrics['sparsity']:.2%}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal learnable parameters: {n_params:,}")

    print("\nModel test passed!")
