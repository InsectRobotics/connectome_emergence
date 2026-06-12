"""
Connectome-constrained neural network layers for larval Drosophila olfactory pathway.

Following Lappalainen et al. 2024 methodology:
- Connectivity patterns fixed from connectome (Winding et al. 2023)
- Synapse counts (N_ij) determine connection structure
- Learned parameters: unitary synapse strength, thresholds, time constants

Layers:
- ConnectomeLinear: Linear layer with connectome-constrained connectivity mask
- AntennalLobe: ORN→PN with LN lateral inhibition
- APL: Global KC inhibition for sparsity control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class ConnectomeLinear(nn.Module):
    """
    Linear layer with connectome-constrained connectivity.

    The connectivity pattern is fixed (from connectome), but synapse
    strengths are learned. Following Lappalainen et al., we learn a
    single "unitary synapse strength" that scales all synapse counts.

    Weight computation: W_ij = s * N_ij * mask_ij
    where:
    - s: learned unitary synapse strength (SINGLE SCALAR)
    - N_ij: synapse count from connectome (fixed)
    - mask_ij: binary mask (1 if connected, 0 otherwise)

    This is the most constrained formulation - only ONE learnable parameter
    per connection type (e.g., ORN→PN, PN→KC).
    """

    def __init__(
        self,
        synapse_counts: torch.Tensor,
        learn_per_synapse: bool = False,
        init_strength: float = 0.1,
        sign: str = "excitatory",
        use_bias: bool = False,
    ):
        """
        Args:
            synapse_counts: (n_source, n_target) matrix of synapse counts
            learn_per_synapse: If True, learn separate strength per connection.
                              If False (default), learn single global strength.
            init_strength: Initial unitary synapse strength
            sign: "excitatory" (positive weights) or "inhibitory" (negative weights)
            use_bias: If True, add learnable bias (default False for minimal params)
        """
        super().__init__()

        n_source, n_target = synapse_counts.shape
        self.n_source = n_source
        self.n_target = n_target
        self.sign = sign
        self.use_bias = use_bias

        # Store synapse counts (fixed)
        self.register_buffer("synapse_counts", synapse_counts.float())

        # Create binary mask from non-zero synapse counts
        mask = (synapse_counts > 0).float()
        self.register_buffer("mask", mask)

        # Normalize synapse counts for numerical stability
        max_count = synapse_counts.max()
        if max_count > 0:
            self.register_buffer("norm_counts", synapse_counts.float() / max_count)
        else:
            self.register_buffer("norm_counts", synapse_counts.float())

        # Learnable synapse strengths
        if learn_per_synapse:
            # One strength parameter per existing connection (NOT recommended)
            self.strengths = nn.Parameter(
                torch.full((n_source, n_target), init_strength) * mask
            )
        else:
            # Single global strength (most constrained - recommended)
            self.strengths = nn.Parameter(torch.tensor(init_strength))

        self.learn_per_synapse = learn_per_synapse

        # Optional bias per target neuron (disabled by default)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(n_target))
        else:
            self.register_buffer("bias", torch.zeros(n_target))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_source) input activations

        Returns:
            (batch, n_target) output activations
        """
        # Compute effective weights
        if self.learn_per_synapse:
            # Per-synapse strengths, masked and scaled by synapse counts
            if self.sign == "excitatory":
                strengths = F.softplus(self.strengths)
            else:
                strengths = -F.softplus(self.strengths)
            weights = strengths * self.norm_counts * self.mask
        else:
            # Global strength
            if self.sign == "excitatory":
                strength = F.softplus(self.strengths)
            else:
                strength = -F.softplus(self.strengths)
            weights = strength * self.norm_counts * self.mask

        # Linear transformation: (batch, n_source) @ (n_source, n_target)
        output = torch.matmul(x, weights) + self.bias

        return output

    def get_weights(self) -> torch.Tensor:
        """Return the effective weight matrix."""
        with torch.no_grad():
            if self.learn_per_synapse:
                if self.sign == "excitatory":
                    strengths = F.softplus(self.strengths)
                else:
                    strengths = -F.softplus(self.strengths)
                return strengths * self.norm_counts * self.mask
            else:
                if self.sign == "excitatory":
                    strength = F.softplus(self.strengths)
                else:
                    strength = -F.softplus(self.strengths)
                return strength * self.norm_counts * self.mask


class AntennalLobe(nn.Module):
    """
    Antennal Lobe processing: ORN → PN with LN lateral inhibition.

    Models the transformation from ORN input to PN output through:
    1. Direct ORN→PN excitation
    2. LN-mediated lateral inhibition (decorrelation/normalization)

    The LN population receives input from ORNs and provides
    inhibition to PNs, implementing a form of divisive normalization.
    """

    def __init__(
        self,
        orn_to_pn: torch.Tensor,
        ln_to_pn: torch.Tensor,
        orn_to_ln: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            orn_to_pn: (n_orn, n_pn) synapse counts
            ln_to_pn: (n_ln, n_pn) synapse counts (inhibitory)
            orn_to_ln: (n_orn, n_ln) synapse counts (optional, else uniform)
        """
        super().__init__()

        n_orn = orn_to_pn.shape[0]
        n_pn = orn_to_pn.shape[1]
        n_ln = ln_to_pn.shape[0]

        self.n_orn = n_orn
        self.n_pn = n_pn
        self.n_ln = n_ln

        # ORN → PN excitatory connections (single global unitary strength)
        self.orn_pn = ConnectomeLinear(orn_to_pn, learn_per_synapse=False, sign="excitatory")

        # LN → PN inhibitory connections (single global unitary strength)
        self.ln_pn = ConnectomeLinear(ln_to_pn, learn_per_synapse=False, sign="inhibitory")

        # ORN → LN connections (if provided, else simple pooling)
        if orn_to_ln is not None:
            self.orn_ln = ConnectomeLinear(orn_to_ln, learn_per_synapse=False, sign="excitatory")
        else:
            # Simple mean pooling if no specific connectivity
            self.orn_ln = None
            self.ln_pool_weight = nn.Parameter(torch.ones(1) * 0.5)

        # PN threshold (learnable)
        self.pn_threshold = nn.Parameter(torch.zeros(n_pn))

    def forward(
        self,
        orn_input: torch.Tensor,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through antennal lobe.

        Args:
            orn_input: (batch, n_orn) ORN activations
            return_intermediates: If True, also return LN activations

        Returns:
            pn_output: (batch, n_pn) PN activations
            (optional) ln_activity: (batch, n_ln) LN activations
        """
        # LN receives input from ORNs
        if self.orn_ln is not None:
            ln_input = self.orn_ln(orn_input)
        else:
            # Mean pooling with learned gain
            ln_input = orn_input.mean(dim=-1, keepdim=True) * F.softplus(self.ln_pool_weight)
            ln_input = ln_input.expand(-1, self.n_ln)

        # LN activation (ReLU for now, could add dynamics)
        ln_activity = F.relu(ln_input)

        # PN receives excitation from ORN and inhibition from LN
        pn_excitation = self.orn_pn(orn_input)
        pn_inhibition = self.ln_pn(ln_activity)

        # Net PN input (inhibition is already negative from ConnectomeLinear)
        pn_input = pn_excitation + pn_inhibition

        # Threshold-linear activation
        pn_output = F.relu(pn_input - self.pn_threshold)

        if return_intermediates:
            return pn_output, ln_activity
        return pn_output


class APLInhibition(nn.Module):
    """
    APL (Anterior Paired Lateral) neuron for global KC inhibition.

    APL provides feedback inhibition to enforce KC sparsity:
    - Receives excitatory input from all KCs
    - Provides inhibitory output to all KCs
    - Acts as a global activity regulator

    Uses connectome-constrained KC→APL and APL→KC connectivity.
    """

    def __init__(
        self,
        kc_to_apl: torch.Tensor,
        apl_to_kc: torch.Tensor,
        apl_gain: float = 1.0,
    ):
        """
        Args:
            kc_to_apl: (n_kc, n_apl) synapse counts (typically n_apl=1 or 2)
            apl_to_kc: (n_apl, n_kc) synapse counts
            apl_gain: Initial APL inhibition gain
        """
        super().__init__()

        self.n_kc = kc_to_apl.shape[0]
        self.n_apl = kc_to_apl.shape[1]

        # KC → APL excitatory
        self.kc_apl = ConnectomeLinear(kc_to_apl, learn_per_synapse=False, sign="excitatory")

        # APL → KC inhibitory
        self.apl_kc = ConnectomeLinear(apl_to_kc, learn_per_synapse=False, sign="inhibitory")

        # Global APL gain (affects inhibition strength)
        self.apl_gain = nn.Parameter(torch.tensor(apl_gain))

    def forward(self, kc_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute APL inhibition given KC activity.

        Args:
            kc_input: (batch, n_kc) KC pre-inhibition activations

        Returns:
            inhibition: (batch, n_kc) inhibitory current to subtract from KCs
            apl_activity: (batch, n_apl) APL neuron activity
        """
        # APL receives KC input
        apl_input = self.kc_apl(F.relu(kc_input))
        apl_activity = F.relu(apl_input) * F.softplus(self.apl_gain)

        # APL provides inhibition back to KCs (returns negative values)
        inhibition = self.apl_kc(apl_activity)

        return inhibition, apl_activity


class KenyonCellLayer(nn.Module):
    """
    Kenyon Cell layer with PN input and APL feedback inhibition.

    Implements:
    - PN→KC excitatory connections (connectome-constrained)
    - APL feedback inhibition for sparsity
    - Threshold-linear activation with learned thresholds
    """

    def __init__(
        self,
        pn_to_kc: torch.Tensor,
        kc_to_apl: torch.Tensor,
        apl_to_kc: torch.Tensor,
        target_sparsity: float = 0.05,
        n_iterations: int = 3,
    ):
        """
        Args:
            pn_to_kc: (n_pn, n_kc) synapse counts
            kc_to_apl: (n_kc, n_apl) synapse counts
            apl_to_kc: (n_apl, n_kc) synapse counts
            target_sparsity: Target fraction of active KCs (~5-10%)
            n_iterations: Number of KC-APL feedback iterations
        """
        super().__init__()

        self.n_pn = pn_to_kc.shape[0]
        self.n_kc = pn_to_kc.shape[1]
        self.target_sparsity = target_sparsity
        self.n_iterations = n_iterations

        # PN → KC excitatory connections (single global unitary strength)
        self.pn_kc = ConnectomeLinear(pn_to_kc, learn_per_synapse=False, sign="excitatory")

        # APL inhibition circuit
        self.apl = APLInhibition(kc_to_apl, apl_to_kc)

        # KC thresholds (learned)
        self.kc_threshold = nn.Parameter(torch.zeros(self.n_kc))

    def forward(
        self,
        pn_input: torch.Tensor,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through KC layer with APL feedback.

        Args:
            pn_input: (batch, n_pn) PN activations
            return_intermediates: If True, return iteration history

        Returns:
            kc_output: (batch, n_kc) sparse KC activations
        """
        # Initial KC excitation from PNs
        kc_exc = self.pn_kc(pn_input)

        # Iterative KC-APL interaction
        kc_activity = kc_exc.clone()
        history = [kc_activity.clone()] if return_intermediates else None

        for _ in range(self.n_iterations):
            # Compute current KC output (pre-APL)
            kc_output = F.relu(kc_activity - self.kc_threshold)

            # APL feedback inhibition
            inhibition, _ = self.apl(kc_output)

            # Update KC activity (excitation + inhibition)
            kc_activity = kc_exc + inhibition

            if return_intermediates:
                history.append(kc_activity.clone())

        # Final KC output
        kc_output = F.relu(kc_activity - self.kc_threshold)

        if return_intermediates:
            return kc_output, history
        return kc_output

    def compute_sparsity(self, kc_output: torch.Tensor) -> float:
        """Compute fraction of active KCs (non-differentiable, for logging)."""
        with torch.no_grad():
            active = (kc_output > 0).float()
            return active.mean().item()

    def compute_differentiable_sparsity(self, kc_output: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable proxy for sparsity.

        Uses soft thresholding: sigmoid of activity level gives continuous
        approximation of fraction active. Gradients can flow through this
        to adjust thresholds and APL gains.

        Returns:
            Differentiable sparsity proxy (tensor, not float)
        """
        # Use mean activity normalized to [0,1] range as proxy
        # Higher mean activity = higher sparsity violation
        mean_activity = kc_output.mean()

        # Also penalize large number of active units (soft count)
        # sigmoid(x) ≈ 1 for x >> 0, so this counts "strongly active" units
        soft_active = torch.sigmoid(kc_output * 10.0)  # Sharp sigmoid
        frac_active = soft_active.mean()

        return frac_active


class LeakyIntegrator(nn.Module):
    """
    Leaky integrator dynamics for time-varying inputs.

    τ dV/dt = -V + I(t)

    Used when processing time-series odor inputs.
    """

    def __init__(self, n_neurons: int, tau: float = 0.02, dt: float = 0.001):
        """
        Args:
            n_neurons: Number of neurons
            tau: Time constant (seconds)
            dt: Integration time step (seconds)
        """
        super().__init__()
        self.n_neurons = n_neurons
        self.dt = dt

        # Learnable time constant
        self.log_tau = nn.Parameter(torch.log(torch.tensor(tau)))

    @property
    def tau(self):
        return torch.exp(self.log_tau)

    def forward(self, current_input: torch.Tensor, prev_state: torch.Tensor) -> torch.Tensor:
        """
        One step of leaky integration.

        Args:
            current_input: (batch, n_neurons) input current
            prev_state: (batch, n_neurons) previous membrane state

        Returns:
            new_state: (batch, n_neurons) updated membrane state
        """
        alpha = self.dt / self.tau
        new_state = prev_state * (1 - alpha) + current_input * alpha
        return new_state

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize membrane state to zero."""
        return torch.zeros(batch_size, self.n_neurons, device=device)


if __name__ == "__main__":
    # Test the layers
    print("Testing ConnectomeLinear...")
    synapse_counts = torch.tensor([
        [5, 0, 3],
        [0, 2, 1],
        [4, 0, 0],
    ]).float()

    layer = ConnectomeLinear(synapse_counts, learn_per_synapse=True)
    x = torch.randn(2, 3)
    y = layer(x)
    print(f"  Input shape: {x.shape}, Output shape: {y.shape}")
    print(f"  Effective weights:\n{layer.get_weights()}")

    print("\nTesting AntennalLobe...")
    orn_to_pn = torch.randint(0, 10, (21, 21)).float()
    ln_to_pn = torch.randint(0, 5, (10, 21)).float()

    al = AntennalLobe(orn_to_pn, ln_to_pn)
    orn_input = torch.randn(4, 21).abs()  # ORN rates are non-negative
    pn_output = al(orn_input)
    print(f"  ORN input shape: {orn_input.shape}")
    print(f"  PN output shape: {pn_output.shape}")

    print("\nTesting KenyonCellLayer...")
    pn_to_kc = torch.randint(0, 5, (21, 72)).float()
    kc_to_apl = torch.randint(0, 3, (72, 2)).float()
    apl_to_kc = torch.randint(0, 3, (2, 72)).float()

    kc_layer = KenyonCellLayer(pn_to_kc, kc_to_apl, apl_to_kc)
    pn_input = torch.randn(4, 21).abs()
    kc_output = kc_layer(pn_input)
    sparsity = kc_layer.compute_sparsity(kc_output)
    print(f"  PN input shape: {pn_input.shape}")
    print(f"  KC output shape: {kc_output.shape}")
    print(f"  KC sparsity: {sparsity:.2%}")

    print("\nAll tests passed!")
