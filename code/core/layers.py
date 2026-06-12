"""
Spiking neural network layers for connectome-constrained olfactory pathway.

This module is the low-level "circuit primitives" library for the larval
Drosophila olfactory model described in the CCN 2026 paper. It provides the
neuron models, synapse model, and composite circuit blocks that model.py wires
together. The end-to-end pipeline is:

    OR responses (Kreher 2008, graded)
        -> ORN (LIF spiking)
        -> LN (LIF spiking, lateral inhibition / excitation)
        -> PN (LIF spiking)
        -> KC (single- or two-compartment spiking)  <-- APL (graded divisive inhibition)
        -> linear decoder over 28 odors  (decoder lives in model.py, not here)

Connectivity (who-connects-to-whom and how many synapses) is FIXED by the
Winding 2023 connectome and passed in as synapse-count matrices. The ~449 free
biological parameters (synapse strengths, time constants, thresholds, soma
conductance, gap-junction conductances, STD parameters, gains) are LEARNED.

Implements all neuron and synapse layer types used in the paper model:
- LIFNeuron: Leaky integrate-and-fire with surrogate gradients (ORN, LN, PN)
- TwoCompartmentKC: Dendrite + axon with conductance-based soma coupling (KC)
- SpikingConnectomeLinear: 1 learnable strength per connectome-derived pathway, with STD
- SpikingAntennalLobe: AL circuit (ORN→LN/PN, LN→PN/LN, PN→LN + gap junctions)
- SpikingKenyonCellLayer: MB calyx (PN→KC, KC→APL, APL→KC, KC-KC recurrence)
- SpikingAPLInhibition: Graded APL interneuron for global KC inhibition

Key parameter initialization defaults (SpikingParams below). Learnable parameters
are clamped to the biological ranges defined by the *_MIN/*_MAX constants below,
not to the init values listed here:
- tau_m: 20 ms init (membrane time constant, learnable per population; clamp 5-50 ms)
- tau_syn: 5 ms (synaptic current decay)
- tau_ref: 2 ms (refractory period, fixed)
- v_th: -40 mV init (spike threshold, learnable per neuron; clamp -55 to -30 mV)
- v_reset: -60 mV (reset voltage, = v_rest)
- g_soma: 10 nS init (KC soma conductance, learnable; clamp 1-100 nS, trained range 1-20 nS)

Biological justification for non-spiking components:
- OR responses: Receptor-mediated transduction produces graded signals
- APL: Invertebrate interneurons often use graded potentials
  (Papadopoulou et al. 2011, Neuron: APL uses graded transmission)

Conventions in this file:
- All physical quantities are in SI base units: time in seconds (s), voltage in
  volts (V), current in amperes (A), conductance in siemens (S), capacitance in
  farads (F), resistance in ohms (Ω).
- Tensors are batched as (batch, n_neurons) unless otherwise stated.
- Positivity-constrained learnable parameters (time constants, conductances,
  synapse strengths) are stored in LOG space and exponentiated on read, so the
  optimizer works in an unconstrained space while the realized value stays > 0.
  Probabilities (e.g. STD release probability U) are stored as logits and passed
  through a sigmoid for the same reason.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

# --- build-banner logging: silenced by default; set this module's _BUILD_VERBOSE = True to restore ---
_BUILD_VERBOSE = False
def _blog(*args, **kwargs):
    if _BUILD_VERBOSE:
        print(*args, **kwargs)


# Small constants for numerical stability
# _EPS guards divisions that can otherwise hit 0 in the two-compartment KC
# integration-matrix denominator (see _compute_integration_matrices).
_EPS = 1e-8

# =============================================================================
# BIOLOGICAL BOUNDS FOR LEARNED PARAMETERS
# =============================================================================
# These bounds constrain learned parameters to biologically realistic ranges.
# References: Gerstner & Kistler 2002, Turner et al. 2008
# They are enforced after each optimizer step via the clamp_to_biological_bounds()
# methods on the layers below. Log-space bounds are pre-computed here so the
# clamp can operate directly on the log-space nn.Parameter.

# Voltage threshold (V): -55 to -30 mV
# Wide range allows network to learn appropriate thresholds for decorrelation
# Runaway firing prevented by tau_ref=2ms, not by tight v_th bounds
V_TH_MIN = -0.055  # -55 mV (lower bound)
V_TH_MAX = -0.030  # -30 mV (upper bound)

# Membrane time constant (s): 5-50 ms
TAU_M_MIN = 5e-3   # 5 ms
TAU_M_MAX = 50e-3  # 50 ms
LOG_TAU_M_MIN = np.log(TAU_M_MIN)  # ~-5.30  (natural log of the second-valued bound)
LOG_TAU_M_MAX = np.log(TAU_M_MAX)  # ~-3.00

# Soma conductance (S): 1-100 nS (default range; training narrows to 1-20 nS)
# This is the conductance coupling the two KC compartments (dendrite<->axon).
G_SOMA_MIN = 1e-9   # 1 nS
G_SOMA_MAX = 100e-9 # 100 nS
LOG_G_SOMA_MIN = np.log(G_SOMA_MIN)  # ~-20.7  (log-space clamp bound)
LOG_G_SOMA_MAX = np.log(G_SOMA_MAX)  # ~-16.1

# APL time constant (s): 10-50 ms (GGN/APL tau_m ~33ms from Ray et al. 2020)
TAU_APL_MIN = 10e-3
TAU_APL_MAX = 50e-3
LOG_TAU_APL_MIN = np.log(TAU_APL_MIN)
LOG_TAU_APL_MAX = np.log(TAU_APL_MAX)

# Synaptic strength bounds (A): 1e-12 to 1e-7
# A single unitary-synapse current "strength" s is learned per pathway and stored
# in log space; these are its clamp bounds (1 pA to 100 nA of injected current).
LOG_STRENGTH_MIN = np.log(1e-12)  # 1 pA
LOG_STRENGTH_MAX = np.log(1e-7)   # 100 nA


class SurrogateSpike(torch.autograd.Function):
    """
    Surrogate gradient for spike detection (SuperSpike).

    A hard threshold (v > v_th) has a derivative that is zero almost everywhere
    and undefined at the threshold, so gradients cannot flow back through a spike.
    The surrogate-gradient trick keeps the exact step function in the FORWARD pass
    (so the network still emits binary spikes) but substitutes a smooth, nonzero
    pseudo-derivative in the BACKWARD pass so backprop can train upstream weights.

    Forward: hard threshold (v > v_th)
    Backward: smooth sigmoid-like derivative for gradient flow

    This is essential for training spiking neural networks with backprop.
    Uses the SuperSpike surrogate from Zenke & Ganguli (2018).
    """

    @staticmethod
    def forward(ctx, v_centered, scale=25.0):
        """
        Forward pass: hard threshold at 0.

        Args:
            v_centered: (batch, n) tensor of (v - v_th), i.e. membrane voltage
                relative to threshold in volts. A neuron spikes where this is > 0.
            scale: Steepness of the surrogate gradient (units 1/V); stashed for
                the backward pass, does not affect the forward output.

        Returns:
            (batch, n) float tensor of 0./1. spikes (1 where v_centered > 0).

        Side effects:
            Saves v_centered and scale on ctx for use in backward().
        """
        # Cache inputs needed to compute the pseudo-gradient in backward().
        ctx.save_for_backward(v_centered)
        ctx.scale = scale
        # Hard Heaviside step: this is the actual (non-differentiable) spike.
        return (v_centered > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: smooth sigmoid derivative (SuperSpike pseudo-gradient).

        d/dv sigmoid(scale * v) = scale * sigmoid * (1 - sigmoid)
        For simplicity, use: 1 / (1 + |scale * v|)^2

        Returns the gradient w.r.t. v_centered (and None for the `scale` arg,
        which is not learnable).
        """
        v_centered, = ctx.saved_tensors
        scale = ctx.scale

        # SuperSpike surrogate gradient: a smooth bump peaked at threshold
        # (v_centered = 0, value = scale) that decays polynomially as |v - v_th|
        # grows. This replaces the Dirac-delta derivative of the true step.
        grad = scale / (1 + scale * torch.abs(v_centered)) ** 2

        # Chain rule: multiply by upstream grad; second None is for `scale`.
        return grad * grad_output, None


def surrogate_spike(v_centered: torch.Tensor, scale: float = 100.0) -> torch.Tensor:
    """Apply the SuperSpike surrogate spike function.

    Thin functional wrapper around SurrogateSpike.apply so callers can use it
    like an ordinary differentiable op.

    Args:
        v_centered: (batch, n) (v - v_th) in volts.
        scale: gradient steepness (1/V). Larger -> sharper learning window.

    Returns:
        (batch, n) binary spikes (hard forward, SuperSpike backward).

    Note: scale is in 1/Volts since v_centered is in Volts.
    With scale=100, gradient at 10mV from threshold is ~25.
    With scale=25 (old), gradient at 10mV was only ~6.
    """
    return SurrogateSpike.apply(v_centered, scale)


def soft_spike(v_centered: torch.Tensor, temperature: float = 0.001) -> torch.Tensor:
    """
    Soft spike function for training (continuous relaxation).

    Unlike the surrogate-gradient functions, this does NOT emit a hard spike in
    the forward pass; it returns a graded sigmoid value in [0, 1]. As the
    temperature -> 0 the sigmoid approaches the true step function, so it behaves
    like a smooth, fully-differentiable spike. This is the default ('soft')
    surrogate_method used by the neuron layers during training.

    Args:
        v_centered: (batch, n) (v - v_th), voltage relative to threshold (in Volts)
        temperature: Controls sharpness (in Volts, e.g., 1mV = 0.001). Smaller =
            sharper / more spike-like; larger = smoother / easier gradients.

    Returns:
        (batch, n) graded "spike probability" in [0, 1].
    """
    return torch.sigmoid(v_centered / temperature)


class SLAYERSpike(torch.autograd.Function):
    """
    SLAYER (Spike Layer Error Reassignment in Time) surrogate gradient.

    From Shrestha & Orchard (2018): "SLAYER: Spike Layer Error Reassignment in Time"

    Key insight: Use a probability density function (PDF) that models the
    probability of a spike given the membrane potential. This gives gradients
    that are:
    - Zero when membrane is far below threshold (changing it won't help)
    - Maximum just below threshold (high impact zone)
    - Decay above threshold (spike already happened)

    The PDF is: ρ(u) = exp(-|u|/τ_pdf) / (2τ_pdf)

    This creates a "learning window" around the threshold that focuses
    gradient information where it matters most.

    Compared to SuperSpike (1/(1+|u|)²):
    - SLAYER has sharper peak at threshold
    - SLAYER decays exponentially (SuperSpike decays polynomially)
    - SLAYER better handles temporal credit assignment

    Like SurrogateSpike, the forward pass is still a hard threshold; only the
    backward pseudo-gradient differs.

    Parameters:
        tau_pdf: Width of the learning window (in Volts)
                 Typical: 5-20mV (0.005-0.020 V)
        alpha: Peak scaling factor
    """

    @staticmethod
    def forward(ctx, v_centered: torch.Tensor, tau_pdf: float = 0.010, alpha: float = 1.0):
        """
        Forward pass: hard threshold at 0.

        Args:
            v_centered: (batch, n) (v - v_th), voltage relative to threshold (V)
            tau_pdf: Width of PDF (learning window) in Volts; stashed for backward.
            alpha: Peak gradient scaling; stashed for backward.

        Returns:
            (batch, n) float tensor of 0./1. spikes.

        Side effects:
            Saves v_centered, tau_pdf, alpha on ctx for backward().
        """
        ctx.save_for_backward(v_centered)
        ctx.tau_pdf = tau_pdf
        ctx.alpha = alpha
        # Hard Heaviside step (the true spike); gradient handled in backward().
        return (v_centered > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass: exponential PDF surrogate gradient.

        ∂s/∂u ≈ ρ(u) = (α / 2τ) * exp(-|u| / τ)

        This creates symmetric exponential decay from threshold. Returns the
        gradient w.r.t. v_centered plus two None's for the non-learnable
        tau_pdf and alpha arguments.
        """
        v_centered, = ctx.saved_tensors
        tau_pdf = ctx.tau_pdf
        alpha = ctx.alpha

        # Exponential PDF: peaks at u=0, decays symmetrically
        # ρ(u) = (1 / 2τ) * exp(-|u| / τ)
        grad = (alpha / (2 * tau_pdf)) * torch.exp(-torch.abs(v_centered) / tau_pdf)

        # Chain rule; trailing None,None are for tau_pdf and alpha.
        return grad * grad_output, None, None


def slayer_spike(v_centered: torch.Tensor, tau_pdf: float = 0.010, alpha: float = 1.0) -> torch.Tensor:
    """
    Apply SLAYER surrogate spike function.

    SLAYER uses an exponential PDF surrogate that creates a focused learning
    window around the threshold. This is particularly effective for:
    - Temporal coding (precise spike timing matters)
    - Multi-layer SNNs (better temporal credit assignment)
    - Low firing rate regimes

    Args:
        v_centered: (batch, n) (v - v_th), voltage relative to threshold (V)
        tau_pdf: Width of learning window (default 10mV)
        alpha: Peak gradient scaling

    Returns:
        Binary spikes (hard threshold in forward, SLAYER gradient in backward)

    Example:
        At v_centered = 0 (threshold): gradient = alpha / (2 * tau_pdf) ≈ 50
        At v_centered = tau_pdf: gradient = 50 * exp(-1) ≈ 18
        At v_centered = 3*tau_pdf: gradient = 50 * exp(-3) ≈ 2.5
    """
    return SLAYERSpike.apply(v_centered, tau_pdf, alpha)



@dataclass
class SpikingParams:
    """
    Container of shared biophysical constants for the spiking network (SI units).

    A single SpikingParams instance is typically created once and threaded
    through every layer so all populations share the same dt, noise levels, and
    default biophysics. Fields that the model later makes learnable (tau_m, v_th,
    g_soma, synapse strengths, ...) use these only as INITIAL values; the realized
    learnable values are stored on the individual layers and clamped to the
    *_MIN/*_MAX bounds above.

    Based on brian2_like.py parameters with additional olfactory-specific values.
    """
    # Timing
    dt: float = 1e-3              # Integration timestep (1 ms); all exponential-decay factors below assume this dt

    # Membrane properties
    R_m: float = 1e9              # Membrane resistance (1 GΩ); sets current->voltage scale V_inf = V_rest + R_m*I
    tau_m: float = 20e-3          # Membrane time constant (20 ms); default init for learnable per-population tau_m
    tau_syn: float = 5e-3         # Synaptic current decay (5 ms); sets exponential decay of post-synaptic current
    tau_ref: float = 2e-3         # Refractory period (2 ms) - biological range 2-5 ms for KCs

    # Voltage thresholds (V)
    v_th: float = -40e-3          # Spike threshold (-40 mV); default init for learnable per-neuron v_th
    v_reset: float = -60e-3       # Reset voltage (-60 mV); voltage clamped to here right after a spike
    v_rest: float = -60e-3        # Resting potential (-60 mV); leak target (equal to v_reset by design)
    v_min: float = -80e-3         # Hard lower bound (-80 mV); floor that prevents runaway hyperpolarization

    # KC 2-compartment model
    g_soma: float = 10e-9         # Soma conductance (10 nS); default init for learnable dendrite<->axon coupling

    # Circuit noise (biologically realistic stochasticity)
    # The model injects six biological noise sources; five are configured here
    # (the sixth, ORN receptor noise, is applied upstream in model.py). Each is
    # tagged [Noise N] at its injection point in the forward() methods below.
    # 1. Membrane voltage noise - stochastic ion channel gating
    v_noise_std: float = 0.5e-3   # Voltage noise std (0.5 mV); additive Gaussian on membrane voltage
    # 2. Background synaptic bombardment - ongoing input from other brain regions
    i_noise_std: float = 5e-12    # Current noise std (5 pA); additive Gaussian on input current
    # 3. Synaptic release stochasticity - probabilistic vesicle release
    syn_noise_std: float = 0.15   # Multiplicative noise on synaptic transmission (15% CV)
    # 4. Spike threshold jitter - uncertainty in exact threshold crossing
    threshold_jitter_std: float = 0.2e-3  # Threshold noise std (0.2 mV); additive Gaussian on v_th at spike check
    # 5. ORN receptor noise - stochastic odorant-receptor binding (applied in model.py)
    orn_receptor_noise_std: float = 0.05  # Per-timestep multiplicative noise (5%)
    # Master switch
    circuit_noise_enabled: bool = True    # If False, all noise sources above are skipped (deterministic eval)

    @property
    def C_m(self) -> float:
        """Membrane capacitance derived from tau_m and R_m.

        From the membrane equation tau_m = R_m * C_m, so C_m = tau_m / R_m.
        With defaults (20 ms / 1 GΩ) this is ~20 pF, a typical small-neuron value.
        """
        return self.tau_m / self.R_m  # ~20 pF


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron for ORN, LN, PN populations.

    Models a homogeneous population of n_neurons single-compartment LIF cells
    sharing one time constant but (optionally) per-neuron thresholds. Used for the
    spiking populations of the antennal lobe (ORN, LN, PN) and as the default
    single-compartment KC.

    Implements standard LIF dynamics:
        τ_m dV/dt = -(V - V_rest) + R_m * I
        spike when V > V_th, then V → V_reset

    Uses exact exponential integration for numerical stability (closed-form
    solution over one dt assuming constant input current), which is unconditionally
    stable regardless of dt/tau_m ratio.

    Surrogate gradient methods (for training):
        - 'soft': Soft sigmoid (continuous relaxation)
        - 'superspike': SuperSpike 1/(1+|u|)² (Zenke & Ganguli 2018)
        - 'slayer': Exponential PDF (Shrestha & Orchard 2018)
    """

    def __init__(
        self,
        n_neurons: int,
        params: Optional[SpikingParams] = None,
        learnable_tau: bool = True,
        learnable_threshold: bool = True,
        surrogate_method: str = 'soft',
    ):
        """
        Args:
            n_neurons: Number of neurons in population
            params: SpikingParams (uses defaults if None)
            learnable_tau: If True, time constant is learned per neuron type
                (one shared scalar tau_m for the whole population)
            learnable_threshold: If True, threshold is learned per neuron
                (a length-n_neurons vector)
            surrogate_method: 'soft', 'superspike', or 'slayer'

        Side effects:
            Registers learnable nn.Parameters (log_tau_m, v_th) and fixed buffers
            (v_reset, v_rest, v_min, R_m, dt, tau_ref). Parameters vs buffers is
            controlled by the learnable_* flags.
        """
        super().__init__()

        self.n_neurons = n_neurons
        self.params = params or SpikingParams()
        self.surrogate_method = surrogate_method

        # Learnable time constant (log-domain for positivity)
        # Stored as log(tau_m) so the optimizer is unconstrained while tau_m = exp(.) > 0.
        if learnable_tau:
            self.log_tau_m = nn.Parameter(torch.log(torch.tensor(self.params.tau_m)))
        else:
            # Non-learnable: stored as a buffer (moves with .to(device), not optimized).
            self.register_buffer('log_tau_m', torch.log(torch.tensor(self.params.tau_m)))

        # Learnable threshold per neuron (one v_th per cell, init to the shared default).
        if learnable_threshold:
            self.v_th = nn.Parameter(torch.full((n_neurons,), self.params.v_th))
        else:
            self.register_buffer('v_th', torch.full((n_neurons,), self.params.v_th))

        # Fixed parameters (never trained): registered as buffers for device handling.
        self.register_buffer('v_reset', torch.tensor(self.params.v_reset))
        self.register_buffer('v_rest', torch.tensor(self.params.v_rest))
        self.register_buffer('v_min', torch.tensor(self.params.v_min))
        self.register_buffer('R_m', torch.tensor(self.params.R_m))
        self.register_buffer('dt', torch.tensor(self.params.dt))
        self.register_buffer('tau_ref', torch.tensor(self.params.tau_ref))

    @property
    def tau_m(self) -> torch.Tensor:
        # Realized membrane time constant (s): exponentiate the log-space parameter.
        return torch.exp(self.log_tau_m)

    def forward(
        self,
        current: torch.Tensor,
        v: torch.Tensor,
        refr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One timestep (dt) of LIF dynamics for the whole population.

        Args:
            current: (batch, n_neurons) input current (A)
            v: (batch, n_neurons) membrane voltage (V)
            refr: (batch, n_neurons) refractory timer (s); seconds remaining in refractory period

        Returns:
            v_new: (batch, n_neurons) updated membrane voltage (V)
            spikes: (batch, n_neurons) spike tensor (binary at eval, graded/surrogate during training)
            refr_new: (batch, n_neurons) updated refractory timer (s)
        """
        dt = self.dt
        tau_m = self.tau_m

        # Decrement refractory timer toward 0; neurons with timer <= 0 are eligible to fire.
        refr_new = torch.clamp(refr - dt, min=0.0)
        can_fire = refr_new <= 0.0

        # [Noise 2] Background synaptic bombardment: additive current noise (5 pA std)
        # modeling ongoing input from the rest of the brain.
        if self.params.circuit_noise_enabled:
            current = current + torch.randn_like(current) * self.params.i_noise_std

        # Exact exponential integration of  tau_m dV/dt = -(V - V_rest) + R_m*I
        # over one dt with I held constant. The closed-form solution is
        #   V(t+dt) = V_inf + (V(t) - V_inf) * exp(-dt/tau_m)
        # where V_inf is the steady-state the membrane relaxes toward.
        v_inf = self.v_rest + self.R_m * current      # steady-state voltage (V) for this input current
        alpha = torch.exp(-dt / tau_m)                # leak decay factor in (0,1); fraction of distance-to-V_inf retained
        v_new = v_inf + (v - v_inf) * alpha           # exponential relaxation toward V_inf

        # [Noise 1] Membrane voltage noise (stochastic ion channel gating): additive 0.5 mV std.
        if self.params.circuit_noise_enabled:
            v_new = v_new + torch.randn_like(v_new) * self.params.v_noise_std

        # Clamp voltage to a hard floor (-80 mV) to prevent runaway hyperpolarization.
        v_new = torch.clamp(v_new, min=self.v_min)

        # Neurons still in their refractory period are pinned at v_reset (cannot integrate).
        v_new = torch.where(can_fire, v_new, self.v_reset.expand_as(v_new))

        # Spike detection with [Noise 4] threshold jitter: perturb v_th by 0.2 mV std this step.
        if self.params.circuit_noise_enabled:
            v_th_jitter = torch.randn(v_new.shape, device=v_new.device, dtype=v_new.dtype) * self.params.threshold_jitter_std
            v_centered = v_new - (self.v_th + v_th_jitter)   # (v - v_th); spike where > 0
        else:
            v_centered = v_new - self.v_th

        # Use surrogate gradient during training (so backprop can flow), hard spikes at eval.
        if self.training:
            if self.surrogate_method == 'soft':
                # Soft sigmoid (continuous relaxation): graded output, fully differentiable.
                spikes = soft_spike(v_centered, temperature=1e-3)
            elif self.surrogate_method == 'superspike':
                # SuperSpike: hard forward spike, smooth polynomial backward gradient.
                spikes = surrogate_spike(v_centered, scale=100.0)
            elif self.surrogate_method == 'slayer':
                # SLAYER: hard forward spike, exponential-PDF backward gradient.
                spikes = slayer_spike(v_centered, tau_pdf=0.010, alpha=1.0)
            else:
                raise ValueError(f"Unknown surrogate method: {self.surrogate_method}")
        else:
            # Hard spikes for evaluation (deterministic Heaviside).
            spikes = (v_centered > 0).float()

        # Mask out refractory neurons (can_fire is boolean) - they cannot emit spikes.
        spikes = spikes * can_fire.float()

        # Reset spiking neurons (using threshold 0.5 to binarize soft/graded spikes for the reset).
        spike_mask = spikes > 0.5
        v_new = torch.where(spike_mask, self.v_reset.expand_as(v_new), v_new)          # reset voltage to v_reset
        refr_new = torch.where(spike_mask, self.tau_ref.expand_as(refr_new), refr_new) # start refractory countdown

        return v_new, spikes, refr_new

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize membrane voltage and refractory timer.

        Returns (v, refr): v starts at v_reset (resting), refr starts at 0
        (no neuron is refractory). Shapes are (batch_size, n_neurons).
        """
        v = torch.full((batch_size, self.n_neurons), self.params.v_reset, device=device)
        refr = torch.zeros(batch_size, self.n_neurons, device=device)
        return v, refr

    def clamp_to_biological_bounds(self):
        """Clamp all learnable parameters to biological bounds (in place).

        Called after each optimizer step. Only clamps fields that are actually
        nn.Parameters (i.e. were created learnable); buffers are left untouched.
        """
        with torch.no_grad():
            # Clamp v_th to biological range (-55..-30 mV).
            if isinstance(self.v_th, nn.Parameter):
                self.v_th.clamp_(V_TH_MIN, V_TH_MAX)
            # Clamp tau_m (via its log) to 5..50 ms.
            if isinstance(self.log_tau_m, nn.Parameter):
                self.log_tau_m.clamp_(LOG_TAU_M_MIN, LOG_TAU_M_MAX)


class TwoCompartmentKC(nn.Module):
    """
    Two-compartment Kenyon Cell model with dendrite and axon/soma.

    Models a population of KCs each split into two electrically-coupled
    compartments: a dendrite (calyx) that receives synaptic input, and an
    axon/soma that generates the spike. The two compartments are coupled by a
    learnable soma conductance g_soma. This captures the biology that KC
    dendrites can integrate/decay graded signals separately from the
    spike-initiation zone.

    Based on brian2_like.py KC implementation:
    - Dendrite receives synaptic input
    - Axon/soma is the spike-generating compartment
    - Compartments coupled by soma conductance g_soma (NOW LEARNABLE)

    Uses exact matrix exponential integration (computed dynamically with learnable g_soma).
    Because g_soma is learnable, the 2x2 integration matrices cannot be precomputed
    once; they are rebuilt every forward() from the current g_soma (see
    _compute_integration_matrices).

    Equations:
        C_m dV_d/dt = -g_L(V_d - V_rest) + g_soma(V_a - V_d) + I_syn
        C_m dV_a/dt = -g_L(V_a - V_rest) + g_soma(V_d - V_a)
        spike when V_a > V_th

    where g_L = C_m / tau_m (leak conductance)

    Note: KC dendrites in insects can have graded potentials and local
    processing. The axon hillock is the spike initiation zone.

    g_soma is learnable with biological bounds (default 1-100 nS, narrowed to 1-20 nS during training).
    """

    def __init__(
        self,
        n_kc: int,
        params: Optional[SpikingParams] = None,
        learnable_threshold: bool = True,
        learnable_g_soma: bool = True,
        surrogate_method: str = 'soft',
    ):
        """
        Args:
            n_kc: Number of Kenyon Cells
            params: SpikingParams (uses defaults if None)
            learnable_threshold: If True, KC thresholds are learnable (per-KC vector)
            learnable_g_soma: If True, soma conductance is learnable (RECOMMENDED) -
                stored as a single shared scalar in log space
            surrogate_method: 'soft', 'superspike', or 'slayer'

        Side effects:
            Registers learnable parameters (v_th, log_g_soma) and fixed buffers
            for the biophysical constants (including C_m derived from tau_m/R_m).
        """
        super().__init__()

        self.n_kc = n_kc
        self.params = params or SpikingParams()
        self.surrogate_method = surrogate_method
        self.learnable_g_soma = learnable_g_soma

        # Learnable threshold per KC (one v_th per cell).
        if learnable_threshold:
            self.v_th = nn.Parameter(torch.full((n_kc,), self.params.v_th))
        else:
            self.register_buffer('v_th', torch.full((n_kc,), self.params.v_th))

        # Learnable soma conductance (log-domain for positivity)
        # This is critical for 2-compartment KC to train properly!
        # Couples dendrite<->axon; in log space so g_soma = exp(.) > 0.
        if learnable_g_soma:
            self.log_g_soma = nn.Parameter(torch.log(torch.tensor(self.params.g_soma)))
        else:
            self.register_buffer('log_g_soma', torch.log(torch.tensor(self.params.g_soma)))

        # Fixed parameters (buffers; not trained).
        self.register_buffer('v_reset', torch.tensor(self.params.v_reset))
        self.register_buffer('v_rest', torch.tensor(self.params.v_rest))
        self.register_buffer('v_min', torch.tensor(self.params.v_min))
        self.register_buffer('dt', torch.tensor(self.params.dt))
        self.register_buffer('tau_ref', torch.tensor(self.params.tau_ref))
        self.register_buffer('tau_m', torch.tensor(self.params.tau_m))
        self.register_buffer('R_m', torch.tensor(self.params.R_m))
        self.register_buffer('C_m', torch.tensor(self.params.C_m))  # membrane capacitance (F), tau_m/R_m

    @property
    def g_soma(self) -> torch.Tensor:
        """Get soma conductance (S) from log-domain parameter."""
        return torch.exp(self.log_g_soma)

    def _compute_integration_matrices(self) -> Dict[str, torch.Tensor]:
        """
        Compute matrix-exponential coefficients for exact integration of the
        coupled 2-compartment system over one dt.

        The two-compartment ODE is a linear system dV/dt = A V + b. Over one dt
        with constant forcing, the exact update is
            V(t+dt) = M @ V(t) + K @ b
        where M = exp(A*dt) is the state-transition matrix and K = A^{-1}(M - I)
        maps the forcing terms. Because the system is symmetric (both compartments
        share leak and coupling), A has eigenvalues -1/tau and -(1/tau + 2*gs),
        giving the closed-form e1/e2 below and the symmetric M (M11=M22, M12=M21).

        Now computed dynamically (every forward) to support learnable g_soma.
        Following brian2_like.py implementation exactly.

        Returns:
            Dict of scalar tensors M11,M12,M21,M22 (state-transition entries),
            K11..K22 (forcing/input matrix entries), and b0,bI (driving-term
            coefficients used to build the per-step forcing).
        """
        dt = self.dt
        tau = self.tau_m
        C_m = self.C_m
        gs = self.g_soma / C_m  # Normalized soma conductance (1/s); coupling rate between compartments

        # Eigenvalue decomposition of 2x2 coupled system:
        # e1 = decay of the common mode (both compartments move together), rate 1/tau.
        # e2 = decay of the differential mode (compartments differ), rate 1/tau + 2*gs.
        e1 = torch.exp(-dt / tau)
        e2 = torch.exp(-dt * (1.0 / tau + 2.0 * gs))

        # State transition matrix M = exp(A*dt), expressed in the original
        # (dendrite, axon) basis. Symmetric because the two compartments are
        # interchangeable in the leak+coupling structure.
        M11 = 0.5 * (e1 + e2)   # self-coupling (each compartment to itself)
        M12 = 0.5 * (e1 - e2)   # cross-coupling (dendrite<->axon mixing per step)
        M21 = M12
        M22 = M11

        # Input matrix K (for forcing terms) = A^{-1} (M - I), built from the 2x2
        # generator A whose diagonal is a = -(1/tau + gs) and off-diagonal is gs.
        a = -(1.0 / tau + gs)   # diagonal entry of generator A (1/s)
        off = gs                # off-diagonal coupling entry of A (1/s)
        den = (a * a - off * off) + _EPS  # det(A); add epsilon for numerical stability (avoid div-by-zero)

        # Closed-form entries of K = A^{-1}(M - I), derived per row/column.
        K11 = (a * (M11 - 1.0) - off * M21) / den
        K12 = (a * M12 - off * (M22 - 1.0)) / den
        K21 = (-off * (M11 - 1.0) + a * M21) / den
        K22 = (-off * M12 + a * (M22 - 1.0)) / den

        # Driving term coefficients that turn (reset/leak + input current) into forcing:
        # the constant leak drive (b0) and the per-amp current scaling (bI).
        b0 = self.v_reset / tau   # constant leak/rest forcing term (V/s)
        bI = self.R_m / tau       # current-to-forcing gain (V/(s·A)); = 1/C_m since tau=R_m*C_m

        return {
            'M11': M11, 'M12': M12, 'M21': M21, 'M22': M22,
            'K11': K11, 'K12': K12, 'K21': K21, 'K22': K22,
            'b0': b0, 'bI': bI,
        }

    def forward(
        self,
        I_syn: torch.Tensor,
        v_d: torch.Tensor,
        v_a: torch.Tensor,
        refr: torch.Tensor,
        I_axon: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One timestep (dt) of 2-compartment KC dynamics.

        Args:
            I_syn: (batch, n_kc) synaptic current to dendrite (A)
            v_d: (batch, n_kc) dendritic voltage (V)
            v_a: (batch, n_kc) axonal voltage (V)
            refr: (batch, n_kc) refractory timer (s)
            I_axon: (batch, n_kc) optional synaptic current to axon (A),
                    e.g. from KC-KC axon→axon recurrent connections

        Returns:
            v_d_new: (batch, n_kc) updated dendritic voltage (V)
            v_a_new: (batch, n_kc) updated axonal voltage (V)
            spikes: (batch, n_kc) spike tensor (axon-driven)
            refr_new: (batch, n_kc) updated refractory timer (s)
        """
        dt = self.dt

        # Decrement refractory timer; axon may only fire when timer has elapsed.
        refr_new = torch.clamp(refr - dt, min=0.0)
        can_fire = refr_new <= 0.0

        # Compute integration matrices dynamically (for learnable g_soma) every step.
        matrices = self._compute_integration_matrices()

        # Driving terms: bd (dendrite forcing = leak drive + scaled synaptic input),
        # ba (axon forcing = leak drive, plus optional axonal input current).
        bd = matrices['b0'] + matrices['bI'] * I_syn
        if I_axon is not None:
            ba = matrices['b0'] + matrices['bI'] * I_axon
        else:
            # No direct axonal input: axon forcing is just the constant leak drive.
            ba = matrices['b0'].expand_as(I_syn)

        # Exact matrix exponential update:  V(t+dt) = M @ V(t) + K @ b
        # Md/Ma are the state-transition (homogeneous) part; Kd/Ka the forced part.
        Md = matrices['M11'] * v_d + matrices['M12'] * v_a   # dendrite from prior (v_d, v_a)
        Ma = matrices['M21'] * v_d + matrices['M22'] * v_a   # axon from prior (v_d, v_a)
        Kd = matrices['K11'] * bd + matrices['K12'] * ba     # dendrite forced response to (bd, ba)
        Ka = matrices['K21'] * bd + matrices['K22'] * ba     # axon forced response to (bd, ba)

        v_d_new = Md + Kd
        v_a_new = Ma + Ka

        # [Noise 1] Membrane voltage noise (stochastic ion channel gating) on both compartments.
        if self.params.circuit_noise_enabled:
            v_d_new = v_d_new + torch.randn_like(v_d_new) * self.params.v_noise_std
            v_a_new = v_a_new + torch.randn_like(v_a_new) * self.params.v_noise_std

        # Clamp voltages to biological range (hard floor at v_min = -80 mV).
        v_d_new = torch.clamp(v_d_new, min=self.v_min)
        v_a_new = torch.clamp(v_a_new, min=self.v_min)

        # Pin the axon of still-refractory KCs at v_reset (cannot fire this step).
        v_a_new = torch.where(can_fire, v_a_new, self.v_reset.expand_as(v_a_new))

        # Spike detection (axon compartment is the spike-initiation zone) with [Noise 4] threshold jitter.
        if self.params.circuit_noise_enabled:
            v_th_jitter = torch.randn(v_a_new.shape, device=v_a_new.device, dtype=v_a_new.dtype) * self.params.threshold_jitter_std
            v_centered = v_a_new - (self.v_th + v_th_jitter)
        else:
            v_centered = v_a_new - self.v_th

        # Use surrogate gradient during training, hard spikes at eval (same scheme as LIFNeuron).
        if self.training:
            if self.surrogate_method == 'soft':
                spikes = soft_spike(v_centered, temperature=1e-3)
            elif self.surrogate_method == 'superspike':
                spikes = surrogate_spike(v_centered, scale=100.0)
            elif self.surrogate_method == 'slayer':
                spikes = slayer_spike(v_centered, tau_pdf=0.010, alpha=1.0)
            else:
                raise ValueError(f"Unknown surrogate method: {self.surrogate_method}")
        else:
            # Hard spikes for evaluation.
            spikes = (v_centered > 0).float()

        # Mask out refractory neurons.
        spikes = spikes * can_fire.float()

        # Reset BOTH compartments on a spike (binarize soft spikes at 0.5).
        spike_mask = spikes > 0.5
        v_d_new = torch.where(spike_mask, self.v_reset.expand_as(v_d_new), v_d_new)
        v_a_new = torch.where(spike_mask, self.v_reset.expand_as(v_a_new), v_a_new)
        refr_new = torch.where(spike_mask, self.tau_ref.expand_as(refr_new), refr_new)

        return v_d_new, v_a_new, spikes, refr_new

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize dendritic, axonal voltages and refractory timer.

        Returns (v_d, v_a, refr): both compartments start at v_reset, refr at 0.
        Shapes (batch_size, n_kc).
        """
        v_d = torch.full((batch_size, self.n_kc), self.params.v_reset, device=device)
        v_a = torch.full((batch_size, self.n_kc), self.params.v_reset, device=device)
        refr = torch.zeros(batch_size, self.n_kc, device=device)
        return v_d, v_a, refr

    def clamp_to_biological_bounds(self):
        """Clamp all learnable parameters to biological bounds (in place)."""
        with torch.no_grad():
            # Clamp v_th to -55..-30 mV.
            if isinstance(self.v_th, nn.Parameter):
                self.v_th.clamp_(V_TH_MIN, V_TH_MAX)
            # Clamp g_soma (via its log) to 1..100 nS.
            if isinstance(self.log_g_soma, nn.Parameter):
                self.log_g_soma.clamp_(LOG_G_SOMA_MIN, LOG_G_SOMA_MAX)


class SpikingConnectomeLinear(nn.Module):
    """
    Connectome-constrained synaptic connection layer (one pathway).

    Represents a single directed pathway (e.g. ORN→PN). The WHO-connects-to-WHOM
    structure and per-pair synapse counts come from the connectome and are FIXED;
    only a single scalar unitary strength s (plus STD params) is learned. The
    effective weight matrix is therefore strength × normalized-counts × mask, so
    the connectome's relative connection weights are preserved while one knob
    scales the whole pathway.

    Maps presynaptic spike counts to postsynaptic current using:
        I_post = W * spike_count
        W_ij = s * N_ij * mask_ij

    where:
    - s: learned unitary synapse strength (single scalar)
    - N_ij: synapse count from connectome (fixed)
    - mask_ij: binary connectivity mask

    Also implements:
    - synaptic current dynamics (exponential decay with tau_syn), and
    - optional Tsodyks-Markram short-term depression (STD) of the synapse.
    """

    def __init__(
        self,
        synapse_counts: torch.Tensor,
        params: Optional[SpikingParams] = None,
        init_strength: float = 5e-9,
        sign: str = "excitatory",
    ):
        """
        Args:
            synapse_counts: (n_pre, n_post) synapse counts from connectome
            params: SpikingParams
            init_strength: Initial unitary synapse strength (A)
            sign: "excitatory" (positive weights) or "inhibitory" (negative weights)

        Side effects:
            Registers fixed connectome buffers (synapse_counts, mask, norm_counts,
            alpha_syn) and learnable parameters (log_strength, log_tau_rec, logit_U).
        """
        super().__init__()

        self.params = params or SpikingParams()
        self.sign = sign
        n_pre, n_post = synapse_counts.shape
        self.n_pre = n_pre
        self.n_post = n_post

        # Fixed connectivity from connectome (stored as float buffer).
        self.register_buffer('synapse_counts', synapse_counts.float())
        # Binary connectivity mask: 1 where a connection exists (count > 0), else 0.
        mask = (synapse_counts > 0).float()
        self.register_buffer('mask', mask)

        # Normalize synapse counts to [0,1] by the max count, so the learned scalar
        # `strength` carries the absolute scale while norm_counts encode relative weighting.
        max_count = synapse_counts.max()
        if max_count > 0:
            self.register_buffer('norm_counts', synapse_counts.float() / max_count)
        else:
            # Degenerate empty pathway: keep zeros (avoids 0/0).
            self.register_buffer('norm_counts', synapse_counts.float())

        # Learnable unitary synapse strength (A), stored in log space for positivity.
        self.log_strength = nn.Parameter(torch.log(torch.tensor(init_strength)))

        # Short-term synaptic depression (Tsodyks-Markram model)
        # tau_rec: vesicle recovery time constant (biological range ~50-500 ms)
        # U: release probability per spike (biological range ~0.1-0.6)
        # Both stored in unconstrained space (log / logit) and mapped back on read.
        self.log_tau_rec = nn.Parameter(torch.log(torch.tensor(200e-3)))  # 200 ms
        self.logit_U = nn.Parameter(torch.logit(torch.tensor(0.3)))      # 0.3

        # Synaptic current decay factor per dt: exp(-dt/tau_syn). Postsynaptic
        # current decays exponentially with tau_syn between input events.
        self.register_buffer('alpha_syn',
                            torch.tensor(np.exp(-self.params.dt / self.params.tau_syn)))

    @property
    def strength(self) -> torch.Tensor:
        """Get positive synapse strength (A) from the log-space parameter."""
        return torch.exp(self.log_strength)

    @property
    def tau_rec(self) -> torch.Tensor:
        """STD vesicle recovery time constant (s)."""
        return torch.exp(self.log_tau_rec)

    @property
    def U_std(self) -> torch.Tensor:
        """STD release probability per spike (dimensionless, in (0,1))."""
        return torch.sigmoid(self.logit_U)

    def get_weights(self) -> torch.Tensor:
        """Return effective weight matrix W = s * norm_counts * mask.

        Sign is applied here: inhibitory pathways negate the (positive) strength,
        so matmul with non-negative spike counts yields a negative (hyperpolarizing)
        current. Shape (n_pre, n_post), units A per unit spike count.
        """
        s = self.strength
        if self.sign == "inhibitory":
            s = -s
        return s * self.norm_counts * self.mask

    def forward(
        self,
        spikes: torch.Tensor,
        I_syn: torch.Tensor,
        x_std: Optional[torch.Tensor] = None,
    ):
        """
        Update synaptic current given presynaptic spikes (one dt).

        Args:
            spikes: (batch, n_pre) presynaptic spike counts
            I_syn: (batch, n_post) current synaptic current (A)
            x_std: (batch, n_pre) available vesicle fraction for STD.
                   If provided, applies Tsodyks-Markram depression and
                   returns (I_syn_new, x_std_new). If None, no STD.

        Returns:
            I_syn_new if x_std is None, else (I_syn_new, x_std_new)
        """
        W = self.get_weights()

        if x_std is not None:
            # Short-term depression (Tsodyks-Markram), three steps per timestep:
            # 1. Recovery: available vesicle fraction x relaxes back toward 1.0 with
            #    time constant tau_rec.  x <- 1 - (1 - x)*exp(-dt/tau_rec)
            alpha_rec = torch.exp(-self.params.dt / self.tau_rec)
            x_std = 1.0 - (1.0 - x_std) * alpha_rec
            # 2. Effective current: each spike releases a fraction U of the currently
            #    available resources x, so the transmitted "effective spike" is U*x*spike.
            U = self.U_std
            I_inject = torch.matmul(spikes * U * x_std, W)
            # 3. Depletion: spikes consume vesicles, reducing x by the released fraction
            #    U (per spike, clamped to a single release event so counts>1 don't over-deplete).
            x_std = x_std * (1.0 - U * spikes.clamp(0, 1))
        else:
            # No STD: linear synapse, injected current = spikes @ W.
            I_inject = torch.matmul(spikes, W)

        # [Noise 3] Synaptic release stochasticity (probabilistic vesicle release):
        # multiplicative Gaussian noise (15% CV) on the injected current.
        if self.params.circuit_noise_enabled:
            I_inject = I_inject * (1.0 + torch.randn_like(I_inject) * self.params.syn_noise_std)

        # Exponential decay of existing current plus newly injected current.
        I_syn_new = I_syn * self.alpha_syn + I_inject

        if x_std is not None:
            return I_syn_new, x_std
        return I_syn_new

    def init_current(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize synaptic current to zero. Shape (batch, n_post), units A."""
        return torch.zeros(batch_size, self.n_post, device=device)

    def init_std_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize STD vesicle pool to full (x=1.0). Shape (batch, n_pre)."""
        return torch.ones(batch_size, self.n_pre, device=device)

    def clamp_to_biological_bounds(self):
        """Clamp synaptic strength and STD params to biological bounds (in place)."""
        with torch.no_grad():
            # Strength to 1 pA .. 100 nA (log space).
            self.log_strength.clamp_(LOG_STRENGTH_MIN, LOG_STRENGTH_MAX)
            # STD: tau_rec in [20 ms, 2 s], U in [0.05, 0.8]
            self.log_tau_rec.clamp_(np.log(20e-3), np.log(2.0))
            # Clamp logit_U so the realized release probability stays in [0.05, 0.8].
            self.logit_U.clamp_(torch.logit(torch.tensor(0.05)).item(),
                                torch.logit(torch.tensor(0.8)).item())


class SpikingAntennalLobe(nn.Module):
    """
    Spiking Antennal Lobe: ORN → PN with LN lateral inhibition/excitation.

    Assembles the first processing stage of the pipeline. It owns three LIF
    populations (ORN, LN, PN), the connectome-constrained chemical synapses
    between them, three families of electrical (gap-junction) coupling, and the
    "non-AD" weaker contact variants. Its forward() runs the AL time loop and
    returns PN spike counts over the window.

    Components:
    - ORN: LIF neurons receiving OR input (graded → spike conversion)
    - LN: LIF interneurons, split into inhibitory (Broad/Choosy, GABAergic) and
          excitatory (Picky, glutamatergic) subtypes by fan-out
    - PN: LIF projection neurons

    Pathway:
        OR (rate) → ORN (LIF) → LN (LIF) → PN (LIF)
                            ↓           ↓ (inhibition)
                            └───────────→ PN
    """

    def __init__(
        self,
        orn_to_pn: torch.Tensor,
        ln_to_pn: torch.Tensor,
        orn_to_ln: Optional[torch.Tensor] = None,
        ln_to_ln: Optional[torch.Tensor] = None,
        pn_to_ln: Optional[torch.Tensor] = None,
        ln_to_orn: Optional[torch.Tensor] = None,
        params: Optional[SpikingParams] = None,
        # Non-ad compartment connections (separate from canonical ad synapses)
        orn_to_ln_nonad: Optional[torch.Tensor] = None,
        ln_to_pn_nonad: Optional[torch.Tensor] = None,
        ln_to_ln_nonad: Optional[torch.Tensor] = None,
        pn_to_ln_nonad: Optional[torch.Tensor] = None,
        ln_to_orn_nonad: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            orn_to_pn: (n_orn, n_pn) synapse counts
            ln_to_pn: (n_ln, n_pn) synapse counts (inhibitory)
            orn_to_ln: (n_orn, n_ln) synapse counts (optional)
            ln_to_ln: (n_ln, n_ln) optional LN lateral inhibition synapse counts
            pn_to_ln: (n_pn, n_ln) optional PN→LN feedback synapse counts
            ln_to_orn: (n_ln, n_orn) optional LN→ORN feedback synapse counts
            params: SpikingParams
            orn_to_ln_nonad: non-ad compartment ORN→LN (optional)
            ln_to_pn_nonad: non-ad compartment LN→PN (optional)
            ln_to_ln_nonad: non-ad compartment LN→LN (optional)
            pn_to_ln_nonad: non-ad compartment PN→LN (optional)
            ln_to_orn_nonad: non-ad compartment LN→ORN (optional)

        Side effects:
            Constructs all sub-layers, classifies LNs into excitatory/inhibitory,
            builds the three gap-junction masks, and PRINTS a summary of subtype
            counts and connection counts as it builds each pathway.
        """
        super().__init__()

        self.params = params or SpikingParams()

        # Population sizes are inferred from the connectome matrix shapes.
        n_orn = orn_to_pn.shape[0]
        n_pn = orn_to_pn.shape[1]
        n_ln = ln_to_pn.shape[0]

        self.n_orn = n_orn
        self.n_pn = n_pn
        self.n_ln = n_ln

        # LIF neurons for the three AL populations.
        self.orn_neurons = LIFNeuron(n_orn, params)
        self.ln_neurons = LIFNeuron(n_ln, params)
        self.pn_neurons = LIFNeuron(n_pn, params)

        # Synaptic connections (feedforward): ORN→PN excitation.
        self.orn_pn = SpikingConnectomeLinear(orn_to_pn, params, sign="excitatory")

        # === EXCITATORY LN SUBTYPES (Berck 2016) ===
        # Classify LNs by fan-out: low fan-out = Picky LNs = glutamatergic (excitatory)
        # High fan-out = Broad/Choosy LNs = GABAergic (inhibitory)
        # IMPORTANT: Only classify LNs that have LN→PN connections (many LNs have zero)
        fan_out = (ln_to_pn > 0).sum(dim=1)            # number of PN targets per LN
        has_pn_connections = fan_out > 0               # LNs that actually project to PNs
        n_connected = has_pn_connections.sum().item()
        if n_connected > 0:
            connected_fan_out = fan_out[has_pn_connections].float()
            # Bottom-third fan-out among connected LNs is labeled excitatory (Picky).
            threshold = torch.quantile(connected_fan_out, 0.33)
            is_excitatory_ln = has_pn_connections & (fan_out <= threshold)
        else:
            # No connected LNs at all: treat none as excitatory.
            is_excitatory_ln = torch.zeros(n_ln, dtype=torch.bool)
        self.register_buffer('is_excitatory_ln', is_excitatory_ln)
        n_excit = is_excitatory_ln.sum().item()
        n_inhib = (has_pn_connections & ~is_excitatory_ln).sum().item()
        n_silent = (~has_pn_connections).sum().item()
        _blog(f"  LN subtypes: {n_inhib} inhibitory (Broad/Choosy), {n_excit} excitatory (Picky), {n_silent} no LN→PN")

        # Split LN→PN into excitatory and inhibitory pathways by zeroing the rows
        # of the opposite subtype (so each pathway only sees its own LNs).
        ln_pn_inhib = ln_to_pn.clone()
        ln_pn_inhib[is_excitatory_ln] = 0
        ln_pn_excit = ln_to_pn.clone()
        ln_pn_excit[~is_excitatory_ln] = 0

        self.ln_pn = SpikingConnectomeLinear(ln_pn_inhib, params, sign="inhibitory")
        self.ln_pn_excit = SpikingConnectomeLinear(ln_pn_excit, params, sign="excitatory")

        # --- Build-banner: canonical (AD) AL connections ---
        # edges    = connected pre->post pairs (nonzero matrix entries)
        # synapses = total synapse count (matrix sum); the connectome stores synapse
        # counts, so one edge can carry many synapses (the two numbers differ).
        def _es(W):  # -> (edges, synapses)
            return int((W > 0).sum()), int(W.sum())
        _e, _s = _es(orn_to_pn); _blog(f"  ORN→PN (AD): {_e} edges ({_s} synapses)")
        if orn_to_ln is not None and orn_to_ln.sum() > 0:
            _e, _s = _es(orn_to_ln); _blog(f"  ORN→LN (AD): {_e} edges ({_s} synapses)")
        _ei, _si = _es(ln_pn_inhib); _ee, _se = _es(ln_pn_excit)
        _blog(f"  LN→PN (AD): {_ei} inhib + {_ee} excit edges ({_si + _se} synapses)")
        if ln_to_ln is not None and ln_to_ln.sum() > 0:
            _e, _s = _es(ln_to_ln); _blog(f"  LN→LN (AD): {_e} edges ({_s} synapses)")
        if pn_to_ln is not None and pn_to_ln.sum() > 0:
            _e, _s = _es(pn_to_ln); _blog(f"  PN→LN (AD): {_e} edges ({_s} synapses)")

        # === GAP JUNCTIONS (3 types) ===
        # Gap junctions are electrical synapses: current flows instantaneously and
        # is proportional to the voltage difference, I = g * (V_pre - V_post),
        # summed over coupled partners. Each family below has one learnable
        # conductance g (log space) and a fixed binary coupling mask.

        # A1: LN-LN gap junctions — symmetrized chemical synapse proxy (ShakB/Inx7)
        if ln_to_ln is not None:
            # Symmetrize (gap junctions are bidirectional) and remove self-coupling.
            gap_ln = ((ln_to_ln + ln_to_ln.T) > 0).float()
            gap_ln.fill_diagonal_(0)
            self.register_buffer('gap_ln_ln_mask', gap_ln)
            self.log_g_gap_ln = nn.Parameter(torch.tensor(np.log(1e-10)))  # 0.1 nS init, log space
            n_gap_ln = int(gap_ln.sum().item()) // 2   # //2 because each pair is counted twice
            _blog(f"  Gap LN-LN: {n_gap_ln} pairs")
        else:
            self.register_buffer('gap_ln_ln_mask', None)
            self.log_g_gap_ln = None

        # A2: PN-PN gap junctions — sister PNs (shared ORN input proxy, ShakB/Inx7)
        # Two PNs are coupled if they share at least one common ORN input; the
        # Gram matrix orn_to_pn.T @ orn_to_pn has positive (i,j) iff PNs i and j
        # share ORN drive, which is used as a "sister PN" proxy for coupling.
        shared_input = (orn_to_pn.T @ orn_to_pn)  # (n_pn, n_pn)
        gap_pn = (shared_input > 0).float()
        gap_pn.fill_diagonal_(0)
        self.register_buffer('gap_pn_pn_mask', gap_pn)
        self.log_g_gap_pn = nn.Parameter(torch.tensor(np.log(1e-10)))  # 0.1 nS init, log space
        n_gap_pn = int(gap_pn.sum().item()) // 2
        _blog(f"  Gap PN-PN: {n_gap_pn} pairs (sister PNs)")

        # A3: eLN-PN gap junctions — excitatory LNs electrically coupled to PNs (ShakB)
        # Built from the excitatory-LN rows of ln_to_pn (binarized): an asymmetric
        # (n_ln, n_pn) mask that is used in both directions in forward().
        gap_eln_pn = ln_to_pn.clone().float()
        gap_eln_pn[~is_excitatory_ln] = 0  # only excitatory LN rows
        gap_eln_pn = (gap_eln_pn > 0).float()  # binary mask
        self.register_buffer('gap_eln_pn_mask', gap_eln_pn)  # (n_ln, n_pn)
        self.log_g_gap_eln_pn = nn.Parameter(torch.tensor(np.log(1e-10)))  # 0.1 nS init, log space
        n_gap_eln_pn = int(gap_eln_pn.sum().item())
        _blog(f"  Gap eLN-PN: {n_gap_eln_pn} connections")

        if orn_to_ln is not None:
            self.orn_ln = SpikingConnectomeLinear(orn_to_ln, params, sign="excitatory")
        else:
            # Simple pooling if no specific connectivity: mean ORN activity broadcast to all LNs.
            self.orn_ln = None
            self.ln_pool_weight = nn.Parameter(torch.ones(1) * 0.5)

        # Recurrent/feedback connections (use previous timestep's spikes)
        # LN→LN lateral inhibition (1,716 synapses — mutual inhibition for contrast)
        if ln_to_ln is not None and ln_to_ln.sum() > 0:
            self.ln_ln = SpikingConnectomeLinear(ln_to_ln, params, sign="inhibitory")
        else:
            self.ln_ln = None

        # PN→LN feedback (1,397 synapses — PN activity recruits LN inhibition)
        if pn_to_ln is not None and pn_to_ln.sum() > 0:
            self.pn_ln = SpikingConnectomeLinear(pn_to_ln, params, sign="excitatory")
        else:
            self.pn_ln = None

        # LN→ORN feedback (0 AD synapses in Winding 2023 — loaded for completeness)
        if ln_to_orn is not None and ln_to_orn.sum() > 0:
            self.ln_orn = SpikingConnectomeLinear(ln_to_orn, params, sign="inhibitory")
        else:
            self.ln_orn = None

        # LN→ORN non-AD (310 synapses); weaker init strength than canonical AD synapses.
        if ln_to_orn_nonad is not None and ln_to_orn_nonad.sum() > 0:
            self.ln_orn_nonad = SpikingConnectomeLinear(
                ln_to_orn_nonad, params, init_strength=1e-10, sign="inhibitory")
            _blog(f"  LN→ORN (non-AD): {int((ln_to_orn_nonad > 0).sum())} edges ({int(ln_to_orn_nonad.sum())} synapses)")
        else:
            self.ln_orn_nonad = None

        # === NON-AD COMPARTMENT CONNECTIONS (weaker, non-canonical contacts) ===
        # Non-axon-dendrite connections (aa, da, dd) exist in the connectome but are
        # much weaker than canonical AD synapses. Initialized 50× weaker.
        NONAD_INIT_STRENGTH = 1e-10  # 50× weaker than AD default (5e-9)

        # ORN→LN non-AD (182 synapses, mostly aa)
        if orn_to_ln_nonad is not None and orn_to_ln_nonad.sum() > 0:
            self.orn_ln_nonad = SpikingConnectomeLinear(
                orn_to_ln_nonad, params, init_strength=NONAD_INIT_STRENGTH, sign="excitatory")
            _blog(f"  ORN→LN (non-AD): {int((orn_to_ln_nonad > 0).sum())} edges ({int(orn_to_ln_nonad.sum())} synapses)")
        else:
            self.orn_ln_nonad = None

        # LN→PN non-AD (582 synapses, aa+da+dd) — split by LN subtype (same scheme as canonical LN→PN)
        if ln_to_pn_nonad is not None and ln_to_pn_nonad.sum() > 0:
            ln_pn_nonad_inhib = ln_to_pn_nonad.clone()
            ln_pn_nonad_inhib[is_excitatory_ln] = 0
            ln_pn_nonad_excit = ln_to_pn_nonad.clone()
            ln_pn_nonad_excit[~is_excitatory_ln] = 0
            n_nonad_inhib = int((ln_pn_nonad_inhib > 0).sum())
            n_nonad_excit = int((ln_pn_nonad_excit > 0).sum())
            self.ln_pn_nonad = SpikingConnectomeLinear(
                ln_pn_nonad_inhib, params, init_strength=NONAD_INIT_STRENGTH, sign="inhibitory")
            self.ln_pn_excit_nonad = SpikingConnectomeLinear(
                ln_pn_nonad_excit, params, init_strength=NONAD_INIT_STRENGTH, sign="excitatory")
            _blog(f"  LN→PN (non-AD): {n_nonad_inhib} inhib + {n_nonad_excit} excit edges ({int(ln_to_pn_nonad.sum())} synapses)")
        else:
            self.ln_pn_nonad = None
            self.ln_pn_excit_nonad = None

        # LN→LN non-AD (605 synapses, aa+da+dd)
        if ln_to_ln_nonad is not None and ln_to_ln_nonad.sum() > 0:
            self.ln_ln_nonad = SpikingConnectomeLinear(
                ln_to_ln_nonad, params, init_strength=NONAD_INIT_STRENGTH, sign="inhibitory")
            _blog(f"  LN→LN (non-AD): {int((ln_to_ln_nonad > 0).sum())} edges ({int(ln_to_ln_nonad.sum())} synapses)")
        else:
            self.ln_ln_nonad = None

        # PN→LN non-AD (518 synapses)
        if pn_to_ln_nonad is not None and pn_to_ln_nonad.sum() > 0:
            self.pn_ln_nonad = SpikingConnectomeLinear(
                pn_to_ln_nonad, params, init_strength=NONAD_INIT_STRENGTH, sign="excitatory")
            _blog(f"  PN→LN (non-AD): {int((pn_to_ln_nonad > 0).sum())} edges ({int(pn_to_ln_nonad.sum())} synapses)")
        else:
            self.pn_ln_nonad = None

        # OR → ORN conversion gain (graded to current)
        # Use log-space for positivity constraint (like other learned parameters)
        # To fire: V_inf > V_th → V_rest + R_m*I > V_th → I > (V_th - V_rest)/R_m = 20mV/1GΩ = 20pA
        # For ORN input ~0.3 (normalized), need gain > 20pA/0.3 ≈ 70pA = 7e-11 A
        # Initialize to 5e-10 A for stronger, more reliable firing
        self.log_or_gain = nn.Parameter(torch.log(torch.tensor(5e-10)))

    def forward(
        self,
        or_input: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        n_steps: int = 50,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process OR input through spiking antennal lobe (run the AL time loop).

        Args:
            or_input: (batch, n_or) OR response rates (Hz or normalized)
            state: Previous state dict (or None to initialize)
            n_steps: Number of simulation timesteps. NOTE: the canonical paper
                runtime uses N_STEPS=30 set in the run_*.py drivers; this default
                of 50 is a legacy/standalone default that the scripts override.

        Returns:
            pn_spike_count: (batch, n_pn) PN spike counts accumulated over the window
            state: Updated state dict (voltages, refractory timers, synaptic currents)
        """
        batch_size = or_input.shape[0]
        device = or_input.device

        # Initialize or unpack state
        if state is None:
            state = self._init_state(batch_size, device)

        v_orn, refr_orn = state['v_orn'], state['refr_orn']
        v_ln, refr_ln = state['v_ln'], state['refr_ln']
        v_pn, refr_pn = state['v_pn'], state['refr_pn']
        I_orn_pn = state['I_orn_pn']
        I_orn_ln = state['I_orn_ln']
        I_ln_pn = state['I_ln_pn']
        # Excitatory LN→PN current may be absent in older states; default to zeros.
        I_ln_pn_excit = state.get('I_ln_pn_excit',
                                   torch.zeros(batch_size, self.n_pn, device=device))

        # Convert OR response to ORN input current
        # OR responses are sustained, so we provide constant current
        # Use exp() to get positive gain from log-space parameter
        I_or = or_input * torch.exp(self.log_or_gain)

        # Spike accumulators (summed over the n_steps window).
        pn_spike_count = torch.zeros(batch_size, self.n_pn, device=device)
        orn_spike_count = torch.zeros(batch_size, self.n_orn, device=device)
        ln_spike_count = torch.zeros(batch_size, self.n_ln, device=device)

        for _ in range(n_steps):
            # ORN dynamics: receives constant OR-derived input current.
            v_orn, spk_orn, refr_orn = self.orn_neurons(I_or, v_orn, refr_orn)
            orn_spike_count += spk_orn

            # --- Gap junction currents (instantaneous, voltage-dependent) ---
            # General form for a population coupled by mask G:
            #   I_i = g * sum_j G_ij (V_j - V_i) = g * ((V @ G)_i - V_i * sum_j G_ij)
            # i.e. inflow from coupled partners minus outflow proportional to degree.

            # A1: LN-LN gap junctions
            I_gap_ln = torch.zeros_like(v_ln)
            if self.gap_ln_ln_mask is not None:
                g_ln = torch.exp(self.log_g_gap_ln)
                I_gap_ln = g_ln * (torch.matmul(v_ln, self.gap_ln_ln_mask)
                                    - v_ln * self.gap_ln_ln_mask.sum(1))

            # A2: PN-PN gap junctions (sister PNs)
            g_pn = torch.exp(self.log_g_gap_pn)
            I_gap_pn = g_pn * (torch.matmul(v_pn, self.gap_pn_pn_mask)
                                - v_pn * self.gap_pn_pn_mask.sum(1))

            # A3: eLN-PN gap junctions (bidirectional via the same (n_ln, n_pn) mask)
            g_eln = torch.exp(self.log_g_gap_eln_pn)
            # Current into PNs from eLNs:  g * (V_ln @ mask - V_pn * column_degree)
            I_gap_eln_to_pn = g_eln * (torch.matmul(v_ln, self.gap_eln_pn_mask)
                                        - v_pn * self.gap_eln_pn_mask.sum(0))
            # Current into LNs from PNs (reverse direction, uses mask.T and row degree)
            I_gap_pn_to_eln = g_eln * (torch.matmul(v_pn, self.gap_eln_pn_mask.T)
                                        - v_ln * self.gap_eln_pn_mask.sum(1))

            # LN dynamics: receives ORN input + gap junction currents
            if self.orn_ln is not None:
                I_orn_ln = self.orn_ln(spk_orn, I_orn_ln)
            else:
                # Simple mean pooling fallback when no ORN→LN connectome is provided:
                # broadcast mean ORN activity to all LNs scaled by a positive pool weight.
                I_orn_ln = spk_orn.mean(dim=-1, keepdim=True) * F.softplus(self.ln_pool_weight)
                I_orn_ln = I_orn_ln.expand(-1, self.n_ln)

            # Total LN input = chemical ORN drive + LN-LN gap + reverse eLN-PN gap.
            I_ln_total = I_orn_ln + I_gap_ln + I_gap_pn_to_eln
            v_ln, spk_ln, refr_ln = self.ln_neurons(I_ln_total, v_ln, refr_ln)
            ln_spike_count += spk_ln

            # PN dynamics: receives ORN excitation, split LN inhibition/excitation, gap junctions
            I_orn_pn = self.orn_pn(spk_orn, I_orn_pn)
            I_ln_pn = self.ln_pn(spk_ln, I_ln_pn)              # inhibitory (negative weights)
            I_ln_pn_excit = self.ln_pn_excit(spk_ln, I_ln_pn_excit)  # excitatory (Picky LNs)
            I_pn_gap = I_gap_pn + I_gap_eln_to_pn               # PN-PN + forward eLN-PN gap currents
            I_pn_total = I_orn_pn + I_ln_pn + I_ln_pn_excit + I_pn_gap

            v_pn, spk_pn, refr_pn = self.pn_neurons(I_pn_total, v_pn, refr_pn)
            pn_spike_count += spk_pn

        # Update state (carry voltages/timers/currents to the next call).
        # Note: this sequential AL-only path does not thread STD or all recurrent
        # currents; the biologically-faithful path is model._unified_forward().
        state = {
            'v_orn': v_orn, 'refr_orn': refr_orn,
            'v_ln': v_ln, 'refr_ln': refr_ln,
            'v_pn': v_pn, 'refr_pn': refr_pn,
            'I_orn_pn': I_orn_pn,
            'I_orn_ln': I_orn_ln if self.orn_ln is not None else torch.zeros(1),
            'I_ln_pn': I_ln_pn,
            'I_ln_pn_excit': I_ln_pn_excit,
        }

        return pn_spike_count, state

    def _init_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Initialize all state variables for the antennal lobe.

        Builds a dict of fresh per-population voltages/refractory timers, all
        synaptic currents (zero), STD vesicle pools (full, x=1), and previous-spike
        buffers for recurrent connections. Optional pathways only add their keys
        when the corresponding sub-layer exists. Shapes are (batch, n_*).
        """
        v_orn, refr_orn = self.orn_neurons.init_state(batch_size, device)
        v_ln, refr_ln = self.ln_neurons.init_state(batch_size, device)
        v_pn, refr_pn = self.pn_neurons.init_state(batch_size, device)

        state = {
            'v_orn': v_orn, 'refr_orn': refr_orn,
            'v_ln': v_ln, 'refr_ln': refr_ln,
            'v_pn': v_pn, 'refr_pn': refr_pn,
            'I_orn_pn': self.orn_pn.init_current(batch_size, device),
            'I_orn_ln': self.orn_ln.init_current(batch_size, device) if self.orn_ln else torch.zeros(1, device=device),
            'I_ln_pn': self.ln_pn.init_current(batch_size, device),
            'I_ln_pn_excit': self.ln_pn_excit.init_current(batch_size, device),
            # STD vesicle states (full at start of each stimulus)
            'x_std_orn_pn': self.orn_pn.init_std_state(batch_size, device),
            'x_std_ln_pn': self.ln_pn.init_std_state(batch_size, device),
            'x_std_ln_pn_excit': self.ln_pn_excit.init_std_state(batch_size, device),
            # Previous spikes for recurrent connections
            'spk_ln_prev': torch.zeros(batch_size, self.n_ln, device=device),
            'spk_pn_prev': torch.zeros(batch_size, self.n_pn, device=device),
        }
        if self.orn_ln is not None:
            state['x_std_orn_ln'] = self.orn_ln.init_std_state(batch_size, device)
        # Recurrent connection currents (only present when the pathway exists)
        if self.ln_ln is not None:
            state['I_ln_ln'] = self.ln_ln.init_current(batch_size, device)
            state['x_std_ln_ln'] = self.ln_ln.init_std_state(batch_size, device)
        if self.pn_ln is not None:
            state['I_pn_ln'] = self.pn_ln.init_current(batch_size, device)
            state['x_std_pn_ln'] = self.pn_ln.init_std_state(batch_size, device)
        if self.ln_orn is not None:
            state['I_ln_orn'] = self.ln_orn.init_current(batch_size, device)
            state['x_std_ln_orn'] = self.ln_orn.init_std_state(batch_size, device)
        # Non-AD connection states
        if self.orn_ln_nonad is not None:
            state['I_orn_ln_nonad'] = self.orn_ln_nonad.init_current(batch_size, device)
            state['x_std_orn_ln_nonad'] = self.orn_ln_nonad.init_std_state(batch_size, device)
        if self.ln_pn_nonad is not None:
            state['I_ln_pn_nonad'] = self.ln_pn_nonad.init_current(batch_size, device)
            state['x_std_ln_pn_nonad'] = self.ln_pn_nonad.init_std_state(batch_size, device)
        if self.ln_pn_excit_nonad is not None:
            state['I_ln_pn_excit_nonad'] = self.ln_pn_excit_nonad.init_current(batch_size, device)
            state['x_std_ln_pn_excit_nonad'] = self.ln_pn_excit_nonad.init_std_state(batch_size, device)
        if self.ln_ln_nonad is not None:
            state['I_ln_ln_nonad'] = self.ln_ln_nonad.init_current(batch_size, device)
            state['x_std_ln_ln_nonad'] = self.ln_ln_nonad.init_std_state(batch_size, device)
        if self.pn_ln_nonad is not None:
            state['I_pn_ln_nonad'] = self.pn_ln_nonad.init_current(batch_size, device)
            state['x_std_pn_ln_nonad'] = self.pn_ln_nonad.init_std_state(batch_size, device)
        if self.ln_orn_nonad is not None:
            state['I_ln_orn_nonad'] = self.ln_orn_nonad.init_current(batch_size, device)
            state['x_std_ln_orn_nonad'] = self.ln_orn_nonad.init_std_state(batch_size, device)
        return state

    def clamp_to_biological_bounds(self):
        """Clamp all learnable parameters in antennal lobe to biological bounds.

        Delegates to each owned sub-layer's clamp method (neurons + every present
        chemical-synapse pathway). Gap-junction conductances and gains are not
        clamped here (kept positive by their log/softplus parameterization).
        """
        self.orn_neurons.clamp_to_biological_bounds()
        self.ln_neurons.clamp_to_biological_bounds()
        self.pn_neurons.clamp_to_biological_bounds()
        self.orn_pn.clamp_to_biological_bounds()
        self.ln_pn.clamp_to_biological_bounds()
        self.ln_pn_excit.clamp_to_biological_bounds()
        if self.orn_ln is not None:
            self.orn_ln.clamp_to_biological_bounds()
        if self.ln_ln is not None:
            self.ln_ln.clamp_to_biological_bounds()
        if self.pn_ln is not None:
            self.pn_ln.clamp_to_biological_bounds()
        if self.ln_orn is not None:
            self.ln_orn.clamp_to_biological_bounds()
        # Non-AD connections
        if self.orn_ln_nonad is not None:
            self.orn_ln_nonad.clamp_to_biological_bounds()
        if self.ln_pn_nonad is not None:
            self.ln_pn_nonad.clamp_to_biological_bounds()
        if self.ln_pn_excit_nonad is not None:
            self.ln_pn_excit_nonad.clamp_to_biological_bounds()
        if self.ln_ln_nonad is not None:
            self.ln_ln_nonad.clamp_to_biological_bounds()
        if self.pn_ln_nonad is not None:
            self.pn_ln_nonad.clamp_to_biological_bounds()
        if self.ln_orn_nonad is not None:
            self.ln_orn_nonad.clamp_to_biological_bounds()


class SpikingAPLInhibition(nn.Module):
    """
    APL (Anterior Paired Lateral) neuron for global KC inhibition.

    The single APL neuron pools activity from all KCs and feeds graded inhibition
    back to all KCs, enforcing sparse KC coding. It is modeled as a GRADED leaky
    integrator (no spikes), consistent with the biology of invertebrate
    interneurons. The inhibition can be returned either subtractively (a current
    to add to the KC input) or divisively (a shunting gain factor); the paper uses
    the divisive form.

    APL uses GRADED transmission (not spiking) in Drosophila:
    - Receives excitatory input from all KCs
    - Provides graded inhibitory output to all KCs
    - Acts as a global activity regulator for sparsity

    Reference: Papadopoulou et al. 2011 (Neuron) - APL uses graded transmission

    This is biologically appropriate: many invertebrate interneurons
    use graded potentials rather than action potentials.
    """

    def __init__(
        self,
        kc_to_apl: torch.Tensor,
        apl_to_kc: torch.Tensor,
        params: Optional[SpikingParams] = None,
        kc_to_apl_da: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            kc_to_apl: (n_kc, n_apl) synapse counts for KC axon → APL dendrite (ad)
            apl_to_kc: (n_apl, n_kc) synapse counts for APL axon → KC dendrite (ad)
            params: SpikingParams
            kc_to_apl_da: (n_kc, n_apl) synapse counts for KC dendrite → APL axon (da)
                          If provided, enables graded dendritic contribution to APL

        Side effects:
            Registers normalized connectivity buffers and learnable gains/strengths
            (kc_apl_log_strength, optional kc_dend_gain, apl_gain, log_tau_apl).
        """
        super().__init__()

        self.params = params or SpikingParams()

        self.n_kc = kc_to_apl.shape[0]
        self.n_apl = kc_to_apl.shape[1]

        # KC axon → APL dendrite connectivity (excitatory, uses spike counts)
        # This is the dominant KC→APL pathway (925 synapses, 61%)
        # Learnable log_strength scales this pathway (consistent with SpikingConnectomeLinear)
        max_kc_apl = kc_to_apl.max()
        if max_kc_apl > 0:
            self.register_buffer('kc_apl_weights', kc_to_apl.float() / max_kc_apl)  # normalized to [0,1]
        else:
            self.register_buffer('kc_apl_weights', kc_to_apl.float())
        self.kc_apl_log_strength = nn.Parameter(torch.tensor(0.0))  # Learnable (init=1.0, preserves previous implicit strength)

        # KC dendrite → APL axon connectivity (graded, uses dendritic voltage)
        # This pathway (578 synapses, 38%) represents subthreshold dendritic activity
        if kc_to_apl_da is not None:
            max_kc_apl_da = kc_to_apl_da.max()
            if max_kc_apl_da > 0:
                self.register_buffer('kc_apl_da_weights', kc_to_apl_da.float() / max_kc_apl_da)
            else:
                self.register_buffer('kc_apl_da_weights', kc_to_apl_da.float())
            # Learnable gain for KC dendritic → APL pathway (graded, non-spiking)
            self.kc_dend_gain = nn.Parameter(torch.tensor(0.1))  # Start small
        else:
            self.register_buffer('kc_apl_da_weights', None)
            self.kc_dend_gain = None

        # APL axon → KC dendrite connectivity (inhibitory)
        # This is nearly all APL→KC output (681 synapses, 99%)
        max_apl_kc = apl_to_kc.max()
        if max_apl_kc > 0:
            self.register_buffer('apl_kc_weights', apl_to_kc.float() / max_apl_kc)  # normalized to [0,1]
        else:
            self.register_buffer('apl_kc_weights', apl_to_kc.float())

        # Learnable APL output gain (scales APL→KC inhibition); passed through softplus on use.
        self.apl_gain = nn.Parameter(torch.tensor(1.0))

        # APL time constant (graded dynamics), log space for positivity.
        self.log_tau_apl = nn.Parameter(torch.log(torch.tensor(15e-3)))  # 15 ms (bio: ~33ms, Ray et al. 2020)

    @property
    def tau_apl(self) -> torch.Tensor:
        # Realized APL integration time constant (s).
        return torch.exp(self.log_tau_apl)

    def forward(
        self,
        kc_spikes: torch.Tensor,
        apl_activity: torch.Tensor,
        kc_v_dend: Optional[torch.Tensor] = None,
        return_divisive: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute graded APL inhibition given KC spikes and optional dendritic voltage (one dt).

        Args:
            kc_spikes: (batch, n_kc) KC spike counts (axonal output)
            apl_activity: (batch, n_apl) current APL activity (graded)
            kc_v_dend: (batch, n_kc) KC dendritic voltage (optional, for 2-compartment)
                       If provided and kc_to_apl_da was given at init, contributes
                       graded dendritic activity to APL via KC dendrite → APL axon pathway
            return_divisive: If True, return divisive gain factor instead of subtractive current
                            Divisive inhibition is more biologically realistic (shunting)

        Returns:
            inhibition: (batch, n_kc) inhibitory current to subtract OR (per-KC) divisive factor
            apl_activity_new: (batch, n_apl) updated APL activity
        """
        # KC axon → APL dendrite excitation (spike-driven, learnable strength).
        kc_apl_strength = torch.exp(self.kc_apl_log_strength)
        apl_input = kc_apl_strength * torch.matmul(kc_spikes, self.kc_apl_weights)

        # KC dendrite → APL axon excitation (graded, voltage-driven)
        # This represents subthreshold dendritic activity influencing APL
        if kc_v_dend is not None and self.kc_apl_da_weights is not None:
            # Convert dendritic voltage to positive activity (ReLU above rest)
            # v_rest is typically -60 mV, so (v_dend - v_rest) gives depolarization
            v_rest = self.params.v_reset  # Use reset as proxy for rest
            kc_dend_activity = F.relu(kc_v_dend - v_rest)
            # Scale by learnable gain (softplus -> positive) and add to APL input.
            apl_input_da = torch.matmul(kc_dend_activity, self.kc_apl_da_weights)
            apl_input = apl_input + F.softplus(self.kc_dend_gain) * apl_input_da

        # Leaky integrator for APL (graded): exponential low-pass of apl_input toward
        # its instantaneous value with time constant tau_apl.
        dt = self.params.dt
        alpha = torch.exp(-dt / self.tau_apl)
        apl_activity_new = apl_activity * alpha + apl_input * (1 - alpha)

        # APL → KC inhibition (ReLU transfer — linear operating regime; gain via softplus).
        apl_output = F.relu(apl_activity_new) * F.softplus(self.apl_gain)

        if return_divisive:
            # Divisive (shunting) inhibition: returns a non-negative per-KC factor.
            # The caller applies  I_kc = I_pn / (1 + divisive_factor), modeling a
            # conductance-based shunt rather than a subtracted current.
            # Factor is per-KC based on APL→KC connectivity.
            divisive_factor = torch.matmul(apl_output, self.apl_kc_weights)
            return divisive_factor, apl_activity_new
        else:
            # Subtractive inhibition (original behavior): negative current to add to KC input.
            inhibition = -torch.matmul(apl_output, self.apl_kc_weights)
            return inhibition, apl_activity_new

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize APL activity to zero. Shape (batch, n_apl)."""
        return torch.zeros(batch_size, self.n_apl, device=device)

    def clamp_to_biological_bounds(self):
        """Clamp APL time constant to biological bounds (10..50 ms)."""
        with torch.no_grad():
            self.log_tau_apl.clamp_(LOG_TAU_APL_MIN, LOG_TAU_APL_MAX)


class SpikingKenyonCellLayer(nn.Module):
    """
    Spiking Kenyon Cell layer with APL feedback.

    The mushroom-body stage: receives PN spikes, drives a large sparse KC
    population, applies global APL inhibition, and optionally adds several
    KC-KC recurrent pathways (axon→axon spike-driven, plus graded dendritic
    couplings that require the 2-compartment KC model). Returns KC spike counts,
    which the downstream linear decoder reads.

    Supports two KC models:
    - Single-compartment LIF (default, trains better)
    - 2-compartment model (dendrite + axon, more biologically detailed)

    Both are biologically valid:
    - Single-compartment: Abstract model of KC integration
    - 2-compartment: Explicit calyx (dendrite) and lobe (axon) compartments
    """

    def __init__(
        self,
        pn_to_kc: torch.Tensor,
        kc_to_apl: torch.Tensor,
        apl_to_kc: torch.Tensor,
        params: Optional[SpikingParams] = None,
        target_sparsity: float = 0.10,
        use_two_compartment: bool = False,  # Single-compartment by default
        surrogate_method: str = 'soft',  # Surrogate gradient method
        kc_to_kc_aa: Optional[torch.Tensor] = None,
        kc_to_apl_da: Optional[torch.Tensor] = None,
        kc_to_kc_dd: Optional[torch.Tensor] = None,
        kc_to_kc_ad: Optional[torch.Tensor] = None,
        kc_to_kc_da: Optional[torch.Tensor] = None,
        pn_to_kc_nonad: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            pn_to_kc: (n_pn, n_kc) synapse counts
            kc_to_apl: (n_kc, n_apl) synapse counts for KC axon → APL dendrite
            apl_to_kc: (n_apl, n_kc) synapse counts for APL axon → KC dendrite
            params: SpikingParams
            target_sparsity: Target fraction of active KCs
            use_two_compartment: If True, use 2-compartment KC model
            surrogate_method: 'soft', 'superspike', or 'slayer'
            kc_to_kc_aa: (n_kc, n_kc) optional axon→axon KC-KC synapse counts
            kc_to_apl_da: (n_kc, n_apl) optional KC dendrite → APL axon synapse counts
                          Enables graded dendritic contribution to APL (requires 2-compartment)
            kc_to_kc_dd: (n_kc, n_kc) optional dendrite→dendrite KC-KC synapse counts
                         Graded dendritic coupling between KCs (excitatory, cholinergic)
            kc_to_kc_ad: (n_kc, n_kc) optional axon→dendrite KC-KC synapse counts
                         Spike-driven axonal input to dendrites (12 synapses, negligible)
            kc_to_kc_da: (n_kc, n_kc) optional dendrite→axon KC-KC synapse counts
                         Graded dendritic output to axons (30 synapses, negligible)
            pn_to_kc_nonad: (n_pn, n_kc) optional non-AD PN→KC synapse counts (3 synapses)

        Side effects:
            Builds the PN→KC synapse, the KC neuron model (LIF or two-compartment),
            the APL inhibition module, and any requested KC-KC recurrent pathways.
            Prints a line for the non-AD PN→KC pathway if present.
        """
        super().__init__()

        self.params = params or SpikingParams()
        self.target_sparsity = target_sparsity
        self.use_two_compartment = use_two_compartment
        self.surrogate_method = surrogate_method

        self.n_pn = pn_to_kc.shape[0]
        self.n_kc = pn_to_kc.shape[1]

        # PN → KC synaptic connections (excitatory feedforward drive).
        self.pn_kc = SpikingConnectomeLinear(pn_to_kc, params, sign="excitatory")

        # --- Build-banner: canonical (AD) KC / APL connections ---
        # edges = connected pre->post pairs; synapses = total synapse count.
        def _es(W):  # -> (edges, synapses)
            return int((W > 0).sum()), int(W.sum())
        _e, _s = _es(pn_to_kc);  _blog(f"  PN→KC (AD): {_e} edges ({_s} synapses)")
        _e, _s = _es(kc_to_apl); _blog(f"  KC→APL (AD): {_e} edges ({_s} synapses)")
        _e, _s = _es(apl_to_kc); _blog(f"  APL→KC (AD): {_e} edges ({_s} synapses)")

        # KC neurons - single compartment LIF by default (trains better)
        if use_two_compartment:
            self.kc_neurons = TwoCompartmentKC(self.n_kc, params, surrogate_method=surrogate_method)
        else:
            self.kc_neurons = LIFNeuron(self.n_kc, params, surrogate_method=surrogate_method)

        # APL inhibition (graded) with optional KC dendrite → APL pathway.
        self.apl = SpikingAPLInhibition(kc_to_apl, apl_to_kc, params, kc_to_apl_da=kc_to_apl_da)

        # KC-KC axon→axon recurrent connections (excitatory, cholinergic)
        # These are the dominant KC-KC connections (92% of all KC-KC synapses).
        # Spike-driven (uses previous-step KC spikes) and injected into the axon compartment.
        if kc_to_kc_aa is not None:
            self.kc_kc_aa = SpikingConnectomeLinear(kc_to_kc_aa, params, sign="excitatory")
        else:
            self.kc_kc_aa = None

        # KC-KC dendrite→dendrite graded connections (excitatory, cholinergic)
        # 987 synapses (4.8% dense) — graded dendritic coupling between KCs
        # Same mechanism as KC dendrite → APL (da): uses ReLU(v_dend - v_rest)
        # IMPORTANT: Graded output is voltage (V), but I_kc is in Amperes.
        # We need a conductance scale (1e-9 S = 1 nS) to convert V→A.
        self.graded_conductance_scale = 1e-9  # nS, biophysical V→A conversion
        if kc_to_kc_dd is not None and use_two_compartment:
            max_kc_kc_dd = kc_to_kc_dd.max()
            if max_kc_kc_dd > 0:
                self.register_buffer('kc_kc_dd_weights', kc_to_kc_dd.float() / max_kc_kc_dd)
            else:
                self.register_buffer('kc_kc_dd_weights', kc_to_kc_dd.float())
            # Initialize weak — softplus(-5) ≈ 0.007 (excitatory recurrence needs gentle start)
            self.kc_kc_dd_gain = nn.Parameter(torch.tensor(-5.0))
        else:
            self.register_buffer('kc_kc_dd_weights', None)
            self.kc_kc_dd_gain = None

        # KC-KC axon→dendrite (12 synapses — spike-driven, targets dendrite)
        if kc_to_kc_ad is not None and use_two_compartment and kc_to_kc_ad.sum() > 0:
            self.kc_kc_ad = SpikingConnectomeLinear(kc_to_kc_ad, params, sign="excitatory")
        else:
            self.kc_kc_ad = None

        # KC-KC dendrite→axon (30 synapses — graded, targets axon)
        if kc_to_kc_da is not None and use_two_compartment and kc_to_kc_da.sum() > 0:
            max_kc_kc_da = kc_to_kc_da.max()
            if max_kc_kc_da > 0:
                self.register_buffer('kc_kc_da_weights', kc_to_kc_da.float() / max_kc_kc_da)
            else:
                self.register_buffer('kc_kc_da_weights', kc_to_kc_da.float())
            self.kc_kc_da_gain = nn.Parameter(torch.tensor(-5.0))  # weak init (softplus(-5) ≈ 0.007)
        else:
            self.register_buffer('kc_kc_da_weights', None)
            self.kc_kc_da_gain = None

        # PN→KC non-AD (3 synapses — aa, negligible but for completeness); weak init strength.
        if pn_to_kc_nonad is not None and pn_to_kc_nonad.sum() > 0:
            self.pn_kc_nonad = SpikingConnectomeLinear(
                pn_to_kc_nonad, params, init_strength=1e-10, sign="excitatory")
            _blog(f"  PN→KC (non-AD): {int((pn_to_kc_nonad > 0).sum())} edges ({int(pn_to_kc_nonad.sum())} synapses)")
        else:
            self.pn_kc_nonad = None

    def forward(
        self,
        pn_spikes: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        n_steps: int = 50,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process PN spikes through KC layer.

        NOTE: This method is used for SEQUENTIAL simulation (backward compatibility).
        For UNIFIED simulation (biologically realistic), use the model's _unified_forward()
        which runs AL and KC together with proper spike-by-spike PN→KC transmission.

        In sequential mode, PN spike counts are converted to rates since the actual
        spike timing information is lost. For proper MB decorrelation, use unified simulation.

        Args:
            pn_spikes: (batch, n_pn) PN spike counts (from AL)
            state: Previous state dict (or None to initialize)
            n_steps: Number of simulation timesteps. NOTE: paper drivers set
                N_STEPS=30; the 50 here is a legacy/standalone default.

        Returns:
            kc_spike_count: (batch, n_kc) KC spike counts accumulated over the window
            state: Updated state dict
        """
        batch_size = pn_spikes.shape[0]
        device = pn_spikes.device

        # Initialize or unpack state
        if state is None:
            state = self._init_state(batch_size, device)

        I_pn_kc = state['I_pn_kc']
        apl_activity = state['apl_activity']
        refr = state['refr']

        # Handle both single and 2-compartment KC modes (different voltage state keys).
        if self.use_two_compartment:
            v_d = state['v_d']
            v_a = state['v_a']
        else:
            v = state['v']

        # KC-KC recurrent state: only unpack what exists. spk_kc_prev (previous-step
        # KC spikes) is shared by all recurrent pathways; per-pathway currents are
        # unpacked individually.
        has_any_kc_kc = (self.kc_kc_aa is not None or self.kc_kc_ad is not None
                         or self.kc_kc_dd_weights is not None or self.kc_kc_da_weights is not None)
        if has_any_kc_kc:
            spk_kc_prev = state['spk_kc_prev']
        if self.kc_kc_aa is not None:
            I_kc_kc_aa = state['I_kc_kc_aa']
        if self.kc_kc_ad is not None:
            I_kc_kc_ad = state['I_kc_kc_ad']

        # Spike accumulator over the window.
        kc_spike_count = torch.zeros(batch_size, self.n_kc, device=device)

        # Convert PN spike counts to rates (sequential mode loses spike timing)
        # For proper spike-by-spike transmission, use unified simulation instead
        pn_rate = pn_spikes / n_steps

        for step in range(n_steps):
            # PN → KC synaptic current (using rates in sequential mode).
            I_pn_kc = self.pn_kc(pn_rate, I_pn_kc)

            # APL inhibition (graded, uses running KC activity + dendritic voltage if 2-comp).
            # The running KC rate so far (count / steps-so-far) drives APL; 2-comp also
            # contributes dendritic voltage via the KC dendrite → APL pathway.
            kc_v_dend = v_d if self.use_two_compartment else None
            apl_inhib, apl_activity = self.apl(
                kc_spike_count / max(1, step + 1),
                apl_activity,
                kc_v_dend=kc_v_dend
            )

            # Total dendritic current to KC (PN drive + subtractive APL inhibition).
            I_kc = I_pn_kc + apl_inhib

            # KC-KC dendrite→dendrite graded current (excitatory, 987 synapses)
            # Clamp activity to prevent positive feedback runaway (bio: voltage bounded)
            # graded_conductance_scale converts voltage → current (V × nS → A)
            if self.kc_kc_dd_weights is not None and self.use_two_compartment:
                v_rest = self.params.v_reset
                kc_dend_activity = F.relu(v_d - v_rest).clamp(max=0.030)  # Cap at 30mV depol
                I_kc_dd = torch.matmul(kc_dend_activity, self.kc_kc_dd_weights)
                I_kc = I_kc + self.graded_conductance_scale * F.softplus(self.kc_kc_dd_gain) * I_kc_dd

            # KC-KC axon→dendrite spike-driven current (12 synapses), added to dendritic input.
            if self.kc_kc_ad is not None:
                I_kc_kc_ad = self.kc_kc_ad(spk_kc_prev, I_kc_kc_ad)
                I_kc = I_kc + I_kc_kc_ad

            # KC-KC axon→axon recurrent current (collected into the axon-compartment input).
            I_kc_axon = None
            if self.kc_kc_aa is not None:
                I_kc_kc_aa = self.kc_kc_aa(spk_kc_prev, I_kc_kc_aa)
                I_kc_axon = I_kc_kc_aa

            # KC-KC dendrite→axon graded current (30 synapses), also into axon compartment.
            if self.kc_kc_da_weights is not None and self.use_two_compartment:
                v_rest = self.params.v_reset
                kc_dend_act = F.relu(v_d - v_rest).clamp(max=0.030)
                I_kc_da = torch.matmul(kc_dend_act, self.kc_kc_da_weights)
                I_kc_da_scaled = self.graded_conductance_scale * F.softplus(self.kc_kc_da_gain) * I_kc_da
                # Combine with axon→axon current if present, else start it here.
                I_kc_axon = I_kc_da_scaled if I_kc_axon is None else I_kc_axon + I_kc_da_scaled

            # KC dynamics: dendritic input I_kc (and axonal input I_kc_axon for 2-comp).
            if self.use_two_compartment:
                v_d, v_a, spk_kc, refr = self.kc_neurons(I_kc, v_d, v_a, refr, I_axon=I_kc_axon)
            else:
                v, spk_kc, refr = self.kc_neurons(I_kc, v, refr)

            kc_spike_count += spk_kc

            # Track previous spikes for KC-KC recurrence (used next step as spk_kc_prev).
            if has_any_kc_kc:
                spk_kc_prev = spk_kc

        # Update state (mode-dependent voltage keys).
        if self.use_two_compartment:
            state = {
                'v_d': v_d, 'v_a': v_a, 'refr': refr,
                'I_pn_kc': I_pn_kc, 'apl_activity': apl_activity,
            }
        else:
            state = {
                'v': v, 'refr': refr,
                'I_pn_kc': I_pn_kc, 'apl_activity': apl_activity,
            }

        if has_any_kc_kc:
            state['spk_kc_prev'] = spk_kc_prev
        if self.kc_kc_aa is not None:
            state['I_kc_kc_aa'] = I_kc_kc_aa
        if self.kc_kc_ad is not None:
            state['I_kc_kc_ad'] = I_kc_kc_ad

        return kc_spike_count, state

    def _init_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Initialize all state variables for the KC layer.

        Voltages start at rest, refractory timer at 0, synaptic currents at 0,
        APL activity at 0, STD pools full, and KC-KC recurrent buffers (previous
        spikes / per-pathway currents) only when those pathways exist. Keys are
        mode-dependent (single- vs two-compartment voltage state).
        """
        if self.use_two_compartment:
            v_d, v_a, refr = self.kc_neurons.init_state(batch_size, device)
            state = {
                'v_d': v_d, 'v_a': v_a, 'refr': refr,
                'I_pn_kc': self.pn_kc.init_current(batch_size, device),
                'apl_activity': self.apl.init_state(batch_size, device),
            }
        else:
            v, refr = self.kc_neurons.init_state(batch_size, device)
            state = {
                'v': v, 'refr': refr,
                'I_pn_kc': self.pn_kc.init_current(batch_size, device),
                'apl_activity': self.apl.init_state(batch_size, device),
            }

        # STD vesicle state for PN→KC (full pool at start of stimulus).
        state['x_std_pn_kc'] = self.pn_kc.init_std_state(batch_size, device)

        # PN→KC non-AD state
        if self.pn_kc_nonad is not None:
            state['I_pn_kc_nonad'] = self.pn_kc_nonad.init_current(batch_size, device)
            state['x_std_pn_kc_nonad'] = self.pn_kc_nonad.init_std_state(batch_size, device)

        # KC-KC recurrent state (spk_kc_prev shared by aa, ad, and dd/da)
        has_any_kc_kc = (self.kc_kc_aa is not None or self.kc_kc_ad is not None
                         or self.kc_kc_dd_weights is not None or self.kc_kc_da_weights is not None)
        if has_any_kc_kc:
            state['spk_kc_prev'] = torch.zeros(batch_size, self.n_kc, device=device)
        if self.kc_kc_aa is not None:
            state['I_kc_kc_aa'] = self.kc_kc_aa.init_current(batch_size, device)
            state['x_std_kc_kc_aa'] = self.kc_kc_aa.init_std_state(batch_size, device)
        if self.kc_kc_ad is not None:
            state['I_kc_kc_ad'] = self.kc_kc_ad.init_current(batch_size, device)
            state['x_std_kc_kc_ad'] = self.kc_kc_ad.init_std_state(batch_size, device)

        return state

    def compute_sparsity(self, kc_spikes: torch.Tensor) -> float:
        """Compute fraction of KCs that spiked (mean over batch and KCs).

        Used as a diagnostic against target_sparsity. no_grad since it is a
        pure metric, not part of the loss path.
        """
        with torch.no_grad():
            active = (kc_spikes > 0).float()
            return active.mean().item()

    def clamp_to_biological_bounds(self):
        """Clamp all learnable parameters in KC layer to biological bounds.

        Delegates to the KC neuron, PN→KC synapse, APL module, and any present
        spike-driven KC-KC / non-AD synapses. The graded KC-KC gains are not
        clamped (softplus keeps them positive).
        """
        self.kc_neurons.clamp_to_biological_bounds()
        self.pn_kc.clamp_to_biological_bounds()
        self.apl.clamp_to_biological_bounds()
        if self.kc_kc_aa is not None:
            self.kc_kc_aa.clamp_to_biological_bounds()
        if self.kc_kc_ad is not None:
            self.kc_kc_ad.clamp_to_biological_bounds()
        if self.pn_kc_nonad is not None:
            self.pn_kc_nonad.clamp_to_biological_bounds()
        # kc_kc_dd_gain, kc_kc_da_gain have no explicit bounds (softplus keeps them positive)


if __name__ == "__main__":
    # Self-test / smoke test: exercises each layer with random inputs and prints
    # shapes, so running `python layers.py` verifies the module end-to-end.
    print("Testing spiking layers...")

    # Test LIF neuron
    print("\n1. Testing LIF neuron...")
    params = SpikingParams()
    lif = LIFNeuron(10, params)
    v, refr = lif.init_state(4, torch.device('cpu'))
    current = torch.randn(4, 10) * 1e-10
    v_new, spikes, refr_new = lif(current, v, refr)
    print(f"   Input current: {current.shape}")
    print(f"   Output voltage: {v_new.shape}, spikes: {spikes.sum().item()}")

    # Test 2-compartment KC (without I_axon)
    print("\n2a. Testing 2-compartment KC (dendrite only)...")
    kc = TwoCompartmentKC(72, params)
    v_d, v_a, refr = kc.init_state(4, torch.device('cpu'))
    I_syn = torch.randn(4, 72) * 1e-10
    v_d_new, v_a_new, spikes, refr_new = kc(I_syn, v_d, v_a, refr)
    print(f"   Synaptic current: {I_syn.shape}")
    print(f"   KC spikes: {spikes.sum().item()}")

    # Test 2-compartment KC (with I_axon)
    print("\n2b. Testing 2-compartment KC (dendrite + axon current)...")
    I_axon = torch.randn(4, 72) * 1e-10
    v_d_new2, v_a_new2, spikes2, refr_new2 = kc(I_syn, v_d, v_a, refr, I_axon=I_axon)
    print(f"   Axon current: {I_axon.shape}")
    print(f"   KC spikes (with I_axon): {spikes2.sum().item()}")
    # Sanity check: feeding axonal current must change the axon voltage trajectory.
    assert not torch.allclose(v_a_new, v_a_new2), "I_axon should change axon voltage"
    print(f"   Axon voltage differs with I_axon: PASS")

    # Test SpikingConnectomeLinear
    print("\n3. Testing SpikingConnectomeLinear...")
    syn_counts = torch.randint(0, 10, (21, 72)).float()
    scl = SpikingConnectomeLinear(syn_counts, params)
    spikes = (torch.rand(4, 21) > 0.9).float()
    I_syn = torch.zeros(4, 72)
    I_new = scl(spikes, I_syn)
    print(f"   Weights shape: {scl.get_weights().shape}")
    print(f"   Output current: {I_new.shape}")

    # Test SpikingAPL
    print("\n4. Testing SpikingAPL (graded)...")
    kc_to_apl = torch.randint(0, 5, (72, 2)).float()
    apl_to_kc = torch.randint(0, 5, (2, 72)).float()
    apl = SpikingAPLInhibition(kc_to_apl, apl_to_kc, params)
    kc_spikes = torch.randn(4, 72).abs()
    apl_activity = apl.init_state(4, torch.device('cpu'))
    inhib, apl_new = apl(kc_spikes, apl_activity)
    print(f"   KC spikes: {kc_spikes.shape}")
    print(f"   Inhibition: {inhib.shape}")
    print(f"   APL activity: {apl_new.shape}")

    # Test SpikingKenyonCellLayer with KC-KC aa
    print("\n5. Testing SpikingKenyonCellLayer with KC-KC aa...")
    n_kc = 72
    pn_to_kc = torch.randint(0, 5, (21, n_kc)).float()
    kc_to_apl = torch.randint(0, 3, (n_kc, 2)).float()
    apl_to_kc = torch.randint(0, 3, (2, n_kc)).float()
    kc_to_kc_aa = torch.randint(0, 3, (n_kc, n_kc)).float()
    kc_layer = SpikingKenyonCellLayer(
        pn_to_kc, kc_to_apl, apl_to_kc, params,
        use_two_compartment=True, kc_to_kc_aa=kc_to_kc_aa,
    )
    pn_spk = torch.randn(4, 21).abs()
    kc_counts, kc_state = kc_layer(pn_spk, n_steps=10)
    print(f"   KC spike counts: {kc_counts.shape}")
    print(f"   KC-KC aa current in state: {'I_kc_kc_aa' in kc_state}")
    # KC-KC recurrent state must round-trip through the returned state dict.
    assert 'I_kc_kc_aa' in kc_state, "KC-KC state should be in output"
    assert 'spk_kc_prev' in kc_state, "spk_kc_prev should be in output"
    print(f"   KC-KC aa test: PASS")

    # Test without KC-KC (backward compatibility)
    print("\n6. Testing SpikingKenyonCellLayer without KC-KC (backward compat)...")
    kc_layer_no_kckc = SpikingKenyonCellLayer(
        pn_to_kc, kc_to_apl, apl_to_kc, params, use_two_compartment=True,
    )
    kc_counts2, kc_state2 = kc_layer_no_kckc(pn_spk, n_steps=10)
    print(f"   KC spike counts: {kc_counts2.shape}")
    # Without kc_to_kc_aa the recurrent pathway and its state must be absent.
    assert 'I_kc_kc_aa' not in kc_state2, "No KC-KC state without kc_to_kc_aa"
    assert kc_layer_no_kckc.kc_kc_aa is None
    print(f"   Backward compatibility: PASS")

    print("\nAll spiking layer tests passed!")
