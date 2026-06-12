"""
SpikingConnectomeConstrainedModel: full spiking olfactory pathway model.

Paper model: OR → ORN (LIF) → LN (LIF) → PN (LIF) → KC (2-compartment) ← APL (graded) → Decoder
With gap junctions (LN-LN, PN-PN, eLN-PN), KC-KC recurrent excitation,
Tsodyks-Markram STD on all chemical synapses, and non-AD connections.

Key features:
1. ORN, LN, PN use LIF (Leaky Integrate-and-Fire) dynamics
2. KC uses 2-compartment model (dendrite + axon) with conductance coupling
3. APL uses graded transmission (biologically appropriate for invertebrate interneurons)
4. Temporal dynamics: spike counts accumulated over simulation window
5. Biologically realistic noise (6 sources)

Biological justification for non-spiking components:
- OR responses: Receptor-mediated transduction produces graded receptor potentials
- APL: Uses graded transmission in Drosophila (Papadopoulou et al. 2011, Neuron)

References:
- Lappalainen et al. 2024: Connectome-constrained learning
- Winding et al. 2023: Larval Drosophila connectome
- Kreher et al. 2008: Larval ORN responses
- Papadopoulou et al. 2011: APL graded transmission

================================================================================
MODULE OVERVIEW (extended commentary)
================================================================================
This is the central model file for the CCN 2026 connectome-constrained spiking
neural network of the larval Drosophila olfactory pathway. The full processing
pipeline is:

    OR responses (Kreher 2008 receptor-tuning data, graded)
        -> ORN  (Olfactory Receptor Neuron, LIF spiking)
        -> LN   (Local interneuron of the antennal lobe, LIF spiking)
        -> PN   (Projection Neuron, LIF spiking)
        -> KC   (Kenyon Cell, two-compartment dendrite+axon spiking)
        <- APL  (Anterior Paired Lateral neuron, graded divisive/shunting inhibition)
        -> linear decoder over 28 odor classes.

CONNECTIVITY is FIXED: every synaptic adjacency/projection matrix is taken
directly from the Winding et al. 2023 larval connectome and is NOT learned. Only
~449 biological parameters (synaptic unitary strengths, neuron thresholds,
membrane time constants, dendro-axonal coupling g_soma, APL gain, gap-junction
conductances, OR gains, decoder weights) are learned by gradient descent.

UNITS CONVENTION used throughout (SI):
- Voltages (membrane potentials v_*, thresholds v_th, reset v_reset): volts (V).
  Note physiological values are on the order of -0.055 V to -0.030 V (i.e. mV
  expressed in V); e.g. the 0.030 cap below is 30 mV of depolarisation.
- Currents (all I_* tensors and *_inject_current scalars): amperes (A); synaptic
  unitary strengths live in roughly [1e-12, 1e-7] A.
- Conductances (gap-junction g_*, dendro-axonal g_soma): siemens (S), with the
  log-space parameters expressed so that exp() yields nanosiemens-scale values.
- Time constants (tau_*): seconds (s) at the dynamics level, though clamp bounds
  are documented in ms.

LOG-SPACE PARAMETERS: many learnable quantities are stored as their natural log
(log_or_gain, log_g_gap_*, log_g_soma, log_tau_*, log_strength). They are
exp()'d on read so that the stored parameter is unconstrained while the effective
value stays strictly positive. Gains that must stay positive but need not be
strictly exponential instead use F.softplus().

THE CANONICAL PATH is `_unified_forward`: it runs the antennal-lobe (AL) and
Kenyon-cell (KC) layers together inside ONE time loop so PN spikes are delivered
to the KCs in real time (spike-by-spike), enabling coincidence detection. The
alternative `unified_simulation=False` branch runs AL fully, then KC fully, and
exists only for backward compatibility.

LEGACY DEFAULTS: the in-class `compute_loss` plus the constructor defaults
`target_sparsity=0.10` is LEGACY (n_steps now defaults to the canonical 30). The real
training/experiment scripts (run_training.py and the other run_*.py drivers)
override these: they set N_STEPS=30 after construction and use their own sparsity
objective (sigmoid offset 0.02, target 0.05). Treat compute_loss here as
illustrative, not as the loss that produced the paper's numbers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, Tuple

# --- build-banner logging: silenced by default; set this module's _BUILD_VERBOSE = True to restore ---
_BUILD_VERBOSE = False
def _blog(*args, **kwargs):
    if _BUILD_VERBOSE:
        print(*args, **kwargs)


# Layer primitives live in layers.py. These encapsulate the LIF dynamics, the
# two-compartment KC, the connectome-masked linear synapse (with optional
# Tsodyks-Markram short-term depression), the assembled antennal lobe, the
# assembled KC layer, and the graded APL inhibition unit.
from .layers import (
    SpikingParams,             # Container of all default biophysical constants.
    LIFNeuron,                 # Leaky integrate-and-fire neuron population.
    TwoCompartmentKC,          # Dendrite+axon KC with conductance coupling g_soma.
    SpikingConnectomeLinear,   # Connectome-masked synaptic projection (+ optional STD).
    SpikingAntennalLobe,       # ORN/LN/PN assembly with recurrence and gap junctions.
    SpikingKenyonCellLayer,    # PN->KC, KC-KC recurrence, and the APL loop.
    SpikingAPLInhibition,      # Graded divisive inhibition driven by KC activity.
)


class ORtoORNMapping(nn.Module):
    """
    Fixed biological mapping from OR type responses to ORN neuron activations.

    Same as rate-based model: 21 OR types → 42 ORN neurons (L/R hemispheres).
    OR responses are GRADED (not spiking) - this is the transduction stage.

    This module implements the very first stage of the pipeline: it takes the
    graded receptor-tuning responses (one value per olfactory receptor type from
    the Kreher 2008 data) and expands them into one input current per ORN. The
    expansion is a FIXED binary mapping (not learned): each OR type drives exactly
    two ORNs, the left- and right-hemisphere copies of that glomerulus. The only
    LEARNABLE thing here is a per-OR-type gain (passed through softplus so it is
    always positive). The output is still graded (a current, not spikes); the LIF
    spiking happens downstream inside the antennal lobe.

    Tensor shapes / units:
        Input  or_responses:     (batch, n_or_types)   dimensionless receptor response.
        Output orn_activations:  (batch, n_orn_neurons) graded drive (A-scale, later
                                 multiplied by the AL's OR gain to become a current).
    """

    # Canonical ordering of the 21 larval OR types as they appear in the Kreher
    # 2008 dataset. The order matters because it fixes which OR row maps to which
    # ORN pair; downstream connectome rows assume this exact sequence.
    KREHER_OR_ORDER = [
        'Or1a', 'Or2a', 'Or7a', 'Or13a', 'Or22c', 'Or24a', 'Or30a', 'Or33b',
        'Or35a', 'Or42a', 'Or42b', 'Or45a', 'Or45b', 'Or47a', 'Or49a', 'Or59a',
        'Or67b', 'Or74a', 'Or82a', 'Or83a', 'Or85c'
    ]

    def __init__(self, n_or_types: int = 21, n_orn_neurons: int = 42):
        """
        Build the fixed OR->ORN expansion and the learnable per-OR gains.

        Args:
            n_or_types: Number of olfactory receptor types (21 larval ORs). One
                graded response value is supplied per type per sample.
            n_orn_neurons: Number of ORN neurons (42 = 21 types x 2 hemispheres).

        Side effects:
            Registers a non-trainable buffer 'mapping' of shape
            (n_or_types, n_orn_neurons) and a trainable Parameter 'or_gains' of
            shape (n_or_types,).
        """
        super().__init__()
        self.n_or_types = n_or_types
        self.n_orn_neurons = n_orn_neurons

        # Fixed binary mapping (21 ORs × 2 = 42 ORNs)
        # Build a (n_or_types, n_orn_neurons) matrix of 0/1. OR type i drives the
        # two ORNs at columns 2*i (left hemisphere) and 2*i+1 (right hemisphere).
        # The bounds checks keep this safe if n_orn_neurons is not exactly 2x.
        mapping = torch.zeros(n_or_types, n_orn_neurons)
        for i in range(n_or_types):
            left_idx = 2 * i        # Left-hemisphere ORN column for OR type i.
            right_idx = 2 * i + 1   # Right-hemisphere ORN column for OR type i.
            if left_idx < n_orn_neurons:
                mapping[i, left_idx] = 1.0
            if right_idx < n_orn_neurons:
                mapping[i, right_idx] = 1.0

        # register_buffer => part of state_dict and moved with .to(device), but
        # NOT a learnable parameter. The connectivity here is biologically fixed.
        self.register_buffer('mapping', mapping)

        # Learnable gain per OR type
        # One scalar gain per OR type. softplus() is applied at forward time so
        # the effective gain is always > 0. Initialised to 1.0 (softplus(1)~1.31).
        self.or_gains = nn.Parameter(torch.ones(n_or_types))

    def forward(self, or_responses: torch.Tensor) -> torch.Tensor:
        """Map OR responses to ORN input (still graded at this stage).

        Args:
            or_responses: (batch, n_or_types) graded receptor responses,
                dimensionless.

        Returns:
            orn_activations: (batch, n_orn_neurons) graded ORN drive. Each ORN
                receives the (gain-scaled) response of the single OR type that
                projects to it via the fixed binary mapping.
        """
        # Apply the per-OR-type learnable gain (forced positive by softplus) to
        # each receptor response. Broadcasts over the batch dimension.
        scaled_responses = or_responses * F.softplus(self.or_gains)
        # Expand 21 OR-type values to 42 ORN values via the fixed binary matrix.
        # Because each ORN column has a single 1, this is effectively a scatter:
        # ORN gets exactly the gain-scaled response of its source OR type.
        orn_activations = torch.matmul(scaled_responses, self.mapping)
        return orn_activations


class SpikingConnectomeConstrainedModel(nn.Module):
    """
    Complete spiking connectome-constrained olfactory pathway model.

    Architecture:
        OR (rate) → ORN (LIF) → LN (LIF) → PN (LIF) → KC (2-comp) ← APL (graded) → Decoder

    All connectivity is constrained by the Winding et al. 2023 connectome.

    Spiking components:
    - ORN: LIF neurons converting graded OR input to spikes
    - LN: LIF lateral inhibition in antennal lobe
    - PN: LIF projection neurons
    - KC: 2-compartment model (dendrite + axon) with conductance coupling

    Graded (non-spiking) components:
    - OR responses: Receptor binding kinetics (input is graded)
    - APL: Graded transmission for global inhibition (biologically validated)
    - Decoder: Rate-based readout of KC spike counts

    Learnable parameters:
    - OR gain scaling
    - Unitary synapse strengths (per connection type)
    - Neural thresholds (per population or per neuron)
    - APL gain
    - Decoder weights

    Implementation notes:
        The top-level module wires together the ORtoORNMapping, the
        SpikingAntennalLobe, the SpikingKenyonCellLayer, and a plain linear
        decoder. The forward pass simulates the network for a fixed number of
        discrete timesteps, accumulating spike COUNTS, then converts those counts
        to RATES (counts / n_steps) before the decoder. The canonical simulation
        path (`_unified_forward`) runs AL and KC in a single shared time loop.
    """

    def __init__(
        self,
        connectome: Dict[str, torch.Tensor],
        n_odors: int,
        n_or_types: int = 21,
        params: Optional[SpikingParams] = None,
        target_sparsity: float = 0.10,
        n_steps_al: int = 30,
        n_steps_kc: int = 30,
        surrogate_method: str = 'soft',
    ):
        """
        Args:
            connectome: Dictionary with connectivity matrices
            n_odors: Number of odor classes for decoder
            n_or_types: Number of OR types (21 larval ORs)
            params: SpikingParams (uses defaults if None)
            target_sparsity: Legacy KC-sparsity target for the in-class compute_loss
                only. The canonical training pipeline (scripts/run_training.py) uses
                its own sparsity loss (offset 0.02, target 0.05) and ignores this.
            n_steps_al: Simulation steps for antennal lobe. Default 30 = the canonical
                value used throughout the paper (scripts/run_training.py N_STEPS).
            n_steps_kc: Simulation steps for KC layer (see n_steps_al note; canonical = 30).
            surrogate_method: Gradient method - 'soft' (canonical), 'superspike', or 'slayer'

        connectome dict keys (each value is a fixed adjacency/projection tensor
        from the Winding 2023 connectome; shapes are (n_pre, n_post)):
            'orn_to_pn'  -> (n_orn, n_pn)   : ORN->PN feedforward excitation.
            'ln_to_pn'   -> (n_ln, n_pn)    : LN->PN (sign handled in layer).
            'pn_to_kc'   -> (n_pn, n_kc)    : PN->KC feedforward excitation.
            'kc_to_apl'  -> (n_kc, n_apl)   : KC axon drive onto APL.
            'apl_to_kc'  -> (n_apl, n_kc)   : APL inhibitory feedback onto KCs.
            (optional) orn_to_ln, ln_to_ln, pn_to_ln, ln_to_orn : AL recurrence.
            (optional) *_nonad : non-axon-dendrite compartment collapsed contacts.
            (optional) kc_to_kc_{aa,ad,dd,da}, kc_to_apl_da : KC compartment graph.

        Side effects:
            Constructs and registers the OR->ORN map, antennal lobe, KC layer, and
            decoder submodules. Prints a human-readable build summary to stdout.
        """
        super().__init__()

        # Fall back to default biophysical constants if none supplied.
        self.params = params or SpikingParams()
        self.n_or_types = n_or_types
        # Population sizes are READ OFF the connectome tensors so the model always
        # matches the supplied connectivity (n_pre/n_post of each matrix).
        self.n_orn = connectome['orn_to_pn'].shape[0]   # rows of ORN->PN.
        self.n_pn = connectome['orn_to_pn'].shape[1]    # cols of ORN->PN.
        self.n_ln = connectome['ln_to_pn'].shape[0]     # rows of LN->PN.
        self.n_kc = connectome['pn_to_kc'].shape[1]     # cols of PN->KC.
        self.n_apl = connectome['kc_to_apl'].shape[1]   # cols of KC->APL.
        self.n_odors = n_odors
        self.target_sparsity = target_sparsity          # LEGACY (compute_loss only).
        self.n_steps_al = n_steps_al                    # default 30 (canonical).
        self.n_steps_kc = n_steps_kc                    # default 30 (canonical).
        self.surrogate_method = surrogate_method        # Surrogate-gradient flavour.

        # Check for KC-KC recurrent connectivity
        # Presence of the axon->axon KC graph signals a compartment-resolved
        # connectome was loaded; used only to enrich the build printout below.
        has_kc_kc = 'kc_to_kc_aa' in connectome

        _blog(f"Building SPIKING model with:")
        _blog(f"  OR types (input): {self.n_or_types}")
        _blog(f"  ORN: {self.n_orn} (LIF neurons)")
        _blog(f"  LN: {self.n_ln} (LIF neurons)")
        _blog(f"  PN: {self.n_pn} (LIF neurons)")
        _blog(f"  KC: {self.n_kc} (2-compartment with learnable g_soma)")
        _blog(f"  APL: {self.n_apl} (graded, non-spiking)")
        if has_kc_kc:
            kc_kc_aa = connectome['kc_to_kc_aa']
            # Count the nonzero entries = number of KC axon->axon synapses present.
            n_kc_kc = int((kc_kc_aa > 0).sum().item())
            n_kc_kc_syn = int(kc_kc_aa.sum().item())
            _blog(f"  KC-KC aa: {n_kc_kc} edges ({n_kc_kc_syn} synapses, axon→axon, excitatory)")
        _blog(f"  Simulation: {n_steps_al} AL steps, {n_steps_kc} KC steps")
        _blog(f"  Target sparsity: {target_sparsity:.1%}")
        _blog(f"  Surrogate gradient: {surrogate_method}")
        _blog(f"  Biological bounds: ENABLED (v_th, tau_m, g_soma clamped)")

        # OR → ORN mapping (graded transduction)
        # Stage 1: fixed binary expansion + learnable per-OR gains (see class above).
        self.or_to_orn = ORtoORNMapping(n_or_types, self.n_orn)

        # Spiking Antennal Lobe (with recurrent connections + non-AD)
        # Stage 2: ORN/LN/PN spiking assembly. .get(...) returns None for any
        # optional pathway that was not loaded, and the layer treats None as
        # "this connection is absent". The *_nonad arguments add the
        # non-axon-dendrite (collapsed) compartment contacts when available.
        self.antennal_lobe = SpikingAntennalLobe(
            orn_to_pn=connectome['orn_to_pn'],
            ln_to_pn=connectome['ln_to_pn'],
            orn_to_ln=connectome.get('orn_to_ln', None),
            ln_to_ln=connectome.get('ln_to_ln', None),
            pn_to_ln=connectome.get('pn_to_ln', None),
            ln_to_orn=connectome.get('ln_to_orn', None),
            params=self.params,
            # Non-AD compartment connections
            orn_to_ln_nonad=connectome.get('orn_to_ln_nonad', None),
            ln_to_pn_nonad=connectome.get('ln_to_pn_nonad', None),
            ln_to_ln_nonad=connectome.get('ln_to_ln_nonad', None),
            pn_to_ln_nonad=connectome.get('pn_to_ln_nonad', None),
            ln_to_orn_nonad=connectome.get('ln_to_orn_nonad', None),
        )

        # Spiking KC layer - 2-compartment with learnable g_soma (biological realism)
        # All 4 KC-KC compartment connections loaded from connectome
        # Stage 3: KC layer plus its embedded APL inhibition loop. The four
        # KC-KC compartment graphs (aa=axon->axon, da=dendrite->axon,
        # dd=dendrite->dendrite, ad=axon->dendrite) and the PN->KC non-AD contacts
        # are passed through when present.
        self.kc_layer = SpikingKenyonCellLayer(
            pn_to_kc=connectome['pn_to_kc'],
            kc_to_apl=connectome['kc_to_apl'],
            apl_to_kc=connectome['apl_to_kc'],
            params=self.params,
            target_sparsity=target_sparsity,
            use_two_compartment=True,  # 2-compartment with learnable g_soma
            surrogate_method=surrogate_method,
            kc_to_kc_aa=connectome.get('kc_to_kc_aa', None),
            kc_to_apl_da=connectome.get('kc_to_apl_da', None),
            kc_to_kc_dd=connectome.get('kc_to_kc_dd', None),
            kc_to_kc_ad=connectome.get('kc_to_kc_ad', None),
            kc_to_kc_da=connectome.get('kc_to_kc_da', None),
            pn_to_kc_nonad=connectome.get('pn_to_kc_nonad', None),
        )

        # Linear decoder: KC spike counts → odor classification
        # Stage 4: plain trainable linear readout from KC spike RATES to odor
        # logits. This is the only fully unconstrained (non-connectome) weight set.
        self.decoder = nn.Linear(self.n_kc, n_odors)

    def forward(
        self,
        or_input: torch.Tensor,
        return_all: bool = False,
        al_state: Optional[Dict] = None,
        kc_state: Optional[Dict] = None,
        unified_simulation: bool = True,  # Run AL+KC together (biologically realistic)
        disable_apl: bool = False,  # For biological validation: disable APL inhibition
        disable_std: bool = False,  # STD ablation: skip short-term depression on all chemical synapses
        apl_inject_current: float = 0.0,  # For Mancini 2023 validation: optogenetic APL activation
        kc_inject_current: float = 0.0,  # For Mancini 2023: carbachol-like KC activation
    ) -> torch.Tensor:
        """
        Forward pass through spiking olfactory pathway.

        Args:
            or_input: (batch, n_or_types) OR response pattern
            return_all: If True, return intermediate spike counts
            al_state: Previous antennal lobe state (for temporal processing)
            kc_state: Previous KC state (for temporal processing)
            unified_simulation: If True, run AL and KC simultaneously (biologically realistic)
            disable_apl: If True, skip APL inhibition (for biological validation experiments)
            disable_std: If True, skip short-term depression on all chemical synapses (STD ablation)
            apl_inject_current: Direct current injection into APL (simulates optogenetic activation)
            kc_inject_current: Direct current injection into all KCs (simulates carbachol/ACh agonist)

        Returns:
            logits: (batch, n_odors) classification logits
            (optional) dict with spike counts and states

        Units / shapes detail:
            or_input is dimensionless (batch, n_or_types). apl_inject_current and
            kc_inject_current are scalar currents in A added uniformly. The
            returned logits are (batch, n_odors) pre-softmax scores. When
            return_all is True the second element holds ORN input, PN/KC spike
            COUNTS (not rates), the KC sparsity scalar, and the updated AL/KC
            states (dicts of tensors) so simulation can be continued statefully.
        """
        # OR → ORN (graded input to spiking neurons)
        # Stage 1: expand graded receptor responses to per-ORN graded drive.
        orn_input = self.or_to_orn(or_input)

        if unified_simulation:
            # UNIFIED SIMULATION: Run AL and KC together (like real flies!)
            # This enables proper spike-by-spike PN→KC transmission
            # CANONICAL PATH: a single shared time loop drives AL and KC so that
            # each timestep's PN spikes immediately become KC input. Returns the
            # accumulated PN and KC spike counts plus updated states.
            pn_spikes, kc_spikes, al_state_new, kc_state_new = self._unified_forward(
                orn_input, al_state, kc_state, disable_apl=disable_apl,
                disable_std=disable_std,
                apl_inject_current=apl_inject_current,
                kc_inject_current=kc_inject_current
            )
        else:
            # SEQUENTIAL SIMULATION (old behavior for backward compatibility)
            # Run the AL to completion, collect PN spike counts, THEN feed those
            # counts to the KC layer. This loses real-time PN->KC coincidence and
            # is retained only for backward compatibility / comparison.
            pn_spikes, al_state_new = self.antennal_lobe(
                orn_input, state=al_state, n_steps=self.n_steps_al
            )
            kc_spikes, kc_state_new = self.kc_layer(
                pn_spikes, state=kc_state, n_steps=self.n_steps_kc
            )

        # Convert spike counts to rates (normalize by timesteps)
        # This gives continuous values like rate-based model
        # Dividing the accumulated count by the number of simulated steps yields a
        # per-step firing RATE in [0, 1], the continuous quantity the decoder reads.
        n_steps = max(self.n_steps_al, self.n_steps_kc)
        kc_rates = kc_spikes / n_steps

        # Decode from KC spike rates (not raw counts)
        # Linear readout to odor logits. Using rates (not counts) keeps the decoder
        # input scale independent of how many timesteps were simulated.
        logits = self.decoder(kc_rates)

        if return_all:
            # Compute the KC population sparsity (fraction of active KCs) for logging.
            sparsity = self.kc_layer.compute_sparsity(kc_spikes)
            return logits, {
                'orn_input': orn_input,
                'pn_spikes': pn_spikes,
                'kc_spikes': kc_spikes,
                'sparsity': sparsity,
                'al_state': al_state_new,
                'kc_state': kc_state_new,
            }
        return logits

    def _unified_forward(
        self,
        orn_input: torch.Tensor,
        al_state: Optional[Dict],
        kc_state: Optional[Dict],
        disable_apl: bool = False,
        disable_std: bool = False,
        apl_inject_current: float = 0.0,
        kc_inject_current: float = 0.0,
    ):
        """
        Run AL and KC SIMULTANEOUSLY in a unified time loop.

        This is biologically realistic: PN spikes arrive at KCs as they occur,
        enabling proper coincidence detection and temporal dynamics.

        Args:
            apl_inject_current: Direct current injection into APL activity
                (simulates optogenetic APL activation, for Mancini 2023 validation)
            kc_inject_current: Direct current injection into all KCs
                (simulates carbachol/acetylcholine agonist that broadly activates KCs)

        Full argument / return contract:
            orn_input: (batch, n_orn) graded ORN drive from ORtoORNMapping.
            al_state / kc_state: optional dicts of carried-over state tensors
                (membrane voltages V, refractory counters, per-synapse currents I
                in A, and Tsodyks-Markram vesicle availabilities x_std in [0,1]).
                Pass None to start fresh; the layers' _init_state builds zeros.
            disable_apl: bool, removes the APL divisive term from KC input.
            disable_std: bool, drops every 'x_std*' entry so synapses take the
                no-depression branch (static synaptic strength).
            apl_inject_current / kc_inject_current: scalar currents in A.

        Returns:
            pn_spike_count: (batch, n_pn) summed PN spikes over the window.
            kc_spike_count: (batch, n_kc) summed KC spikes over the window.
            al_state_new, kc_state_new: updated state dicts for continuation.

        TIME-LOOP ORDERING per step (the canonical biophysical sequence):
            1. ORN dynamics (OR drive + LN->ORN feedback from previous step).
            2. Gap-junction currents (LN-LN, PN-PN, eLN<->PN), computed from the
               current voltages (instantaneous, no integration).
            3. LN dynamics (ORN drive + LN->LN + PN->LN feedback + gap junctions).
            4. PN dynamics (ORN->PN + split inhibitory/excitatory LN->PN + gap).
            5. PN spikes delivered to KCs THIS step (real-time PN->KC).
            6. APL divisive inhibition from running KC activity (+ dendritic V).
            7. KC-KC recurrence (dd, ad into dendrite; aa, da into axon).
            8. KC two-compartment dynamics, then spike accumulation.
        """
        batch_size = orn_input.shape[0]
        device = orn_input.device
        # Both sublayers iterate for the SAME number of steps so they stay in lockstep.
        n_steps = max(self.n_steps_al, self.n_steps_kc)

        # Initialize AL state
        # Build zeroed AL state on first call (no carried-over state supplied).
        if al_state is None:
            al_state = self.antennal_lobe._init_state(batch_size, device)
        if disable_std:  # STD ablation: drop x_std state so all synapses use the no-depression path
            # Removing every 'x_std*' key forces each synapse below into its
            # else-branch, where strength is static (no Tsodyks-Markram depression).
            al_state = {k: v for k, v in al_state.items() if not k.startswith('x_std')}

        # --- Unpack AL membrane / refractory / current state into loop-local vars ---
        v_orn = al_state['v_orn']        # (batch, n_orn) ORN membrane potential, V.
        v_ln = al_state['v_ln']          # (batch, n_ln)  LN membrane potential, V.
        v_pn = al_state['v_pn']          # (batch, n_pn)  PN membrane potential, V.
        refr_orn = al_state['refr_orn']  # (batch, n_orn) ORN refractory countdown.
        refr_ln = al_state['refr_ln']    # (batch, n_ln)  LN refractory countdown.
        refr_pn = al_state['refr_pn']    # (batch, n_pn)  PN refractory countdown.
        I_orn_pn = al_state['I_orn_pn']  # (batch, n_pn)  ORN->PN synaptic current, A.
        I_ln_pn = al_state['I_ln_pn']    # (batch, n_pn)  inhibitory LN->PN current, A.
        # ORN->LN current; default to zeros if the connection was not in state.
        I_orn_ln = al_state.get('I_orn_ln', torch.zeros(batch_size, self.antennal_lobe.n_ln, device=device))
        # Split LN→PN excitatory current
        # PNs receive BOTH inhibitory (GABAergic) and excitatory (glutamatergic)
        # LN input; this is the separate excitatory channel (default zeros).
        I_ln_pn_excit = al_state.get('I_ln_pn_excit',
                                      torch.zeros(batch_size, self.antennal_lobe.n_pn, device=device))
        # STD vesicle states (AL synapses)
        # Tsodyks-Markram available-resource fractions x in [0,1]; None => the
        # no-depression path is taken for that synapse (also the disable_std case).
        x_std_orn_pn = al_state.get('x_std_orn_pn')
        x_std_ln_pn = al_state.get('x_std_ln_pn')
        x_std_ln_pn_excit = al_state.get('x_std_ln_pn_excit')
        x_std_orn_ln = al_state.get('x_std_orn_ln')
        # AL recurrent state
        # Previous-step spikes feed the recurrent connections (LN->LN, PN->LN,
        # LN->ORN), implementing the unavoidable one-step transmission delay.
        spk_ln_prev = al_state.get('spk_ln_prev', torch.zeros(batch_size, self.antennal_lobe.n_ln, device=device))
        spk_pn_prev = al_state.get('spk_pn_prev', torch.zeros(batch_size, self.antennal_lobe.n_pn, device=device))
        # AL recurrent currents
        I_ln_ln = al_state.get('I_ln_ln')          # LN->LN lateral inhibition current, A.
        x_std_ln_ln = al_state.get('x_std_ln_ln')  # its STD vesicle state.
        I_pn_ln = al_state.get('I_pn_ln')          # PN->LN feedback current, A.
        x_std_pn_ln = al_state.get('x_std_pn_ln')
        I_ln_orn = al_state.get('I_ln_orn')        # LN->ORN feedback current, A.
        x_std_ln_orn = al_state.get('x_std_ln_orn')
        # Non-AD AL connection currents
        # Non-axon-dendrite (collapsed compartment) contacts: extra, usually small,
        # contributions to the same AL pathways. Each has its own current + STD state.
        I_orn_ln_nonad = al_state.get('I_orn_ln_nonad')
        x_std_orn_ln_nonad = al_state.get('x_std_orn_ln_nonad')
        I_ln_pn_nonad = al_state.get('I_ln_pn_nonad')
        x_std_ln_pn_nonad = al_state.get('x_std_ln_pn_nonad')
        I_ln_pn_excit_nonad = al_state.get('I_ln_pn_excit_nonad')
        x_std_ln_pn_excit_nonad = al_state.get('x_std_ln_pn_excit_nonad')
        I_ln_ln_nonad = al_state.get('I_ln_ln_nonad')
        x_std_ln_ln_nonad = al_state.get('x_std_ln_ln_nonad')
        I_pn_ln_nonad = al_state.get('I_pn_ln_nonad')
        x_std_pn_ln_nonad = al_state.get('x_std_pn_ln_nonad')
        I_ln_orn_nonad = al_state.get('I_ln_orn_nonad')
        x_std_ln_orn_nonad = al_state.get('x_std_ln_orn_nonad')

        # Initialize KC state
        # Build zeroed KC state on first call.
        if kc_state is None:
            kc_state = self.kc_layer._init_state(batch_size, device)
        if disable_std:
            # Same STD ablation applied to the KC layer's synapses.
            kc_state = {k: v for k, v in kc_state.items() if not k.startswith('x_std')}

        I_pn_kc = kc_state['I_pn_kc']            # (batch, n_kc) PN->KC current, A.
        apl_activity = kc_state['apl_activity']  # (batch, n_apl) running APL activation state.
        refr_kc = kc_state['refr']               # (batch, n_kc) KC refractory countdown.
        # STD vesicle states (KC synapses)
        x_std_pn_kc = kc_state.get('x_std_pn_kc')          # PN->KC STD vesicle state.
        x_std_kc_kc_aa = kc_state.get('x_std_kc_kc_aa')    # KC axon->axon STD vesicle state.
        # PN→KC non-AD state
        # Optional PN->KC non-axon-dendrite contacts (3 synapses; negligible but
        # included for completeness). init_current builds the zeros if absent.
        has_pn_kc_nonad = self.kc_layer.pn_kc_nonad is not None
        if has_pn_kc_nonad:
            I_pn_kc_nonad = kc_state.get('I_pn_kc_nonad',
                self.kc_layer.pn_kc_nonad.init_current(batch_size, device))
            x_std_pn_kc_nonad = kc_state.get('x_std_pn_kc_nonad')

        # Two-compartment KCs carry separate dendritic (v_d) and axonal (v_a)
        # voltages; the single-compartment fallback uses one v.
        if self.kc_layer.use_two_compartment:
            v_d = kc_state['v_d']  # (batch, n_kc) KC dendrite membrane potential, V.
            v_a = kc_state['v_a']  # (batch, n_kc) KC axon membrane potential, V.
        else:
            v_kc = kc_state['v']

        # KC-KC recurrent state
        # Detect which of the four KC-KC compartment pathways exist in this model.
        has_kc_kc = self.kc_layer.kc_kc_aa is not None              # axon -> axon (dominant).
        has_kc_kc_ad = self.kc_layer.kc_kc_ad is not None           # axon -> dendrite.
        has_kc_kc_dd = self.kc_layer.kc_kc_dd_weights is not None   # dendrite -> dendrite.
        has_kc_kc_da = self.kc_layer.kc_kc_da_weights is not None   # dendrite -> axon.
        has_any_kc_kc = has_kc_kc or has_kc_kc_ad or has_kc_kc_dd or has_kc_kc_da
        if has_any_kc_kc:
            # Previous-step KC spikes drive all spike-based KC-KC currents.
            spk_kc_prev = kc_state.get('spk_kc_prev', torch.zeros(batch_size, self.kc_layer.n_kc, device=device))
        if has_kc_kc:
            I_kc_kc_aa = kc_state['I_kc_kc_aa']  # axon->axon recurrent current, A.
        if has_kc_kc_ad:
            I_kc_kc_ad = kc_state.get('I_kc_kc_ad', self.kc_layer.kc_kc_ad.init_current(batch_size, device))
            x_std_kc_kc_ad = kc_state.get('x_std_kc_kc_ad')

        # OR input current
        # Convert the graded ORN drive to an actual input current by multiplying by
        # the AL's learnable OR gain (stored in log space, so exp() makes it > 0).
        I_or = orn_input * torch.exp(self.antennal_lobe.log_or_gain)

        # Spike accumulators
        # Running spike-count totals over the whole window (returned at the end).
        pn_spike_count = torch.zeros(batch_size, self.antennal_lobe.n_pn, device=device)
        kc_spike_count = torch.zeros(batch_size, self.kc_layer.n_kc, device=device)

        # UNIFIED TIME LOOP: AL and KC process together
        for step in range(n_steps):
            # === ANTENNAL LOBE ===
            # [Noise 5] Intrinsic ORN receptor noise (stochastic odorant-receptor binding)
            # Per-timestep multiplicative noise on OR→ORN current
            # Models the stochasticity of odorant-receptor binding: scale the OR
            # current by (1 + Gaussian noise) independently each timestep.
            if self.antennal_lobe.orn_neurons.params.circuit_noise_enabled:
                orn_noise = torch.randn_like(I_or) * self.antennal_lobe.orn_neurons.params.orn_receptor_noise_std
                I_or_step = I_or * (1.0 + orn_noise)
            else:
                I_or_step = I_or

            # ORN dynamics (+ LN→ORN feedback from previous timestep)
            # Start ORN's total input from the (noisy) OR drive, then add inhibitory
            # LN->ORN feedback computed from the PREVIOUS step's LN spikes.
            I_orn_total = I_or_step
            if self.antennal_lobe.ln_orn is not None and I_ln_orn is not None:
                # STD vs no-STD branch: with an x_std state the synapse call returns
                # an updated current AND updated vesicle availability; without it,
                # only the current (static strength) is returned.
                if x_std_ln_orn is not None:
                    I_ln_orn, x_std_ln_orn = self.antennal_lobe.ln_orn(spk_ln_prev, I_ln_orn, x_std_ln_orn)
                else:
                    I_ln_orn = self.antennal_lobe.ln_orn(spk_ln_prev, I_ln_orn)
                I_orn_total = I_orn_total + I_ln_orn
            # LN→ORN non-AD (310 synapses, inhibitory)
            # Additional inhibitory LN->ORN drive via non-axon-dendrite contacts.
            if self.antennal_lobe.ln_orn_nonad is not None and I_ln_orn_nonad is not None:
                if x_std_ln_orn_nonad is not None:
                    I_ln_orn_nonad, x_std_ln_orn_nonad = self.antennal_lobe.ln_orn_nonad(spk_ln_prev, I_ln_orn_nonad, x_std_ln_orn_nonad)
                else:
                    I_ln_orn_nonad = self.antennal_lobe.ln_orn_nonad(spk_ln_prev, I_ln_orn_nonad)
                I_orn_total = I_orn_total + I_ln_orn_nonad
            # Integrate ORN LIF one step: returns new voltage, spikes, refractory.
            v_orn, spk_orn, refr_orn = self.antennal_lobe.orn_neurons(I_orn_total, v_orn, refr_orn)

            # --- Gap junction currents (instantaneous, voltage-dependent) ---
            # Electrical synapses: current flows proportionally to the voltage
            # DIFFERENCE between coupled neurons, I = g * (V_pre - V_post). These
            # are computed from the CURRENT voltages and applied the same step (no
            # state integration). Each masked matmul sums neighbours' voltages;
            # subtracting v * (mask row/col sum) supplies the -V_post * degree term.

            # A1: LN-LN gap junctions
            I_gap_ln = torch.zeros_like(v_ln)
            if self.antennal_lobe.gap_ln_ln_mask is not None:
                g_ln = torch.exp(self.antennal_lobe.log_g_gap_ln)  # conductance, S (>0 via exp).
                # matmul term = sum of coupled LN voltages; minus v_ln*degree gives
                # the net g * sum_j(V_j - V_i) electrical current into each LN.
                I_gap_ln = g_ln * (torch.matmul(v_ln, self.antennal_lobe.gap_ln_ln_mask)
                                    - v_ln * self.antennal_lobe.gap_ln_ln_mask.sum(1))

            # A2: PN-PN gap junctions
            g_pn_gap = torch.exp(self.antennal_lobe.log_g_gap_pn)  # conductance, S.
            # Same difference form for PN-PN electrical coupling.
            I_gap_pn = g_pn_gap * (torch.matmul(v_pn, self.antennal_lobe.gap_pn_pn_mask)
                                    - v_pn * self.antennal_lobe.gap_pn_pn_mask.sum(1))

            # A3: eLN-PN gap junctions (bidirectional)
            # Excitatory-LN <-> PN electrical coupling, computed in BOTH directions
            # with the shared conductance g_eln. The mask is (n_ln, n_pn); .T and
            # the two different sum axes give the correct degree term for each side.
            g_eln = torch.exp(self.antennal_lobe.log_g_gap_eln_pn)  # conductance, S.
            # eLN voltages flowing into PNs (PN is the post side here).
            I_gap_eln_to_pn = g_eln * (torch.matmul(v_ln, self.antennal_lobe.gap_eln_pn_mask)
                                        - v_pn * self.antennal_lobe.gap_eln_pn_mask.sum(0))
            # PN voltages flowing back into eLNs (LN is the post side here).
            I_gap_pn_to_eln = g_eln * (torch.matmul(v_pn, self.antennal_lobe.gap_eln_pn_mask.T)
                                        - v_ln * self.antennal_lobe.gap_eln_pn_mask.sum(1))

            # LN dynamics (ORN + LN→LN lateral + PN→LN feedback + gap junctions)
            # ORN->LN excitatory drive (this step's ORN spikes). If there is no
            # explicit ORN->LN connection, fall back to a pooled (mean) drive
            # broadcast to all LNs via a learnable softplus pool weight.
            if self.antennal_lobe.orn_ln is not None:
                if x_std_orn_ln is not None:
                    I_orn_ln, x_std_orn_ln = self.antennal_lobe.orn_ln(spk_orn, I_orn_ln, x_std_orn_ln)
                else:
                    I_orn_ln = self.antennal_lobe.orn_ln(spk_orn, I_orn_ln)
            else:
                I_orn_ln = spk_orn.mean(dim=-1, keepdim=True) * F.softplus(self.antennal_lobe.ln_pool_weight)
                I_orn_ln = I_orn_ln.expand(-1, self.antennal_lobe.n_ln)
            # LN total input begins with ORN drive plus the LN-LN and PN->eLN gap
            # currents already computed above.
            I_ln_total = I_orn_ln + I_gap_ln + I_gap_pn_to_eln
            # LN→LN lateral inhibition (from previous timestep's LN spikes)
            if self.antennal_lobe.ln_ln is not None and I_ln_ln is not None:
                if x_std_ln_ln is not None:
                    I_ln_ln, x_std_ln_ln = self.antennal_lobe.ln_ln(spk_ln_prev, I_ln_ln, x_std_ln_ln)
                else:
                    I_ln_ln = self.antennal_lobe.ln_ln(spk_ln_prev, I_ln_ln)
                I_ln_total = I_ln_total + I_ln_ln
            # PN→LN feedback (from previous timestep's PN spikes)
            if self.antennal_lobe.pn_ln is not None and I_pn_ln is not None:
                if x_std_pn_ln is not None:
                    I_pn_ln, x_std_pn_ln = self.antennal_lobe.pn_ln(spk_pn_prev, I_pn_ln, x_std_pn_ln)
                else:
                    I_pn_ln = self.antennal_lobe.pn_ln(spk_pn_prev, I_pn_ln)
                I_ln_total = I_ln_total + I_pn_ln
            # Non-AD contributions to LN
            # Collapsed-compartment ORN->LN, LN->LN, PN->LN additions (each with its
            # own STD-vs-no-STD branch), summed into the LN total.
            if self.antennal_lobe.orn_ln_nonad is not None and I_orn_ln_nonad is not None:
                if x_std_orn_ln_nonad is not None:
                    I_orn_ln_nonad, x_std_orn_ln_nonad = self.antennal_lobe.orn_ln_nonad(spk_orn, I_orn_ln_nonad, x_std_orn_ln_nonad)
                else:
                    I_orn_ln_nonad = self.antennal_lobe.orn_ln_nonad(spk_orn, I_orn_ln_nonad)
                I_ln_total = I_ln_total + I_orn_ln_nonad
            if self.antennal_lobe.ln_ln_nonad is not None and I_ln_ln_nonad is not None:
                if x_std_ln_ln_nonad is not None:
                    I_ln_ln_nonad, x_std_ln_ln_nonad = self.antennal_lobe.ln_ln_nonad(spk_ln_prev, I_ln_ln_nonad, x_std_ln_ln_nonad)
                else:
                    I_ln_ln_nonad = self.antennal_lobe.ln_ln_nonad(spk_ln_prev, I_ln_ln_nonad)
                I_ln_total = I_ln_total + I_ln_ln_nonad
            if self.antennal_lobe.pn_ln_nonad is not None and I_pn_ln_nonad is not None:
                if x_std_pn_ln_nonad is not None:
                    I_pn_ln_nonad, x_std_pn_ln_nonad = self.antennal_lobe.pn_ln_nonad(spk_pn_prev, I_pn_ln_nonad, x_std_pn_ln_nonad)
                else:
                    I_pn_ln_nonad = self.antennal_lobe.pn_ln_nonad(spk_pn_prev, I_pn_ln_nonad)
                I_ln_total = I_ln_total + I_pn_ln_nonad
            # Integrate LN LIF one step.
            v_ln, spk_ln, refr_ln = self.antennal_lobe.ln_neurons(I_ln_total, v_ln, refr_ln)

            # PN dynamics (with STD at ORN→PN and split LN→PN + gap junctions)
            # ORN->PN feedforward excitation driven by THIS step's ORN spikes.
            if x_std_orn_pn is not None:
                I_orn_pn, x_std_orn_pn = self.antennal_lobe.orn_pn(spk_orn, I_orn_pn, x_std_orn_pn)
            else:
                I_orn_pn = self.antennal_lobe.orn_pn(spk_orn, I_orn_pn)
            # Inhibitory LN→PN (GABAergic Broad/Choosy LNs)
            # The inhibitory LN->PN channel (sign embedded in the synapse weights).
            if x_std_ln_pn is not None:
                I_ln_pn, x_std_ln_pn = self.antennal_lobe.ln_pn(spk_ln, I_ln_pn, x_std_ln_pn)
            else:
                I_ln_pn = self.antennal_lobe.ln_pn(spk_ln, I_ln_pn)
            # Excitatory LN→PN (glutamatergic Picky LNs)
            # The separate excitatory LN->PN channel (different LN subtype).
            if x_std_ln_pn_excit is not None:
                I_ln_pn_excit, x_std_ln_pn_excit = self.antennal_lobe.ln_pn_excit(spk_ln, I_ln_pn_excit, x_std_ln_pn_excit)
            else:
                I_ln_pn_excit = self.antennal_lobe.ln_pn_excit(spk_ln, I_ln_pn_excit)
            # PNs receive both PN-PN gap current and the eLN->PN gap current.
            I_pn_gap = I_gap_pn + I_gap_eln_to_pn
            # Assemble the PN total input: feedforward + inhibitory + excitatory + gap.
            I_pn_total = I_orn_pn + I_ln_pn + I_ln_pn_excit + I_pn_gap
            # Non-AD LN→PN contributions
            # Collapsed-compartment inhibitory and excitatory LN->PN additions.
            if self.antennal_lobe.ln_pn_nonad is not None and I_ln_pn_nonad is not None:
                if x_std_ln_pn_nonad is not None:
                    I_ln_pn_nonad, x_std_ln_pn_nonad = self.antennal_lobe.ln_pn_nonad(spk_ln, I_ln_pn_nonad, x_std_ln_pn_nonad)
                else:
                    I_ln_pn_nonad = self.antennal_lobe.ln_pn_nonad(spk_ln, I_ln_pn_nonad)
                I_pn_total = I_pn_total + I_ln_pn_nonad
            if self.antennal_lobe.ln_pn_excit_nonad is not None and I_ln_pn_excit_nonad is not None:
                if x_std_ln_pn_excit_nonad is not None:
                    I_ln_pn_excit_nonad, x_std_ln_pn_excit_nonad = self.antennal_lobe.ln_pn_excit_nonad(spk_ln, I_ln_pn_excit_nonad, x_std_ln_pn_excit_nonad)
                else:
                    I_ln_pn_excit_nonad = self.antennal_lobe.ln_pn_excit_nonad(spk_ln, I_ln_pn_excit_nonad)
                I_pn_total = I_pn_total + I_ln_pn_excit_nonad

            # Integrate PN LIF one step and accumulate this step's PN spikes.
            v_pn, spk_pn, refr_pn = self.antennal_lobe.pn_neurons(I_pn_total, v_pn, refr_pn)
            pn_spike_count += spk_pn

            # Track AL previous spikes for recurrent connections
            # Save this step's LN/PN spikes so next step's recurrent (one-step-
            # delayed) connections can read them.
            spk_ln_prev = spk_ln
            spk_pn_prev = spk_pn

            # === KENYON CELLS (receive PN spikes in REAL TIME!) ===
            # PN → KC synaptic current (spike-by-spike, not rates!)
            # The KEY unified-loop feature: this step's PN spikes drive the PN->KC
            # synapse immediately, so KC coincidence detection sees real timing.
            if x_std_pn_kc is not None:
                I_pn_kc, x_std_pn_kc = self.kc_layer.pn_kc(spk_pn, I_pn_kc, x_std_pn_kc)
            else:
                I_pn_kc = self.kc_layer.pn_kc(spk_pn, I_pn_kc)

            # APL inhibition (graded, based on running KC activity + dendritic voltage)
            # Pass KC dendritic voltage for KC dendrite → APL pathway (if 2-compartment)
            # Use DIVISIVE inhibition for biologically realistic graded suppression
            # APL is GRADED (non-spiking). It is driven by the running mean KC
            # firing rate (count / steps-so-far) and, for 2-compartment KCs, the
            # dendritic voltage (the KC-dendrite->APL pathway). It returns a
            # per-KC divisive factor and its updated internal activity state.
            kc_v_dend = v_d if self.kc_layer.use_two_compartment else None
            apl_divisive, apl_activity = self.kc_layer.apl(
                kc_spike_count / max(1, step + 1), apl_activity, kc_v_dend=kc_v_dend,
                return_divisive=True
            )

            # Optogenetic APL current injection (for Mancini 2023 validation)
            # This adds constant current to APL activity, simulating optogenetic activation
            # MANCINI 2023 experiment hook: directly boost APL activity, then
            # RECOMPUTE the divisive factor from the boosted activity. The transfer
            # is ReLU(activity) * softplus(apl_gain), projected onto KCs via the
            # fixed apl_kc_weights, matching the APL unit's internal computation.
            if apl_inject_current > 0:
                apl_activity = apl_activity + apl_inject_current
                # Recompute APL divisive factor with boosted activity (ReLU transfer)
                apl_output = F.relu(apl_activity) * F.softplus(self.kc_layer.apl.apl_gain)
                apl_divisive = torch.matmul(apl_output, self.kc_layer.apl.apl_kc_weights)

            # PN→KC non-AD current (3 synapses, negligible)
            # Optional collapsed-compartment PN->KC drive (kept for completeness).
            if has_pn_kc_nonad:
                if x_std_pn_kc_nonad is not None:
                    I_pn_kc_nonad, x_std_pn_kc_nonad = self.kc_layer.pn_kc_nonad(spk_pn, I_pn_kc_nonad, x_std_pn_kc_nonad)
                else:
                    I_pn_kc_nonad = self.kc_layer.pn_kc_nonad(spk_pn, I_pn_kc_nonad)

            # Total dendritic current to KC
            # Add carbachol-like direct KC activation (if specified)
            # This simulates acetylcholine agonist that activates all KCs uniformly
            # Begin the KC dendritic input from PN->KC, add the non-AD PN->KC term,
            # then optionally add the carbachol-like uniform current injection
            # (Mancini 2023): a constant ACh-agonist drive applied to every KC.
            I_kc_total = I_pn_kc
            if has_pn_kc_nonad:
                I_kc_total = I_kc_total + I_pn_kc_nonad
            if kc_inject_current > 0:
                I_kc_total = I_kc_total + kc_inject_current

            # If disable_apl=True, skip APL inhibition (for biological validation)
            # APL ablation experiment: feed KCs the raw input with no inhibition.
            if disable_apl:
                I_kc = I_kc_total  # No APL inhibition
            else:
                # DIVISIVE inhibition: I_kc = I_input / (1 + apl_factor)
                # This is biologically realistic (shunting inhibition)
                # Provides GRADED suppression - all KCs reduced proportionally
                # Dividing (rather than subtracting) is shunting/divisive
                # inhibition: every KC's input is scaled down by 1/(1 + apl_divisive),
                # so stronger APL activity proportionally suppresses all KCs.
                I_kc = I_kc_total / (1.0 + apl_divisive)

            # KC-KC dendrite→dendrite graded current (987 synapses, excitatory)
            # Clamp activity to prevent positive feedback runaway (bio: voltage bounded)
            # graded_conductance_scale converts voltage → current (V × nS → A)
            # GRADED dendro-dendritic KC coupling: use the (rectified) dendritic
            # depolarisation above rest as the "activity", capped at 30 mV to keep
            # the excitatory feedback loop from running away. The masked matmul
            # spreads it across coupled KC dendrites, then a learnable softplus gain
            # and the voltage->current scale turn it into a dendritic current.
            if has_kc_kc_dd and self.kc_layer.use_two_compartment:
                v_rest = self.kc_layer.kc_neurons.params.v_reset
                kc_dend_activity = F.relu(v_d - v_rest).clamp(max=0.030)  # Cap at 30mV depol
                I_kc_dd = torch.matmul(kc_dend_activity, self.kc_layer.kc_kc_dd_weights)
                I_kc = I_kc + self.kc_layer.graded_conductance_scale * F.softplus(self.kc_layer.kc_kc_dd_gain) * I_kc_dd

            # KC-KC axon→dendrite spike-driven current (12 synapses)
            # Spike-driven axon->dendrite KC coupling (few synapses), added to the
            # dendritic current with the usual STD-vs-no-STD branch.
            if has_kc_kc_ad:
                if x_std_kc_kc_ad is not None:
                    I_kc_kc_ad, x_std_kc_kc_ad = self.kc_layer.kc_kc_ad(spk_kc_prev, I_kc_kc_ad, x_std_kc_kc_ad)
                else:
                    I_kc_kc_ad = self.kc_layer.kc_kc_ad(spk_kc_prev, I_kc_kc_ad)
                I_kc = I_kc + I_kc_kc_ad

            # KC-KC axon→axon recurrent current (13,621 synapses, dominant)
            # The dominant KC-KC pathway (spike-driven, axon->axon). It targets the
            # AXONAL compartment, so it is accumulated separately in I_kc_axon and
            # passed to the KC dynamics as I_axon (not added to the dendritic input).
            I_kc_axon = None
            if has_kc_kc:
                if x_std_kc_kc_aa is not None:
                    I_kc_kc_aa, x_std_kc_kc_aa = self.kc_layer.kc_kc_aa(spk_kc_prev, I_kc_kc_aa, x_std_kc_kc_aa)
                else:
                    I_kc_kc_aa = self.kc_layer.kc_kc_aa(spk_kc_prev, I_kc_kc_aa)
                I_kc_axon = I_kc_kc_aa

            # KC-KC dendrite→axon graded current (30 synapses)
            # Graded dendrite->axon KC coupling (few synapses), same capped-activity
            # construction as dd above; it also targets the axonal compartment and is
            # added into I_kc_axon (or initialises it if aa was absent).
            if has_kc_kc_da and self.kc_layer.use_two_compartment:
                v_rest = self.kc_layer.kc_neurons.params.v_reset
                kc_dend_act = F.relu(v_d - v_rest).clamp(max=0.030)
                I_kc_da = torch.matmul(kc_dend_act, self.kc_layer.kc_kc_da_weights)
                I_kc_da_scaled = self.kc_layer.graded_conductance_scale * F.softplus(self.kc_layer.kc_kc_da_gain) * I_kc_da
                I_kc_axon = I_kc_da_scaled if I_kc_axon is None else I_kc_axon + I_kc_da_scaled

            # KC dynamics
            # Integrate the KC one step. The two-compartment path advances both
            # dendrite (driven by I_kc) and axon (driven by I_axon), coupled by the
            # learnable g_soma conductance inside the neuron; spikes are read from
            # the axon. The single-compartment fallback uses one membrane equation.
            if self.kc_layer.use_two_compartment:
                v_d, v_a, spk_kc, refr_kc = self.kc_layer.kc_neurons(
                    I_kc, v_d, v_a, refr_kc, I_axon=I_kc_axon
                )
            else:
                v_kc, spk_kc, refr_kc = self.kc_layer.kc_neurons(I_kc, v_kc, refr_kc)

            # Accumulate this step's KC spikes into the running window total.
            kc_spike_count += spk_kc

            # Track previous spikes for KC-KC recurrence
            # Stash this step's KC spikes for next step's KC-KC recurrent currents.
            if has_any_kc_kc:
                spk_kc_prev = spk_kc

        # Update states
        # Repackage the final loop-local AL variables into a state dict so a caller
        # can continue the simulation (stateful/streaming use).
        al_state_new = {
            'v_orn': v_orn, 'v_ln': v_ln, 'v_pn': v_pn,
            'refr_orn': refr_orn, 'refr_ln': refr_ln, 'refr_pn': refr_pn,
            'I_orn_pn': I_orn_pn, 'I_ln_pn': I_ln_pn, 'I_ln_pn_excit': I_ln_pn_excit,
            'I_orn_ln': I_orn_ln,
            'x_std_orn_pn': x_std_orn_pn, 'x_std_ln_pn': x_std_ln_pn,
            'x_std_ln_pn_excit': x_std_ln_pn_excit, 'x_std_orn_ln': x_std_orn_ln,
            'spk_ln_prev': spk_ln_prev, 'spk_pn_prev': spk_pn_prev,
        }
        # AL recurrent state
        # Only persist recurrent / non-AD currents and their STD states for the
        # pathways that actually exist in this model.
        if self.antennal_lobe.ln_ln is not None:
            al_state_new['I_ln_ln'] = I_ln_ln
            al_state_new['x_std_ln_ln'] = x_std_ln_ln
        if self.antennal_lobe.pn_ln is not None:
            al_state_new['I_pn_ln'] = I_pn_ln
            al_state_new['x_std_pn_ln'] = x_std_pn_ln
        if self.antennal_lobe.ln_orn is not None:
            al_state_new['I_ln_orn'] = I_ln_orn
            al_state_new['x_std_ln_orn'] = x_std_ln_orn
        # Non-AD AL state
        if self.antennal_lobe.orn_ln_nonad is not None:
            al_state_new['I_orn_ln_nonad'] = I_orn_ln_nonad
            al_state_new['x_std_orn_ln_nonad'] = x_std_orn_ln_nonad
        if self.antennal_lobe.ln_pn_nonad is not None:
            al_state_new['I_ln_pn_nonad'] = I_ln_pn_nonad
            al_state_new['x_std_ln_pn_nonad'] = x_std_ln_pn_nonad
        if self.antennal_lobe.ln_pn_excit_nonad is not None:
            al_state_new['I_ln_pn_excit_nonad'] = I_ln_pn_excit_nonad
            al_state_new['x_std_ln_pn_excit_nonad'] = x_std_ln_pn_excit_nonad
        if self.antennal_lobe.ln_ln_nonad is not None:
            al_state_new['I_ln_ln_nonad'] = I_ln_ln_nonad
            al_state_new['x_std_ln_ln_nonad'] = x_std_ln_ln_nonad
        if self.antennal_lobe.pn_ln_nonad is not None:
            al_state_new['I_pn_ln_nonad'] = I_pn_ln_nonad
            al_state_new['x_std_pn_ln_nonad'] = x_std_pn_ln_nonad
        if self.antennal_lobe.ln_orn_nonad is not None:
            al_state_new['I_ln_orn_nonad'] = I_ln_orn_nonad
            al_state_new['x_std_ln_orn_nonad'] = x_std_ln_orn_nonad

        # Repackage KC state; the dict keys differ between the two-compartment
        # (v_d, v_a) and single-compartment (v) cases.
        if self.kc_layer.use_two_compartment:
            kc_state_new = {
                'v_d': v_d, 'v_a': v_a, 'refr': refr_kc,
                'I_pn_kc': I_pn_kc, 'apl_activity': apl_activity,
                'x_std_pn_kc': x_std_pn_kc,
            }
        else:
            kc_state_new = {
                'v': v_kc, 'refr': refr_kc,
                'I_pn_kc': I_pn_kc, 'apl_activity': apl_activity,
                'x_std_pn_kc': x_std_pn_kc,
            }

        # Persist KC-KC and PN->KC-nonAD state only for present pathways.
        if has_any_kc_kc:
            kc_state_new['spk_kc_prev'] = spk_kc_prev
        if has_kc_kc:
            kc_state_new['I_kc_kc_aa'] = I_kc_kc_aa
            kc_state_new['x_std_kc_kc_aa'] = x_std_kc_kc_aa
        if has_kc_kc_ad:
            kc_state_new['I_kc_kc_ad'] = I_kc_kc_ad
            kc_state_new['x_std_kc_kc_ad'] = x_std_kc_kc_ad
        if has_pn_kc_nonad:
            kc_state_new['I_pn_kc_nonad'] = I_pn_kc_nonad
            kc_state_new['x_std_pn_kc_nonad'] = x_std_pn_kc_nonad

        return pn_spike_count, kc_spike_count, al_state_new, kc_state_new

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
            metrics: Dictionary with individual terms

        NOTE (LEGACY PATH): This in-class loss is NOT what produced the paper's
        results. The run_*.py training drivers compute their own objective (a
        sparsity proxy with sigmoid offset 0.02 and target 0.05, plus N_STEPS=30).
        This method is retained for the __main__ smoke test and illustration; it
        uses the constructor's legacy target_sparsity (0.10) and step defaults.
        """
        # Forward pass
        # Run the canonical unified forward and grab the intermediates we need.
        logits, intermediates = self.forward(or_input, return_all=True)

        # Task loss
        # Standard supervised classification loss over the 28 odor classes.
        task_loss = F.cross_entropy(logits, odor_labels)

        # Sparsity loss (differentiable proxy)
        # Penalise deviation of the KC population sparsity from the target. Because
        # a hard "is the KC active?" count is non-differentiable, a sharp sigmoid
        # around a 5%-rate threshold gives a soft, gradient-friendly active mask.
        kc_spikes = intermediates['kc_spikes']
        n_steps = max(self.n_steps_al, self.n_steps_kc)  # Unified simulation uses this
        kc_rates = kc_spikes / n_steps
        # Sparsity = fraction of KCs with any activity (rate > 0)
        # Use soft threshold for differentiability
        soft_active = torch.sigmoid((kc_rates - 0.05) * 50.0)  # Sharp sigmoid around 5% rate
        # Mean over KCs and batch = soft fraction of "active" KCs.
        diff_sparsity = soft_active.mean()
        # Squared deviation from the (legacy) target sparsity.
        sparsity_loss = (diff_sparsity - self.target_sparsity) ** 2

        # Combined loss
        # Total = task term + weighted sparsity regulariser.
        total_loss = task_loss + sparsity_weight * sparsity_loss

        # Metrics
        # Accuracy is a reporting-only metric, so compute it without tracking grads.
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == odor_labels).float().mean().item()

        # Detach everything to plain Python floats for logging.
        metrics = {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'sparsity': intermediates['sparsity'],
            'accuracy': accuracy,
            'pn_spike_rate': intermediates['pn_spikes'].mean().item() / n_steps,
            'kc_spike_rate': kc_spikes.mean().item() / n_steps,
        }

        return total_loss, metrics

    def clamp_to_biological_bounds(self):
        """
        Clamp all learnable parameters to biologically realistic bounds.

        This should be called after each optimizer step to ensure parameters
        remain within biologically meaningful ranges.

        Clamped parameters (these are the wide default bounds in layers.py; the
        canonical training pipeline in scripts/run_training.py further narrows g_soma to
        the [1, 20] nS biological range reported in the paper):
        - v_th: [-55, -30] mV for all neuron populations
        - log_tau_m: [5, 50] ms for membrane time constants
        - log_g_soma: [1, 100] nS for KC dendritic-axonal coupling (trained range [1, 20] nS)
        - log_tau_apl: [10, 50] ms for APL time constant
        - log_strength: [1e-12, 1e-7] A for synaptic strengths

        Side effects:
            Mutates the antennal lobe and KC layer parameters IN PLACE (under
            no_grad inside those submodules). Returns nothing. Intended to be
            invoked by the training loop immediately after optimizer.step().
        """
        # Clamp antennal lobe parameters
        # Delegate the actual per-parameter clamping to each submodule, which knows
        # its own thresholds, time constants, gap conductances, and synapse strengths.
        self.antennal_lobe.clamp_to_biological_bounds()

        # Clamp KC layer parameters
        # Includes the KC thresholds, g_soma coupling, APL time constant, and the
        # KC/PN synaptic strengths.
        self.kc_layer.clamp_to_biological_bounds()

    @classmethod
    def from_data_dir(
        cls,
        data_dir: Path,
        n_odors: int,
        n_or_types: int = 21,
        include_nonad: bool = False,
        **kwargs
    ) -> 'SpikingConnectomeConstrainedModel':
        """Load model from saved connectome tensors.

        Reads the fixed Winding 2023 connectome tensors from disk, assembles the
        `connectome` dict, and constructs the model. Connectivity is biological and
        fixed; only the synapse strengths/thresholds/etc. on top of it are learned.

        Args:
            data_dir: Root directory containing 'winding2023' (and optionally
                'winding2023_compartments') subfolders of .pt tensors.
            n_odors: Number of odor classes (decoder output width).
            n_or_types: Number of OR types (21 larval ORs).
            include_nonad: If True, also load the non-axon-dendrite collapsed
                contact matrices (opt-in; off by default).
            **kwargs: Forwarded to __init__ (e.g. params, n_steps_*, surrogate_method).

        Returns:
            A constructed SpikingConnectomeConstrainedModel.

        Side effects:
            Performs disk reads via torch.load(..., weights_only=True).
        """
        # Core feedforward + APL connectome lives under winding2023/.
        winding_dir = data_dir / "winding2023"

        # The six REQUIRED matrices defining the canonical feedforward path plus
        # the APL feedback loop. weights_only=True is the safe load mode.
        connectome = {
            'orn_to_pn': torch.load(winding_dir / "orn_to_pn.pt", weights_only=True),
            'ln_to_pn': torch.load(winding_dir / "ln_to_pn.pt", weights_only=True),
            'orn_to_ln': torch.load(winding_dir / "orn_to_ln.pt", weights_only=True),
            'pn_to_kc': torch.load(winding_dir / "pn_to_kc.pt", weights_only=True),
            'kc_to_apl': torch.load(winding_dir / "kc_to_apl.pt", weights_only=True),
            'apl_to_kc': torch.load(winding_dir / "apl_to_kc.pt", weights_only=True),
        }

        # AL recurrent connections (collapsed)
        # Optional AL recurrence (LN->LN lateral, PN->LN feedback, LN->ORN feedback).
        # Loaded only if the corresponding file is present.
        for name in ['ln_to_ln', 'pn_to_ln', 'ln_to_orn']:
            path = winding_dir / f"{name}.pt"
            if path.exists():
                connectome[name] = torch.load(path, weights_only=True)

        # Non-AD (non-axon-dendrite) collapsed connectivity (opt-in only)
        # These collapsed non-axon-dendrite contact matrices are loaded ONLY when
        # the caller opts in via include_nonad=True (and the file exists).
        if include_nonad:
            for name in ['orn_to_ln_nonad', 'ln_to_pn_nonad', 'ln_to_ln_nonad', 'pn_to_ln_nonad', 'pn_to_kc_nonad', 'ln_to_orn_nonad']:
                path = winding_dir / f"{name}.pt"
                if path.exists():
                    connectome[name] = torch.load(path, weights_only=True)

        # Load compartment-resolved connectivity if available
        # Compartment-resolved KC graphs live in a separate folder; each is loaded
        # only if its file is present, so the model degrades gracefully.
        compartment_dir = data_dir / "winding2023_compartments"
        if (compartment_dir / "kc_to_kc_aa.pt").exists():
            connectome['kc_to_kc_aa'] = torch.load(
                compartment_dir / "kc_to_kc_aa.pt", weights_only=True
            )
        # KC dendrite → APL axon (graded dendritic contribution to APL)
        if (compartment_dir / "kc_to_apl_da.pt").exists():
            connectome['kc_to_apl_da'] = torch.load(
                compartment_dir / "kc_to_apl_da.pt", weights_only=True
            )
        # KC dendrite → KC dendrite (graded dendritic coupling, 987 synapses)
        if (compartment_dir / "kc_to_kc_dd.pt").exists():
            connectome['kc_to_kc_dd'] = torch.load(
                compartment_dir / "kc_to_kc_dd.pt", weights_only=True
            )
        # KC axon → KC dendrite (12 synapses)
        if (compartment_dir / "kc_to_kc_ad.pt").exists():
            connectome['kc_to_kc_ad'] = torch.load(
                compartment_dir / "kc_to_kc_ad.pt", weights_only=True
            )
        # KC dendrite → KC axon (30 synapses)
        if (compartment_dir / "kc_to_kc_da.pt").exists():
            connectome['kc_to_kc_da'] = torch.load(
                compartment_dir / "kc_to_kc_da.pt", weights_only=True
            )

        # Hand the assembled connectome to the constructor.
        return cls(connectome, n_odors, n_or_types=n_or_types, **kwargs)


# Smoke test / usage example. Builds a tiny RANDOM connectome (not the real
# Winding data), runs one forward pass and one loss computation, and prints
# shapes/metrics. Useful as a quick "does it run end-to-end?" check.
if __name__ == "__main__":
    print("=" * 60)
    print("Testing SpikingConnectomeConstrainedModel")
    print("=" * 60)

    # Create dummy connectome
    # Random integer-count adjacency matrices with the right shapes for a 72-KC
    # toy model (42 ORN, 21 PN, 10 LN, 2 APL, plus a KC axon->axon graph).
    n_kc = 72
    connectome = {
        'orn_to_pn': torch.randint(0, 10, (42, 21)).float(),
        'ln_to_pn': torch.randint(0, 5, (10, 21)).float(),
        'orn_to_ln': torch.randint(0, 5, (42, 10)).float(),
        'pn_to_kc': torch.randint(0, 5, (21, n_kc)).float(),
        'kc_to_apl': torch.randint(0, 3, (n_kc, 2)).float(),
        'apl_to_kc': torch.randint(0, 3, (2, n_kc)).float(),
        'kc_to_kc_aa': torch.randint(0, 3, (n_kc, n_kc)).float(),
    }

    # Create model (with KC-KC)
    # Use the canonical 30-step sim for the smoke test.
    model = SpikingConnectomeConstrainedModel(
        connectome, n_odors=28, n_or_types=21,
        n_steps_al=30, n_steps_kc=30,  # canonical 30-step sim
    )

    # Test forward pass
    # One batch of 4 random positive OR response vectors (21 OR types each).
    batch_size = 4
    or_input = torch.randn(batch_size, 21).abs()  # 21 OR types
    logits, info = model.forward(or_input, return_all=True)

    print(f"\nForward pass:")
    print(f"  OR input shape: {or_input.shape}")
    print(f"  PN spike count shape: {info['pn_spikes'].shape}")
    print(f"  KC spike count shape: {info['kc_spikes'].shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  KC sparsity: {info['sparsity']:.2%}")

    # Test loss computation
    # Random class labels to exercise the legacy compute_loss path.
    labels = torch.randint(0, 28, (batch_size,))
    loss, metrics = model.compute_loss(or_input, labels)

    print(f"\nLoss computation:")
    print(f"  Total loss: {metrics['total_loss']:.4f}")
    print(f"  Task loss: {metrics['task_loss']:.4f}")
    print(f"  Sparsity: {metrics['sparsity']:.2%}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  PN spike rate: {metrics['pn_spike_rate']:.4f}")
    print(f"  KC spike rate: {metrics['kc_spike_rate']:.4f}")

    # Count parameters
    # Report the total number of trainable parameters (the ~449 biological params
    # plus the decoder weights for this toy configuration).
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal learnable parameters: {n_params:,}")

    print("\nSpiking model test passed!")
