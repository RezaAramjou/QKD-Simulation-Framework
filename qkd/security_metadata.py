# qkd/security_metadata.py
# -*- coding: utf-8 -*-
"""
Security-related metadata for source components.

This module intentionally keeps protocol/security bookkeeping separate
from physical source emission models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .datatypes import (
    ConfidenceBoundMethod,
    DoubleClickPolicy,
    DecoderArchitecture,
    SecurityProof,
    SimulationStatus,
    TallyCounts,
    EpsilonAllocation,
    SecurityCertificate,
    SimulationResults,
    ProtocolParameters,
    ProtocolType,
    IntensityConfig,
    IntensityNode,
    AttenuationConfig,
    OpticalComponent,
    DetectionConfig,
    DetectorType,
    ErrorCorrectionConfig,
)


@dataclass(slots=True)
class SourceSecurityMetadata:
    expected_decoder: DecoderArchitecture = DecoderArchitecture.LOCAL
    assumed_double_click: DoubleClickPolicy = DoubleClickPolicy.RANDOM
    intended_proof: SecurityProof = SecurityProof.LIM_2014
    confidence_method: ConfidenceBoundMethod = ConfidenceBoundMethod.GAUSSIAN
    epsilon_budget: Optional[EpsilonAllocation] = None
    internal_tallies: Optional[TallyCounts] = None
    latest_certificate: Optional[SecurityCertificate] = None
    latest_results: Optional[SimulationResults] = None
    status: SimulationStatus = SimulationStatus.OK

    def ensure_defaults(self) -> None:
        if self.internal_tallies is None:
            self.internal_tallies = TallyCounts()

        if self.epsilon_budget is None:
            eps_base = 1e-12
            self.epsilon_budget = EpsilonAllocation(
                eps_sec=1e-9,
                eps_cor=eps_base,
                eps_pe=eps_base,
                eps_smooth=eps_base,
                eps_pa=eps_base,
                eps_phase_est=eps_base,
            )

    def update_sent(self, n: int) -> None:
        self.ensure_defaults()
        self.internal_tallies.sent += int(n)

    def placeholder_results(
        self,
        max_intensity_mu: float,
        source_rate_hz: float,
    ) -> SimulationResults:
        """
        Create a structurally valid placeholder SimulationResults object.

        This is useful during source-level simulations before the full
        protocol-level pipeline is connected.
        """
        self.ensure_defaults()

        cert = SecurityCertificate(
            proof_name=self.intended_proof,
            confidence_bound_method=self.confidence_method,
            assumed_phase_equals_bit_error=True,
            epsilon_allocation=self.epsilon_budget,
            lp_solver_diagnostics=None,
        )
        self.latest_certificate = cert

        params = ProtocolParameters(
            protocol=ProtocolType.BB84,
            intensities=IntensityConfig(
                signal=IntensityNode(mu=max_intensity_mu, probability=1.0),
                decoys=[],
            ),
            attenuation=AttenuationConfig(
                fiber_length=0.0,
                attenuation_coefficient=0.2,
            ),
            optical=OpticalComponent(
                source_rate=source_rate_hz,
            ),
            detection=DetectionConfig(
                efficiency=1.0,
                dark_count_rate=0.0,
                detector_type=DetectorType.SPD,
            ),
            error_correction=ErrorCorrectionConfig(
                efficiency=1.0,
            ),
        )

        results = SimulationResults(
            params=params,
            metadata={
                "component": "source",
                "placeholder": True,
            },
            security_certificate=cert,
            decoy_estimates={},
            secure_key_length=0,
            raw_sifted_key_length=0,
            simulation_time_seconds=0.0,
            status=SimulationStatus.OK,
            tally_stats={"source": self.internal_tallies},
            weak_gllp_rate=0.0,
            tagged_fraction_bound=None,
            beta_y1_deviation=0.0,
            beta_e1_deviation=0.0,
            success_probability=None,
        )
        self.latest_results = results
        self.status = SimulationStatus.OK
        return results

    def to_dict(self) -> dict:
        self.ensure_defaults()
        return {
            "expected_decoder": self.expected_decoder.value,
            "assumed_double_click": self.assumed_double_click.value,
            "intended_proof": self.intended_proof.value,
            "confidence_method": self.confidence_method.value,
            "epsilon_budget": self.epsilon_budget.to_dict() if self.epsilon_budget else None,
            "internal_tallies": self.internal_tallies.to_dict() if self.internal_tallies else None,
            "status": self.status.value,
        }

