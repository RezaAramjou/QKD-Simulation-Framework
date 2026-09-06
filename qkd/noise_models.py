# qkd/noise_models.py
# -*- coding: utf-8 -*-
"""
Electrical noise models for source-side driver / monitor paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .exceptions import ParameterValidationError
from .constants import CONST_BOLTZMANN, CONST_ELECTRON_CHARGE


@dataclass(slots=True)
class ElectricalNoiseConfig:
    """
    Electrical noise configuration for source-side modeling.

    Notes
    -----
    - Thermal noise is always computed from resistor temperature and bandwidth.
    - Shot noise is included only if `monitor_photocurrent_a` is provided,
      representing a monitor photodiode current.
    """
    temperature_k: float = 298.15
    bandwidth_hz: float = 1e9
    load_resistance_ohm: float = 50.0
    monitor_photocurrent_a: Optional[float] = None

    def validate(self) -> None:
        if self.temperature_k <= 0:
            raise ParameterValidationError("temperature_k must be positive.")
        if self.bandwidth_hz < 0:
            raise ParameterValidationError("bandwidth_hz must be non-negative.")
        if self.load_resistance_ohm <= 0:
            raise ParameterValidationError("load_resistance_ohm must be positive.")
        if self.monitor_photocurrent_a is not None and self.monitor_photocurrent_a < 0:
            raise ParameterValidationError("monitor_photocurrent_a must be non-negative.")

    def thermal_voltage_std(self) -> float:
        self.validate()
        return float(
            np.sqrt(
                4.0
                * CONST_BOLTZMANN
                * self.temperature_k
                * self.bandwidth_hz
                * self.load_resistance_ohm
            )
        )

    def shot_voltage_std(self) -> float:
        self.validate()
        if self.monitor_photocurrent_a is None or self.monitor_photocurrent_a == 0.0:
            return 0.0

        i_std = np.sqrt(
            2.0
            * CONST_ELECTRON_CHARGE
            * self.monitor_photocurrent_a
            * self.bandwidth_hz
        )
        return float(i_std * self.load_resistance_ohm)

    def total_voltage_std(self) -> float:
        vt = self.thermal_voltage_std()
        vs = self.shot_voltage_std()
        return float(np.sqrt(vt * vt + vs * vs))

    def to_dict(self) -> dict:
        return {
            "temperature_k": self.temperature_k,
            "bandwidth_hz": self.bandwidth_hz,
            "load_resistance_ohm": self.load_resistance_ohm,
            "monitor_photocurrent_a": self.monitor_photocurrent_a,
        }

