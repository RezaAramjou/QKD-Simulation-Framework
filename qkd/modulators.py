# qkd/modulators.py
# -*- coding: utf-8 -*-
"""
Optical modulator models.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .exceptions import ParameterValidationError
from .constants import NUMERIC_ABS_TOL


@dataclass(slots=True)
class MZMConfig:
    """
    Mach-Zehnder modulator model.

    The output mean photon number is modeled as:
    $$
    \mu_{\mathrm{out}} = \mu_{\min} + (\mu_{\max} - \mu_{\min}) \cos^2(\phi)
    $$
    where
    $$
    \phi = \frac{\pi V}{2V_\pi} + \phi_{\mathrm{bias}}.
    $$
    """
    v_pi: float = 3.5
    phi_bias: float = np.pi / 2.0
    max_intensity_mu: float = 1.0
    extinction_ratio_db: float | None = None

    def validate(self) -> None:
        if not math.isfinite(self.v_pi) or self.v_pi <= 0:
            raise ParameterValidationError(
                f"v_pi must be a positive finite number, got {self.v_pi!r}."
            )
        if not math.isfinite(self.max_intensity_mu) or self.max_intensity_mu < 0:
            raise ParameterValidationError(
                f"max_intensity_mu must be a non-negative finite number, "
                f"got {self.max_intensity_mu!r}."
            )
        if not math.isfinite(self.phi_bias):
            raise ParameterValidationError(
                f"phi_bias must be a finite number, got {self.phi_bias!r}."
            )
        if self.extinction_ratio_db is not None:
            if not math.isfinite(self.extinction_ratio_db) or self.extinction_ratio_db < 0:
                raise ParameterValidationError(
                    f"extinction_ratio_db must be a non-negative finite number, "
                    f"got {self.extinction_ratio_db!r}."
                )

    def leakage_mu(self) -> float:
        self.validate()
        if self.extinction_ratio_db is None:
            return 0.0
        return float(self.max_intensity_mu * (10.0 ** (-self.extinction_ratio_db / 10.0)))

    def apply(self, target_mus: np.ndarray, voltage_noise_std: float, rng) -> np.ndarray:
        self.validate()

        if not math.isfinite(voltage_noise_std) or voltage_noise_std < 0.0:
            raise ParameterValidationError(
                f"voltage_noise_std must be a non-negative finite number, "
                f"got {voltage_noise_std!r}."
            )

        i_min = self.leakage_mu()
        i_max = self.max_intensity_mu
        delta_i = i_max - i_min

        target = np.clip(np.asarray(target_mus, dtype=np.float64), i_min, i_max)

        if delta_i < NUMERIC_ABS_TOL:
            normalized = np.zeros_like(target)
        else:
            normalized = np.clip((target - i_min) / delta_i, 0.0, 1.0)

        theta_target = np.arccos(np.sqrt(normalized))
        v_signal = (2.0 * self.v_pi / np.pi) * (theta_target - self.phi_bias)

        if voltage_noise_std > 0.0:
            v_noise = rng.normal(loc=0.0, scale=voltage_noise_std, size=target.shape)
        else:
            v_noise = 0.0

        v_total = v_signal + v_noise
        noisy_phase = (np.pi * v_total) / (2.0 * self.v_pi) + self.phi_bias
        return i_min + delta_i * (np.cos(noisy_phase) ** 2)

    def to_dict(self) -> dict:
        return {
            "v_pi": self.v_pi,
            "phi_bias": self.phi_bias,
            "max_intensity_mu": self.max_intensity_mu,
            "extinction_ratio_db": self.extinction_ratio_db,
        }