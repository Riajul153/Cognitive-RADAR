"""Monopulse angle estimation for phased array tracking.

Simulates a realistic monopulse radar sensor that produces noisy angular
error estimates from sum (Σ) and difference (Δ) beam patterns.  The
processor uses the **known array geometry** to construct the Σ and Δ
beams — this is standard radar signal processing, not privileged
information.

Physics
-------
A monopulse radar forms three simultaneous beams:

- **Σ (Sum)**: Standard pencil beam — the regular beamformer output.
- **Δ_el (Elevation Difference)**: Upper half of the array is weighted
  +1, lower half −1.  The pattern has a null at boresight.
- **Δ_az (Azimuth Difference)**: Right half +1, left half −1.

The ratio  Re(Δ · Σ*) / |Σ|²  produces a signed, approximately linear
error signal within the main lobe.  The slope of this curve (the
*monopulse sensitivity*) is  k_m ≈ 1.6 / HPBW  for a uniformly
illuminated aperture.

Outside the main lobe, |Σ| drops, the effective SNR collapses, and the
Δ/Σ ratio becomes dominated by thermal noise.  This realistic
degradation forces the RL agent to learn a search-then-track strategy.

Noise model
-----------
Measurement noise on the angular estimate scales as:

    σ_angle ≈ 1 / ( k_m · √(2 · SNR_eff) )

where  SNR_eff = SNR_base × normalized_beam_power  models the fact that
off-axis reception has lower effective SNR.
"""

from __future__ import annotations

import numpy as np
from .array import UniformPlanarArray
from .beamformer import Beamformer


class MonopulseProcessor:
    """Simulates monopulse angle estimation for a Uniform Planar Array.

    Produces noisy elevation and azimuth error signals plus a received
    power measurement — all quantities that a real radar could measure.
    """

    def __init__(
        self,
        array: UniformPlanarArray,
        snr_db: float = 20.0,
    ) -> None:
        """Initialise the monopulse processor.

        Args:
            array: Antenna array geometry (known hardware).
            snr_db: Base signal-to-noise ratio in dB.  Controls the
                measurement noise floor.
        """
        self.array = array
        self.snr_db = float(snr_db)
        self.snr_linear = 10.0 ** (self.snr_db / 10.0)

        # Half-power beamwidth (radians) — used for noise scaling
        self.hpbw_rad: float = array.half_power_beamwidth

        # Monopulse slope (theoretical, uniform aperture):  k_m ≈ 1.6 / HPBW
        self.monopulse_slope: float = 1.6 / max(self.hpbw_rad, 1e-6)

        # Maximum reportable error — beyond this, monopulse is meaningless
        self.max_error_rad: float = 2.0 * self.hpbw_rad

        # Precompute sign masks for the difference beams
        self._el_signs = self._build_elevation_signs()
        self._az_signs = self._build_azimuth_signs()

    # ── Sign masks ───────────────────────────────────────────────────────

    def _build_elevation_signs(self) -> np.ndarray:
        """Sign mask for the elevation difference beam.

        Upper half of the UPA rows → +1, lower half → −1.
        """
        signs = np.ones((self.array.n_rows, self.array.n_cols))
        mid = self.array.n_rows // 2
        signs[:mid, :] = -1.0
        return signs.flatten()

    def _build_azimuth_signs(self) -> np.ndarray:
        """Sign mask for the azimuth difference beam.

        Right half of the UPA columns → +1, left half → −1.
        """
        signs = np.ones((self.array.n_rows, self.array.n_cols))
        mid = self.array.n_cols // 2
        signs[:, :mid] = -1.0
        return signs.flatten()

    # ── Core measurement ─────────────────────────────────────────────────

    def compute_error_signals(
        self,
        beamformer: Beamformer,
        target_theta: float,
        target_phi: float,
        rng: np.random.RandomState,
    ) -> tuple[float, float, float]:
        """Compute noisy monopulse error signals and received power.

        All three outputs are **realistically measurable** by a radar
        receiver.  The ``target_theta`` / ``target_phi`` arguments are
        used internally by the *simulator* to compute the actual signal
        the array would receive; they are never exposed to the agent.

        Args:
            beamformer: Current beamformer state (weights already set).
            target_theta: True target elevation (rad) — sim-internal.
            target_phi: True target azimuth (rad) — sim-internal.
            rng: NumPy random state for reproducible noise.

        Returns:
            Tuple ``(err_el, err_az, received_power)``:

            - **err_el**: Noisy elevation error estimate (rad).
              Positive ≈ target is above beam centre.
            - **err_az**: Noisy azimuth error estimate (rad).
              Positive ≈ target is to the right of beam centre.
            - **received_power**: Normalised Σ-beam power ∈ [0, ~1].
              Proportional to array gain at the target direction.
        """
        weights = beamformer.get_current_weights()
        sv = self.array.get_steering_vector(target_theta, target_phi)

        # ── Sum beam response ────────────────────────────────────────
        sigma = np.dot(weights, sv)
        sigma_power = float(np.abs(sigma) ** 2)

        # ── Difference beam responses ────────────────────────────────
        delta_el = np.dot(self._el_signs * weights, sv)
        delta_az = np.dot(self._az_signs * weights, sv)

        # ── Monopulse ratios ─────────────────────────────────────────
        if sigma_power > 1e-12:
            err_el_ratio = float(np.real(delta_el * np.conj(sigma)) / sigma_power)
            err_az_ratio = float(np.real(delta_az * np.conj(sigma)) / sigma_power)
        else:
            # Sum beam is essentially zero — no useful signal
            err_el_ratio = 0.0
            err_az_ratio = 0.0

        # Convert monopulse ratio → angular error estimate
        err_el_angle = err_el_ratio / self.monopulse_slope
        err_az_angle = err_az_ratio / self.monopulse_slope

        # ── Measurement noise ────────────────────────────────────────
        # Effective SNR drops when beam is off-target (lower received power)
        normalised_power = sigma_power / max(self.array.n_elements, 1)
        effective_snr = self.snr_linear * max(normalised_power, 0.01)

        # Cramér-Rao-like noise floor:  σ ≈ 1 / (k_m · √(2 · SNR_eff))
        noise_std = 1.0 / (self.monopulse_slope * np.sqrt(2.0 * effective_snr))

        err_el_noisy = err_el_angle + float(rng.normal(0.0, noise_std))
        err_az_noisy = err_az_angle + float(rng.normal(0.0, noise_std))

        # Clip to ±max_error — outside this range, monopulse is meaningless
        err_el_noisy = float(np.clip(err_el_noisy, -self.max_error_rad, self.max_error_rad))
        err_az_noisy = float(np.clip(err_az_noisy, -self.max_error_rad, self.max_error_rad))

        # ── Received power (with thermal noise) ─────────────────────
        noise_power = sigma_power / max(self.snr_linear, 1e-6)
        noisy_power = sigma_power + float(rng.normal(0.0, np.sqrt(max(noise_power, 1e-12))))
        noisy_power = max(noisy_power, 0.0)
        normalised_received = float(np.clip(noisy_power / max(self.array.n_elements, 1), 0.0, 1.5))

        return err_el_noisy, err_az_noisy, normalised_received
