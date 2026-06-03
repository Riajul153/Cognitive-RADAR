"""Beamformer simulation for Uniform Planar Arrays."""

from __future__ import annotations

import numpy as np
from .array import UniformPlanarArray


class Beamformer:
    """Manages the complex weights and computes beam patterns for a phased array."""

    def __init__(self, array: UniformPlanarArray):
        """Initializes the Beamformer with an antenna array.

        Args:
            array: A UniformPlanarArray instance.
        """
        self.array = array
        # Default weights: uniform amplitude normalized to unit power (1/sqrt(N)), zero phase
        self._weights = np.ones(self.array.n_elements, dtype=complex) / np.sqrt(self.array.n_elements)

    def set_weights(self, phases: np.ndarray, amplitudes: np.ndarray | None = None) -> None:
        """Sets the complex weights of the array.

        Args:
            phases: Array of phase values in radians, shape (N,).
            amplitudes: Array of amplitude values, shape (N,). If None, uniform amplitudes
                are used, normalized such that the sum of squares is 1.0.
        """
        phases = np.asarray(phases, dtype=float)
        assert phases.shape == (self.array.n_elements,), f"phases must have shape ({self.array.n_elements},)"

        if amplitudes is None:
            # Set uniform amplitude normalized such that sum(|w|^2) = 1.0
            amplitudes = np.ones(self.array.n_elements) / np.sqrt(self.array.n_elements)
        else:
            amplitudes = np.asarray(amplitudes, dtype=float)
            assert amplitudes.shape == (self.array.n_elements,), f"amplitudes must have shape ({self.array.n_elements},)"
            # Ensure proper power normalization if weights represent fractional power allocation
            total_power = np.sum(amplitudes**2)
            if total_power > 0:
                amplitudes = amplitudes / np.sqrt(total_power)

        self._weights = amplitudes * np.exp(1j * phases)

    def get_current_weights(self) -> np.ndarray:
        """Returns the current complex weights."""
        return self._weights

    def get_current_phases(self) -> np.ndarray:
        """Returns the phase angle of the current complex weights in radians, wrapped to [-π, π]."""
        return np.angle(self._weights)

    def compute_array_factor(self, theta: np.ndarray | float, phi: np.ndarray | float) -> np.ndarray | complex:
        """Computes the complex Array Factor (AF) in the specified direction(s).

        AF(θ, φ) = sum_n( w_n * v_n(θ, φ) )

        Args:
            theta: Elevation angle(s) in radians. Can be scalar or numpy array.
            phi: Azimuth angle(s) in radians. Can be scalar or numpy array.

        Returns:
            The complex array factor. Matching the input shape of theta/phi.
        """
        is_scalar = np.isscalar(theta) and np.isscalar(phi)
        
        theta_arr = np.atleast_1d(theta).flatten()
        phi_arr = np.atleast_1d(phi).flatten()
        
        # Get steering vectors for the batch of directions, shape (M, N)
        v_batch = self.array.get_steering_vector_batch(theta_arr, phi_arr)
        
        # Compute dot product along elements axis: (M, N) x (N,) -> (M,)
        # Note: we use w^T * v. If using conjugate-phase weights, w_n = 1/sqrt(N) * v_n*,
        # so w^T * v = 1/sqrt(N) * sum(|v_n|^2) = sqrt(N).
        af = np.dot(v_batch, self._weights)
        
        if is_scalar:
            return af[0]
        
        # Restore shape if inputs were arrays
        if isinstance(theta, np.ndarray):
            return af.reshape(theta.shape)
        return af

    def compute_beam_pattern(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Computes the normalized beam pattern power |AF(θ, φ)|^2 / N.

        At peak direction under conjugate phase, |AF|^2 = N, so dividing by N
        normalizes the peak gain to 1.0 (0 dB).

        Args:
            theta: Array of elevation angles in radians.
            phi: Array of azimuth angles in radians.

        Returns:
            A numpy array of normalized beam power values in [0, 1].
        """
        af = self.compute_array_factor(theta, phi)
        # Power |AF|^2
        power = np.abs(af) ** 2
        # Normalize by array size N so maximum gain is 1.0
        return power / self.array.n_elements

    def compute_gain_at_target(self, target_theta: float, target_phi: float) -> float:
        """Computes the normalized beam pattern power gain at a single target direction.

        Args:
            target_theta: Target elevation in radians.
            target_phi: Target azimuth in radians.

        Returns:
            Normalized power gain in [0, 1].
        """
        af = self.compute_array_factor(target_theta, target_phi)
        return float(np.abs(af) ** 2 / self.array.n_elements)

    def get_beam_peak_direction(self, resolution_deg: float = 1.0) -> tuple[float, float]:
        """Finds the elevation and azimuth coordinates of the beam pattern peak direction.

        Uses a grid search over the visible forward hemisphere (theta in [0, pi/2], phi in [0, 2pi]).

        Args:
            resolution_deg: Grid search angular resolution in degrees.

        Returns:
            Tuple (theta_peak, phi_peak) in radians.
        """
        theta_steps = int(90.0 / resolution_deg) + 1
        phi_steps = int(360.0 / resolution_deg) + 1

        theta_grid = np.linspace(0.0, np.pi / 2.0, theta_steps)
        phi_grid = np.linspace(0.0, 2.0 * np.pi, phi_steps)

        # Create 2D coordinate grid
        theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid, indexing="ij")
        
        # Compute normalized beam power on the grid
        power_grid = self.compute_beam_pattern(theta_mesh, phi_mesh)

        # Find index of maximum power
        max_idx = np.unravel_index(np.argmax(power_grid), power_grid.shape)
        
        return theta_mesh[max_idx], phi_mesh[max_idx]

    def estimate_beam_direction(self) -> tuple[float, float]:
        """Analytically estimates the beam pointing direction (theta, phi) from the current phase weights.

        Computes spatial phase gradients across UPA rows and columns using complex weight relationships.
        This provides a fast, O(N) closed-form calculation of the primary steered beam direction
        without requiring an expensive grid search.

        Returns:
            Tuple (theta_est, phi_est) in radians.
        """
        # Reshape weights to UPA grid (n_rows, n_cols)
        W_grid = self._weights.reshape((self.array.n_rows, self.array.n_cols))

        # Calculate phase gradient in x-axis (across rows)
        # We sum complex products of adjacent elements to average out noise/phase wrap jumps
        grad_x = np.angle(np.sum(W_grid[1:, :] * np.conj(W_grid[:-1, :])))
        # Calculate phase gradient in y-axis (across columns)
        grad_y = np.angle(np.sum(W_grid[:, 1:] * np.conj(W_grid[:, :-1])))

        # Relation:
        # phase_diff = -k * d * sin(theta) * cos(phi)  (for x-axis element shift)
        # phase_diff = -k * d * sin(theta) * sin(phi)  (for y-axis element shift)
        kd = self.array.wavenumber * self.array.spacing
        
        # Avoid division by zero
        if kd == 0:
            return 0.0, 0.0

        # Solve for direction cosines
        cos_dir_x = -grad_x / kd
        cos_dir_y = -grad_y / kd

        # Check bounds for sin(theta)
        sin_theta_sq = cos_dir_x**2 + cos_dir_y**2
        sin_theta = np.sqrt(np.clip(sin_theta_sq, 0.0, 1.0))
        
        theta = np.arcsin(sin_theta)
        phi = np.arctan2(cos_dir_y, cos_dir_x) % (2.0 * np.pi)

        return float(theta), float(phi)

