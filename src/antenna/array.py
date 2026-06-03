"""Uniform Planar Array (UPA) antenna geometry and steering vector computations."""

from __future__ import annotations

import numpy as np

SPEED_OF_LIGHT = 299792458.0  # m/s


class UniformPlanarArray:
    """Represents a Uniform Planar Array (UPA) centered at the origin in the xy-plane.

    The array consists of n_rows along the x-axis and n_cols along the y-axis.
    The elements are spaced uniformly. Boresight of the array is along the positive z-axis.
    """

    def __init__(
        self,
        n_rows: int = 8,
        n_cols: int = 8,
        frequency_hz: float = 10.0e9,
        element_spacing_wavelengths: float = 0.5,
    ):
        """Initializes the Uniform Planar Array.

        Args:
            n_rows: Number of antenna elements along the x-axis.
            n_cols: Number of antenna elements along the y-axis.
            frequency_hz: Operating frequency of the radar system in Hz.
            element_spacing_wavelengths: Spacing between adjacent elements in wavelengths.
        """
        self._n_rows = n_rows
        self._n_cols = n_cols
        self._frequency_hz = frequency_hz
        self._element_spacing_wavelengths = element_spacing_wavelengths

        # Compute physics properties
        self._wavelength = SPEED_OF_LIGHT / self._frequency_hz
        self._wavenumber = 2.0 * np.pi / self._wavelength
        self._spacing = self._element_spacing_wavelengths * self._wavelength

        # Initialize element coordinates
        self._element_positions = self._generate_element_positions()

    def _generate_element_positions(self) -> np.ndarray:
        """Generates the 3D coordinates of all elements, centered at the origin.

        Returns:
            An (N, 3) numpy array representing the (x, y, z) coordinates.
        """
        # Centers the grid around (0, 0, 0)
        x_coords = (np.arange(self._n_rows) - (self._n_rows - 1) / 2.0) * self._spacing
        y_coords = (np.arange(self._n_cols) - (self._n_cols - 1) / 2.0) * self._spacing

        # Create meshgrid
        xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")

        # Reshape to (N, 3) where columns are [x, y, z] and z=0
        positions = np.zeros((self.n_elements, 3))
        positions[:, 0] = xx.flatten()
        positions[:, 1] = yy.flatten()
        positions[:, 2] = 0.0

        return positions

    @property
    def n_elements(self) -> int:
        """Total number of elements in the planar array."""
        return self._n_rows * self._n_cols

    @property
    def n_rows(self) -> int:
        """Number of elements along the x-axis."""
        return self._n_rows

    @property
    def n_cols(self) -> int:
        """Number of elements along the y-axis."""
        return self._n_cols

    @property
    def frequency_hz(self) -> float:
        """Radar operating frequency in Hz."""
        return self._frequency_hz

    @property
    def wavelength(self) -> float:
        """Radar operating wavelength in meters."""
        return self._wavelength

    @property
    def wavenumber(self) -> float:
        """Radar propagation wave number (k = 2π / λ)."""
        return self._wavenumber

    @property
    def spacing(self) -> float:
        """Physical spacing between elements in meters."""
        return self._spacing

    @property
    def element_positions(self) -> np.ndarray:
        """The (N, 3) array of element positions in meters."""
        return self._element_positions

    def get_steering_vector(self, theta: float, phi: float) -> np.ndarray:
        """Computes the complex steering vector for a given elevation and azimuth direction.

        Args:
            theta: Polar angle (elevation) from the z-axis (boresight) in radians [0, pi].
            phi: Azimuth angle in the xy-plane in radians [0, 2pi].

        Returns:
            A complex numpy array of shape (N,) representing the phase offsets at each element.
        """
        # Direction unit vector pointing from origin toward target (theta, phi)
        # Boresight is +z, theta is angle from +z axis
        u_x = np.sin(theta) * np.cos(phi)
        u_y = np.sin(theta) * np.sin(phi)
        u_z = np.cos(theta)

        # Compute k · r_n for all elements
        # element_positions is (N, 3)
        phases = self._wavenumber * (
            self._element_positions[:, 0] * u_x
            + self._element_positions[:, 1] * u_y
            + self._element_positions[:, 2] * u_z
        )

        return np.exp(1j * phases)

    def get_steering_vector_batch(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Computes complex steering vectors for a batch of directions.

        Args:
            theta: Array of polar angles in radians, shape (M,).
            phi: Array of azimuth angles in radians, shape (M,).

        Returns:
            A complex numpy array of shape (M, N) where M is the number of directions
            and N is the number of antenna elements.
        """
        theta = np.asarray(theta)
        phi = np.asarray(phi)

        assert theta.shape == phi.shape, "theta and phi arrays must have the same shape"

        # Direction unit vectors, shape (M, 3)
        u_x = np.sin(theta) * np.cos(phi)
        u_y = np.sin(theta) * np.sin(phi)
        u_z = np.cos(theta)

        u = np.stack([u_x, u_y, u_z], axis=-1)  # (M, 3)

        # Matrix multiplication: (M, 3) x (3, N) -> (M, N)
        phases = self._wavenumber * np.matmul(u, self._element_positions.T)

        return np.exp(1j * phases)

    @property
    def half_power_beamwidth(self) -> float:
        """Approximates the array's half-power beamwidth (HPBW) in radians.

        Using the uniform aperture approximation: HPBW ≈ 0.886 * λ / (N * d)
        where N is the maximum dimension size and d is the element spacing.
        """
        N = max(self._n_rows, self._n_cols)
        return 0.886 * self._wavelength / (N * self._spacing)

