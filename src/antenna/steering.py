"""Coordinate conversions and optimal beamforming weight calculators."""

from __future__ import annotations

import numpy as np
from .array import UniformPlanarArray


def cart_to_spherical(x: float | np.ndarray, y: float | np.ndarray, z: float | np.ndarray) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    """Converts 3D Cartesian coordinates to spherical coordinates.

    Uses the physics convention where:
    - theta is the polar angle (elevation) from the +z axis (boresight), in [0, pi]
    - phi is the azimuth angle in the xy-plane from the +x axis, in [0, 2pi]
    - r is the range (distance from origin)

    Args:
        x: Cartesian x coordinate(s) in meters.
        y: Cartesian y coordinate(s) in meters.
        z: Cartesian z coordinate(s) in meters.

    Returns:
        Tuple (r, theta, phi) where:
            r: Radial distance.
            theta: Polar angle in radians.
            phi: Azimuth angle in radians.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Avoid division by zero at the origin
    r_safe = np.where(r == 0, 1.0, r)
    
    theta = np.arccos(np.clip(z / r_safe, -1.0, 1.0))
    # Map theta to 0 if r was 0
    theta = np.where(r == 0, 0.0, theta)
    
    phi = np.arctan2(y, x) % (2.0 * np.pi)

    # Return scalars if inputs were scalars
    if x.ndim == 0:
        return float(r), float(theta), float(phi)
    return r, theta, phi


def spherical_to_cart(r: float | np.ndarray, theta: float | np.ndarray, phi: float | np.ndarray) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    """Converts spherical coordinates to 3D Cartesian coordinates.

    Args:
        r: Radial distance in meters.
        theta: Polar angle in radians [0, pi].
        phi: Azimuth angle in radians [0, 2pi].

    Returns:
        Tuple (x, y, z) representing Cartesian coordinates in meters.
    """
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    if r.ndim == 0:
        return float(x), float(y), float(z)
    return x, y, z


def compute_optimal_phases(array: UniformPlanarArray, target_theta: float, target_phi: float) -> np.ndarray:
    """Computes the optimal phase-only beamforming weights using conjugate phasing.

    Conjugate phasing aligns the phase of each element's received signal to add
    fully constructively in the target direction.

    Args:
        array: A UniformPlanarArray instance.
        target_theta: Target polar elevation in radians.
        target_phi: Target azimuth in radians.

    Returns:
        Array of optimal phase values in radians of shape (N,).
    """
    # conjugate phasing aligns the phases by multiplying by the conjugate of the steering vector.
    # w_optimal = 1/sqrt(N) * conj(steering_vector)
    # phase of w_optimal is angle(conj(steering_vector)) = -angle(steering_vector)
    steering_vec = array.get_steering_vector(target_theta, target_phi)
    # Use -np.angle to extract the conjugate phase
    optimal_phases = -np.angle(steering_vec)
    # Wrap to [-pi, pi]
    return (optimal_phases + np.pi) % (2.0 * np.pi) - np.pi


def angular_distance(
    theta1: float | np.ndarray,
    phi1: float | np.ndarray,
    theta2: float | np.ndarray,
    phi2: float | np.ndarray,
) -> float | np.ndarray:
    """Computes the great-circle angular distance between two spherical directions.

    Args:
        theta1: Elevation angle of direction 1 in radians.
        phi1: Azimuth angle of direction 1 in radians.
        theta2: Elevation angle of direction 2 in radians.
        phi2: Azimuth angle of direction 2 in radians.

    Returns:
        Angular distance in radians.
    """
    theta1 = np.asarray(theta1, dtype=float)
    phi1 = np.asarray(phi1, dtype=float)
    theta2 = np.asarray(theta2, dtype=float)
    phi2 = np.asarray(phi2, dtype=float)

    # Unit vectors for direction 1
    u1_x = np.sin(theta1) * np.cos(phi1)
    u1_y = np.sin(theta1) * np.sin(phi1)
    u1_z = np.cos(theta1)

    # Unit vectors for direction 2
    u2_x = np.sin(theta2) * np.cos(phi2)
    u2_y = np.sin(theta2) * np.sin(phi2)
    u2_z = np.cos(theta2)

    # Dot product of the unit vectors
    dot_product = u1_x * u2_x + u1_y * u2_y + u1_z * u2_z

    # Clip to avoid float precision issues outside [-1.0, 1.0]
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    dist = np.arccos(dot_product)
    
    if theta1.ndim == 0 and theta2.ndim == 0:
        return float(dist)
    return dist
