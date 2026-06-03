"""Target dynamics simulation using the Singer maneuvering model."""

from __future__ import annotations

import numpy as np
from ..antenna.steering import cart_to_spherical


class AerialTarget:
    """Simulates a 3D maneuvering aerial target using the Singer model.

    The Singer model describes target acceleration as a first-order Gauss-Markov process.
    The radar is located at the origin (0, 0, 0). Boresight is along the positive z-axis.
    """

    def __init__(
        self,
        max_speed: float = 300.0,
        max_acceleration: float = 50.0,
        acceleration_std: float = 30.0,
        correlation_time: float = 2.0,
        initial_range_min: float = 5000.0,
        initial_range_max: float = 15000.0,
        altitude_min: float = 1000.0,
        altitude_max: float = 10000.0,
        azimuth_range_deg: float = 120.0,
        elevation_range_deg: float = 60.0,
        dt: float = 0.01,
        rng_seed: int | None = None,
    ):
        """Initializes the target dynamics model.

        Args:
            max_speed: Maximum target speed in m/s.
            max_acceleration: Maximum target acceleration in m/s^2.
            acceleration_std: Maneuver acceleration standard deviation (sigma_a) in m/s^2.
            correlation_time: Maneuver correlation time constant (tau) in seconds.
            initial_range_min: Minimum initial range in meters.
            initial_range_max: Maximum initial range in meters.
            altitude_min: Minimum target altitude in meters.
            altitude_max: Maximum target altitude in meters.
            azimuth_range_deg: Coverage angle for azimuth initialization in degrees.
            elevation_range_deg: Coverage angle for elevation initialization in degrees.
            dt: Simulation time step in seconds (e.g., 0.01 for 100 Hz updates).
            rng_seed: Seed for random number generator.
        """
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.acceleration_std = acceleration_std
        self.correlation_time = correlation_time
        self.initial_range_min = initial_range_min
        self.initial_range_max = initial_range_max
        self.altitude_min = altitude_min
        self.altitude_max = altitude_max
        self.azimuth_range_deg = azimuth_range_deg
        self.elevation_range_deg = elevation_range_deg
        self.dt = dt

        self.rng = np.random.RandomState(rng_seed)

        # Singer transition matrix parameters
        self.rho = np.exp(-self.dt / self.correlation_time)
        self.tau = self.correlation_time

        # Exact Singer transition matrix per axis for state vector s = [pos, vel, acc]^T
        # s(k+1) = F * s(k) + w(k)
        self.F_axis = np.array(
            [
                [1.0, self.dt, (self.rho - 1.0 + self.dt / self.tau) * (self.tau**2)],
                [0.0, 1.0, (1.0 - self.rho) * self.tau],
                [0.0, 0.0, self.rho],
            ]
        )

        # State vector: [x, y, z, vx, vy, vz, ax, ay, az]
        self.state = np.zeros(9)
        self._prev_angular_pos = (0.0, 0.0)  # (theta, phi) from previous step
        
        # Track initial limits to enforce boundary conditions
        self.max_range = self.initial_range_max * 1.5
        
        self.reset()

    def reset(self) -> np.ndarray:
        """Resets the target to a random state within the surveillance volume.

        Returns:
            The initial 9D state vector.
        """
        # Attempt to sample a valid 3D position satisfying range, altitude, and angle limits
        max_attempts = 100
        for _ in range(max_attempts):
            # Sample range
            r = self.rng.uniform(self.initial_range_min, self.initial_range_max)
            
            # Sample polar angle theta (angle from boresight +z axis)
            # elevation_range_deg represents full cone width, so half-width from boresight
            max_theta = np.radians(self.elevation_range_deg / 2.0)
            theta = self.rng.uniform(0.0, max_theta)
            
            # Sample azimuth angle phi in xy-plane
            max_phi = np.radians(self.azimuth_range_deg / 2.0)
            phi = self.rng.uniform(-max_phi, max_phi)

            # Convert to Cartesian coordinates (boresight is +z, so x, y are off-axis offsets)
            # In our array convention, theta is polar angle from +z, phi is azimuth in xy-plane
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            # Verify altitude constraint
            if self.altitude_min <= z <= self.altitude_max:
                self.state[0:3] = [x, y, z]
                break
        else:
            # Fallback if sampling fails: place directly along boresight
            r = (self.initial_range_min + self.initial_range_max) / 2.0
            z = np.clip(r, self.altitude_min, self.altitude_max)
            self.state[0:3] = [0.0, 0.0, z]

        # Initial speed: 30% to 70% of max speed
        speed = self.rng.uniform(0.3 * self.max_speed, 0.7 * self.max_speed)
        
        # Initial velocity vector direction: random unit vector in the negative z-hemisphere
        # to ensure the target flies generally towards the radar initially
        v_theta = self.rng.uniform(np.pi / 2.0, np.pi)  # pointed downwards
        v_phi = self.rng.uniform(0.0, 2.0 * np.pi)
        
        vx = speed * np.sin(v_theta) * np.cos(v_phi)
        vy = speed * np.sin(v_theta) * np.sin(v_phi)
        vz = speed * np.cos(v_theta)
        
        self.state[3:6] = [vx, vy, vz]

        # Initial acceleration: 0
        self.state[6:9] = 0.0

        # Initialize angular positions
        _, theta_init, phi_init = cart_to_spherical(self.state[0], self.state[1], self.state[2])
        self._prev_angular_pos = (theta_init, phi_init)

        return self.state.copy()

    def step(self) -> np.ndarray:
        """Propagates the target state by one timestep using the Singer model.

        Enforces physical speed limits and spatial boundaries.

        Returns:
            The updated 9D state vector.
        """
        # Save current position to calculate angular rates later
        curr_theta, curr_phi = self.get_angular_position()
        self._prev_angular_pos = (curr_theta, curr_phi)

        # Propagate each axis independently
        for i in range(3):
            # Extract axis sub-state [pos, vel, acc]
            s_axis = self.state[[i, i + 3, i + 6]]
            
            # Deterministic transition
            s_next = np.matmul(self.F_axis, s_axis)
            
            # Stochastic acceleration process noise update
            # a(k+1) = rho * a(k) + sigma_a * sqrt(1 - rho^2) * N(0, 1)
            noise = self.rng.normal(0.0, 1.0)
            a_stochastic = self.acceleration_std * np.sqrt(1.0 - self.rho**2) * noise
            s_next[2] += a_stochastic
            
            # Update state vector
            self.state[[i, i + 3, i + 6]] = s_next

        # ── Clamp Dynamics ──────────────────────────────────────────────────
        # Clamp acceleration magnitude
        accel_mag = np.linalg.norm(self.state[6:9])
        if accel_mag > self.max_acceleration:
            self.state[6:9] = (self.state[6:9] / accel_mag) * self.max_acceleration

        # Clamp speed
        speed = np.linalg.norm(self.state[3:6])
        if speed > self.max_speed:
            self.state[3:6] = (self.state[3:6] / speed) * self.max_speed

        # ── Boundary Enforcements (Soft Reflection) ───────────────────────
        x, y, z = self.state[0:3]
        vx, vy, vz = self.state[3:6]
        ax, ay, az = self.state[6:9]

        # Max range limit (cylinder/hemisphere boundary)
        range_from_radar = np.linalg.norm(self.state[0:3])
        if range_from_radar > self.max_range:
            # Reflect position and velocity radially
            radial_dir = self.state[0:3] / range_from_radar
            self.state[0:3] = radial_dir * (2.0 * self.max_range - range_from_radar)
            
            # Reverse velocity along radial direction
            v_radial = np.dot(self.state[3:6], radial_dir)
            self.state[3:6] -= 2.0 * abs(v_radial) * radial_dir
            self.state[6:9] = 0.0  # Zero acceleration for stability on bounce

        # Altitude bounds
        if z < self.altitude_min:
            self.state[2] = 2.0 * self.altitude_min - z
            self.state[5] = abs(vz)
            self.state[8] = abs(az) * 0.5
        elif z > self.altitude_max:
            self.state[2] = 2.0 * self.altitude_max - z
            self.state[5] = -abs(vz)
            self.state[8] = -abs(az) * 0.5

        # Angular bounds relative to boresight (prevent target flying behind radar)
        max_theta = np.radians(self.elevation_range_deg)
        _, theta, _ = cart_to_spherical(self.state[0], self.state[1], self.state[2])
        if theta > max_theta:
            # Reflect target back towards boresight
            # Simple bounce: reverse the transverse velocities
            self.state[3] = -vx * 0.5
            self.state[4] = -vy * 0.5
            self.state[6:8] = 0.0

        return self.state.copy()

    def get_position(self) -> np.ndarray:
        """Returns the 3D position of the target [x, y, z] in meters."""
        return self.state[0:3].copy()

    def get_velocity(self) -> np.ndarray:
        """Returns the velocity vector [vx, vy, vz] in m/s."""
        return self.state[3:6].copy()

    def get_acceleration(self) -> np.ndarray:
        """Returns the acceleration vector [ax, ay, az] in m/s^2."""
        return self.state[6:9].copy()

    def get_speed(self) -> float:
        """Returns the scalar speed of the target in m/s."""
        return float(np.linalg.norm(self.state[3:6]))

    def get_angular_position(self, radar_position: np.ndarray | None = None) -> tuple[float, float]:
        """Computes target (theta, phi) angles as seen from the radar.

        Args:
            radar_position: 3D position of the radar. Defaults to origin.

        Returns:
            Tuple (theta, phi) in radians relative to boresight (+z).
        """
        pos = self.state[0:3]
        if radar_position is not None:
            pos = pos - radar_position
        _, theta, phi = cart_to_spherical(pos[0], pos[1], pos[2])
        return float(theta), float(phi)

    def get_angular_velocity(self, radar_position: np.ndarray | None = None) -> tuple[float, float]:
        """Computes the angular rates (d_theta/dt, d_phi/dt) of the target.

        Uses backward finite difference.

        Args:
            radar_position: 3D position of the radar. Defaults to origin.

        Returns:
            Tuple (d_theta/dt, d_phi/dt) in radians/second.
        """
        curr_theta, curr_phi = self.get_angular_position(radar_position)
        prev_theta, prev_phi = self._prev_angular_pos

        # Wrap phi difference to [-pi, pi]
        d_phi = (curr_phi - prev_phi + np.pi) % (2.0 * np.pi) - np.pi
        d_theta = curr_theta - prev_theta

        return float(d_theta / self.dt), float(d_phi / self.dt)

    def get_state_dict(self) -> dict:
        """Returns target state details as a dictionary."""
        theta, phi = self.get_angular_position()
        d_theta, d_phi = self.get_angular_velocity()
        return {
            "x": float(self.state[0]),
            "y": float(self.state[1]),
            "z": float(self.state[2]),
            "vx": float(self.state[3]),
            "vy": float(self.state[4]),
            "vz": float(self.state[5]),
            "speed": self.get_speed(),
            "ax": float(self.state[6]),
            "ay": float(self.state[7]),
            "az": float(self.state[8]),
            "theta": theta,
            "phi": phi,
            "d_theta": d_theta,
            "d_phi": d_phi,
        }
