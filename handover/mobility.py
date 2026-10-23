"""Per-user random-direction mobility with gradual turns (paper: 0.5-5 m/s)."""

import numpy as np

from .config import Config


class Users:
    """Positions, speeds and headings of all users, moved one step at a time."""

    def __init__(self, cfg: Config, num_users: int, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.xy = rng.uniform(0, cfg.area_m, size=(num_users, 2))
        self.speed = rng.uniform(cfg.speed_min, cfg.speed_max, size=num_users)
        self.heading = rng.uniform(0, 2 * np.pi, size=num_users)

    def move(self) -> np.ndarray:
        """Advance every user by dt; reflect at the area border. Returns metres moved."""
        cfg = self.cfg
        self.heading += self.rng.normal(0, cfg.turn_std_rad, size=self.heading.shape)
        step = self.speed * cfg.dt_s
        self.xy[:, 0] += step * np.cos(self.heading)
        self.xy[:, 1] += step * np.sin(self.heading)

        # Reflect users that left the area.
        for axis in (0, 1):
            low, high = self.xy[:, axis] < 0, self.xy[:, axis] > cfg.area_m
            self.xy[low, axis] *= -1
            self.xy[high, axis] = 2 * cfg.area_m - self.xy[high, axis]
            if axis == 0:
                self.heading[low | high] = np.pi - self.heading[low | high]
            else:
                self.heading[low | high] = -self.heading[low | high]
        return step
