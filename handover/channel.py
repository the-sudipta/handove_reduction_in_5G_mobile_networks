"""3GPP TR 38.901 UMi Street-Canyon channel, evaluated per user and per cell.

Every user-cell link has its own
  * LOS/NLOS state: a fixed per-link uniform draw compared with the 38.901
    LOS probability at the current distance (LOS changes only as the user
    moves, never by re-drawing, so it cannot chatter),
  * log-normal shadowing, correlated along the user's path (Gudmundson),
  * Rayleigh fast fading, averaged over a few samples by the L1 measurement.
"""

import numpy as np

from .config import Config


def place_cells(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random small-cell positions, shape (num_cells, 2)."""
    return rng.uniform(0, cfg.area_m, size=(cfg.num_cells, 2))


def los_probability(d2d: np.ndarray) -> np.ndarray:
    """38.901 Table 7.4.2-1, UMi Street Canyon."""
    d = np.maximum(d2d, 1e-3)
    p = 18 / d + np.exp(-d / 36) * (1 - 18 / d)
    return np.where(d2d <= 18, 1.0, p)


def path_loss_db(d2d: np.ndarray, los: np.ndarray, cfg: Config) -> np.ndarray:
    """38.901 Table 7.4.1-1, UMi Street Canyon (LOS with breakpoint, NLOS)."""
    fc, h_bs, h_ut = cfg.carrier_ghz, cfg.bs_height_m, cfg.ue_height_m
    d3d = np.sqrt(np.maximum(d2d, 10.0) ** 2 + (h_bs - h_ut) ** 2)
    d_bp = 4 * (h_bs - 1) * (h_ut - 1) * fc * 1e9 / 3e8

    pl_los_near = 32.4 + 21 * np.log10(d3d) + 20 * np.log10(fc)
    pl_los_far = (32.4 + 40 * np.log10(d3d) + 20 * np.log10(fc)
                  - 9.5 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))
    pl_los = np.where(d2d <= d_bp, pl_los_near, pl_los_far)

    pl_nlos = 22.4 + 35.3 * np.log10(d3d) + 21.3 * np.log10(fc) - 0.3 * (h_ut - 1.5)
    return np.where(los, pl_los, np.maximum(pl_los, pl_nlos))


class Channel:
    """Produces the instantaneous per-RE RSRP of every cell at every user."""

    def __init__(self, cfg: Config, cells: np.ndarray, num_users: int,
                 rng: np.random.Generator):
        self.cfg = cfg
        self.cells = cells
        self.rng = rng
        shape = (num_users, len(cells))
        self.shadow = rng.normal(size=shape)      # unit-variance, scaled on use
        self.los_draw = rng.uniform(size=shape)   # link is LOS while draw < P_LOS(d)
        # RSRP is measured per resource element: spread Tx power over 12 * PRBs.
        self.tx_per_re_dbm = cfg.tx_power_dbm - 10 * np.log10(12 * cfg.num_prb)

    def _evolve(self, field: np.ndarray, moved_m: np.ndarray, corr_m: float) -> np.ndarray:
        """First-order (Gudmundson) update of a spatially correlated Gaussian field."""
        rho = np.exp(-moved_m / corr_m)[:, None]
        return rho * field + np.sqrt(1 - rho ** 2) * self.rng.normal(size=field.shape)

    def step(self, user_xy: np.ndarray, moved_m: np.ndarray):
        """Return (rsrp_dbm, distance_m), both of shape (num_users, num_cells)."""
        cfg = self.cfg
        self.shadow = self._evolve(self.shadow, moved_m, cfg.shadow_corr_m)

        d2d = np.linalg.norm(user_xy[:, None, :] - self.cells[None, :, :], axis=2)
        los = self.los_draw < los_probability(d2d)
        fading_db = 10 * np.log10(self.rng.gamma(cfg.fading_samples, 1 / cfg.fading_samples,
                                                 size=d2d.shape))
        rsrp = (self.tx_per_re_dbm - path_loss_db(d2d, los, cfg)
                - cfg.shadow_std_db * self.shadow + fading_db)
        return rsrp, d2d
