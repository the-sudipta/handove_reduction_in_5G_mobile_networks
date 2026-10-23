"""Link-level quantities derived from RSRP: SINR, RSRQ, throughput, L3 filtering."""

import numpy as np

from .config import Config


def noise_per_re_dbm(cfg: Config) -> float:
    subcarrier_hz = cfg.bandwidth_hz / (12 * cfg.num_prb)
    return -174 + 10 * np.log10(subcarrier_hz) + cfg.noise_figure_db


def to_linear(db):
    return 10 ** (np.asarray(db) / 10)


def sinr_db(rsrp_dbm: np.ndarray, noise_dbm: float) -> np.ndarray:
    """SINR of every cell if it were serving; all other cells fully loaded."""
    power = to_linear(rsrp_dbm)
    total = power.sum(axis=1, keepdims=True)
    return 10 * np.log10(power / (total - power + to_linear(noise_dbm)))


def rsrq_db(rsrp_dbm: np.ndarray, noise_dbm: float) -> np.ndarray:
    """RSRQ = N * RSRP / RSSI, with RSSI measured over N PRBs (12 REs each)."""
    total = to_linear(rsrp_dbm).sum(axis=1, keepdims=True) + to_linear(noise_dbm)
    return rsrp_dbm - 10 * np.log10(12 * total)


def shannon_mbps(sinr: np.ndarray, bandwidth_hz: np.ndarray, cfg: Config) -> np.ndarray:
    capped = np.minimum(sinr, cfg.max_sinr_db)
    return bandwidth_hz * np.log2(1 + to_linear(capped)) / 1e6


class L3Filter:
    """TS 38.331 layer-3 filter: F_n = (1 - a) F_{n-1} + a M_n, a = 1 / 2^(k/4)."""

    def __init__(self, k: int):
        self.a = 1 / 2 ** (k / 4)
        self.value = None

    def update(self, measurement: np.ndarray) -> np.ndarray:
        if self.value is None:
            self.value = measurement.copy()
        else:
            self.value = (1 - self.a) * self.value + self.a * measurement
        return self.value
