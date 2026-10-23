"""All simulation parameters in one place.

Values follow the mother paper where it states them (UMi channel, 4 dB /
50 m shadowing, A3 with 3 dB / 160 ms, 20 MHz, scales 1x-3x) and 3GPP
TR 38.901 / TS 38.331 / TS 38.133 where it does not.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    # --- Network layout (paper: Scale1x = 20 cells in 1000 x 1000 m) ---
    area_m: float = 1000.0
    num_cells: int = 20
    bs_height_m: float = 10.0          # 38.901 UMi
    ue_height_m: float = 1.5

    # --- Radio ---
    carrier_ghz: float = 3.5           # n78
    bandwidth_hz: float = 20e6
    num_prb: int = 51                  # 20 MHz @ 30 kHz SCS
    tx_power_dbm: float = 30.0
    noise_figure_db: float = 9.0
    max_sinr_db: float = 30.0          # cap of the Shannon throughput model

    # --- Channel ---
    shadow_std_db: float = 4.0         # paper
    shadow_corr_m: float = 50.0        # paper
    fading_samples: int = 8            # Rayleigh samples averaged by L1 measurement

    # --- Mobility (paper: 0.5-5 m/s, gradually changing direction) ---
    speed_min: float = 0.5
    speed_max: float = 5.0
    turn_std_rad: float = 0.05         # per step heading change
    dt_s: float = 0.04                 # measurement / decision period (40 ms)
    duration_s: float = 60.0

    # --- Measurement filtering (TS 38.331 L3 filter, k = 4) ---
    l3_filter_k: int = 4

    # --- Handover / radio link failure (TS 38.133, TR 36.839) ---
    a3_offset_db: float = 3.0          # paper
    ttt_s: float = 0.16                # paper
    q_out_db: float = -8.0
    q_in_db: float = -6.0
    t310_s: float = 1.0
    ho_interruption_s: float = 0.04
    reestablish_s: float = 0.5
    ping_pong_s: float = 1.0

    @property
    def steps(self) -> int:
        return int(round(self.duration_s / self.dt_s))

    def seconds_to_steps(self, seconds: float) -> int:
        return max(1, int(round(seconds / self.dt_s)))


def scale_config(scale: int, **overrides) -> Config:
    """Paper scales: area x scale per side, cells x scale (Scale1x/2x/3x)."""
    base = Config()
    return replace(base, area_m=base.area_m * scale,
                   num_cells=base.num_cells * scale, **overrides)
