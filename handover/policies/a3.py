"""3GPP A3 event: neighbour better than serving by an offset for Time-To-Trigger."""

import numpy as np

from ..config import Config
from .base import Observation, Policy


class A3Policy(Policy):
    name = "A3"

    def __init__(self, cfg: Config):
        self.offset_db = cfg.a3_offset_db
        self.ttt_steps = cfg.seconds_to_steps(cfg.ttt_s)

    def reset(self, num_users: int) -> None:
        self.timer = np.zeros(num_users, dtype=int)    # steps the condition has held
        self.tracked = np.full(num_users, -1)          # neighbour being timed

    def decide(self, obs: Observation) -> np.ndarray:
        entering = obs.candidate & (obs.target_rsrp > obs.rsrp + self.offset_db)
        same_neighbour = obs.target == self.tracked
        self.timer = np.where(entering, np.where(same_neighbour, self.timer + 1, 1), 0)
        self.tracked = np.where(entering, obs.target, -1)

        fire = self.timer >= self.ttt_steps
        self.timer[fire] = 0
        return fire
