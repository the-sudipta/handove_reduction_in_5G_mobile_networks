"""Reference policy: the action that maximises the paper's reward, Eq. (4), at every step.

With a candidate present, handing over pays +4 when RSRP < -90 dBm, while
staying pays at most +2; above -90 dBm staying (+2) beats handing over (-3).
So the reward is maximised by "hand over iff RSRP < -90 dBm", whatever the
other state features say. Comparing this rule with the DQN shows what the
learned agent actually adds.
"""

import numpy as np

from .base import Observation, Policy


class RewardGreedyPolicy(Policy):
    name = "Eq4-greedy"

    def __init__(self, rsrp_threshold_dbm=-90.0):
        self.threshold = rsrp_threshold_dbm

    def decide(self, obs: Observation) -> np.ndarray:
        return obs.candidate & (obs.rsrp < self.threshold)
