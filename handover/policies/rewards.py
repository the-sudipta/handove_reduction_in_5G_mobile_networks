"""Reward of the mother paper, Eq. (4), shared by Q-learning and the DQN.

  handover and RSRP <  -90 dBm  -> +4   (necessary handover)
  handover and RSRP >= -90 dBm  -> -3   (unnecessary handover)
  no handover and RSRP > -95    -> +2   (good link kept)
  no handover and RSRP <= -95   -> -2   (missed handover)

The "-4 for a handover without a valid candidate" case cannot happen here:
policies are only asked to decide when a candidate neighbour exists.
"""

import numpy as np


def paper_reward(rsrp: np.ndarray, handover: np.ndarray) -> np.ndarray:
    on_handover = np.where(rsrp < -90, 4.0, -3.0)
    on_stay = np.where(rsrp > -95, 2.0, -2.0)
    return np.where(handover, on_handover, on_stay)
