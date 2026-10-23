"""What a handover policy sees and the interface every policy implements."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Observation:
    """Per-user measurements at one step (all arrays have length num_users).

    Values are L3-filtered, as a UE would report them. The candidate is the
    strongest neighbour; a policy only answers "hand over to it now or not".
    """
    step: int
    rsrp: np.ndarray            # serving-cell RSRP (dBm)
    cinr: np.ndarray            # serving-cell SINR (dB)
    rsrq: np.ndarray            # serving-cell RSRQ (dB)
    prev_cinr: np.ndarray       # serving-cell SINR at the previous step (dB)
    distance: np.ndarray        # distance to the serving cell (m)
    speed: np.ndarray           # user speed (m/s)
    serving: np.ndarray         # serving-cell index
    target: np.ndarray          # strongest-neighbour index
    target_rsrp: np.ndarray     # strongest-neighbour RSRP (dBm)
    candidate: np.ndarray       # bool: connected and the neighbour is stronger
    rsrp_all: np.ndarray        # (num_users, num_cells) RSRP of every cell


class Policy:
    """Base class. Subclasses override `decide` and, if stateful, `reset`."""

    name = "policy"

    def reset(self, num_users: int) -> None:
        """Called at the start of every episode."""

    def decide(self, obs: Observation) -> np.ndarray:
        """Return a bool array: True = hand over to `obs.target` now."""
        raise NotImplementedError

    def end_episode(self) -> None:
        """Called at the end of every episode (learning policies flush here)."""
