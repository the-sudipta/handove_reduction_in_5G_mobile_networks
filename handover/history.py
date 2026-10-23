"""Historical measurement logs used to fit the ensemble, the LSTM and scalers.

A `History` is passed to `Simulator.run(..., recorder=history)`; it stores
every step of one episode. The helpers below turn the logs into the
training sets each model needs.
"""

import numpy as np
import pandas as pd

STATE_COLUMNS = ["rsrp", "cinr", "rsrq", "distance", "speed", "prev_cinr"]


class History:
    """Recorder for one episode: per-step observations plus all-cell SINR."""

    def __init__(self):
        self.rows = {name: [] for name in STATE_COLUMNS + ["serving", "target", "candidate"]}
        self.rsrp_all, self.sinr_all = [], []

    def __call__(self, obs, sinr_all):
        for name in self.rows:
            self.rows[name].append(np.asarray(getattr(obs, name)).copy())
        self.rsrp_all.append(obs.rsrp_all.astype(np.float32))
        self.sinr_all.append(sinr_all.astype(np.float32))

    def array(self, name):
        """(steps, users) array of one recorded field."""
        return np.stack(self.rows[name])


def handover_events(histories, horizon_steps, gain_db=2.0) -> pd.DataFrame:
    """Candidate moments labelled by whether switching would have paid off.

    Paper's label: a handover is beneficial when it brings a CINR improvement
    above 2 dB. Here this is measured counterfactually: the SINR the user
    would get from the target cell `horizon_steps` later versus from the
    serving cell at the same moment.
    """
    frames = []
    for h in histories:
        sinr = np.stack(h.sinr_all)
        steps, users = np.nonzero(h.array("candidate")[:-horizon_steps])
        serving = h.array("serving")[steps, users]
        target = h.array("target")[steps, users]
        future = steps + horizon_steps
        gain = sinr[future, users, target] - sinr[future, users, serving]
        frame = pd.DataFrame({c: h.array(c)[steps, users] for c in STATE_COLUMNS})
        frame["serving"], frame["label"] = serving, (gain > gain_db).astype(int)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def cell_statistics(histories, num_cells) -> np.ndarray:
    """Per cell: mean RSRP and SINR of its users and its mean load (paper's K-Means inputs)."""
    rsrp, cinr, load = (np.zeros(num_cells) for _ in range(3))
    counts, steps = np.zeros(num_cells), 0
    for h in histories:
        serving = h.array("serving").ravel()
        np.add.at(rsrp, serving, h.array("rsrp").ravel())
        np.add.at(cinr, serving, h.array("cinr").ravel())
        np.add.at(counts, serving, 1)
        steps += len(h.rows["serving"])
    seen = np.maximum(counts, 1)
    return np.column_stack([rsrp / seen, cinr / seen, counts / max(steps, 1)])


def rsrp_sequences(histories, window, stride, horizon_steps, max_samples, rng):
    """LSTM samples at candidate moments.

    Input: the last `window` RSRP samples (every `stride` steps) of the serving
    cell and of the candidate, relative to the current serving RSRP.
    Target: candidate minus serving RSRP `horizon_steps` ahead.
    """
    span = (window - 1) * stride
    inputs, targets = [], []
    for h in histories:
        rsrp = np.stack(h.rsrp_all)
        candidate = h.array("candidate")
        candidate[:span] = False
        candidate[len(candidate) - horizon_steps:] = False
        steps, users = np.nonzero(candidate)
        serving = h.array("serving")[steps, users]
        target = h.array("target")[steps, users]
        offsets = np.arange(-span, 1, stride)
        idx = steps[:, None] + offsets[None, :]
        seq = np.stack([rsrp[idx, users[:, None], serving[:, None]],
                        rsrp[idx, users[:, None], target[:, None]]], axis=2)
        inputs.append(seq - seq[:, -1:, :1])
        future = steps + horizon_steps
        targets.append(rsrp[future, users, target] - rsrp[future, users, serving])
    x, y = np.concatenate(inputs), np.concatenate(targets)
    keep = rng.permutation(len(x))[:max_samples]
    return x[keep].astype(np.float32), y[keep].astype(np.float32)
