"""Time-stepped, per-user handover simulator.

Each step (40 ms):
  1. users move, the channel is evaluated for every user-cell link;
  2. the UE's L3-filtered measurements are handed to the policy;
  3. requested handovers are executed: they fail (HOF) if the target's
     SINR is below Qout, otherwise the user switches cell and suffers a
     short interruption; a quick return to the previous cell is a ping-pong;
  4. radio link failure (RLF) is declared when the serving SINR stays below
     Qout for T310; failed users are in outage until they re-establish on
     the strongest cell;
  5. throughput is Shannon capacity with the cell bandwidth shared equally
     among the users it serves.

The channel and mobility depend only on the seed, never on the policy, so
every policy is evaluated on exactly the same radio conditions.
"""

import numpy as np

from .channel import Channel, place_cells
from .config import Config
from .mobility import Users
from .policies.base import Observation, Policy
from .radio import L3Filter, noise_per_re_dbm, rsrq_db, shannon_mbps, sinr_db


def make_layout(cfg: Config, layout_seed: int) -> np.ndarray:
    """Cell positions; fixed per scenario so training and testing share a network."""
    return place_cells(cfg, np.random.default_rng(layout_seed))


class Simulator:
    def __init__(self, cfg: Config, cells: np.ndarray):
        self.cfg = cfg
        self.cells = cells
        self.noise_dbm = noise_per_re_dbm(cfg)

    def run(self, policy: Policy, num_users: int, seed: int, recorder=None) -> dict:
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        users = Users(cfg, num_users, rng)
        channel = Channel(cfg, self.cells, num_users, rng)
        rsrp_filter, sinr_filter = L3Filter(cfg.l3_filter_k), L3Filter(cfg.l3_filter_k)
        everyone = np.arange(num_users)

        # Per-user connection state.
        serving = np.zeros(num_users, dtype=int)
        previous_cell = np.full(num_users, -1)
        last_ho_step = np.full(num_users, -10**9)
        outage_left = np.zeros(num_users, dtype=int)        # steps until re-established
        interruption_left = np.zeros(num_users, dtype=int)  # steps of HO interruption
        below_qout = np.zeros(num_users, dtype=int)         # T310 timer (in steps)
        prev_cinr = None

        steps_of = dict(ho_steps=cfg.seconds_to_steps(cfg.ho_interruption_s),
                        reest_steps=cfg.seconds_to_steps(cfg.reestablish_s),
                        t310_steps=cfg.seconds_to_steps(cfg.t310_s),
                        pp_steps=cfg.seconds_to_steps(cfg.ping_pong_s))
        count = dict(handovers=0, attempts=0, hof=0, rlf=0, ping_pong=0, outage_steps=0)
        throughput_sum = 0.0

        policy.reset(num_users)
        for step in range(cfg.steps):
            moved = users.move()
            rsrp_now, distance = channel.step(users.xy, moved)
            sinr_now = sinr_db(rsrp_now, self.noise_dbm)
            rsrp = rsrp_filter.update(rsrp_now)
            sinr = sinr_filter.update(sinr_now)

            if step == 0:
                serving = rsrp.argmax(axis=1)              # initial attachment
            connected = outage_left == 0

            # --- Observation for the policy ---
            neighbours = rsrp.copy()
            neighbours[everyone, serving] = -np.inf
            target = neighbours.argmax(axis=1)
            cinr = sinr[everyone, serving]
            obs = Observation(
                step=step,
                rsrp=rsrp[everyone, serving],
                cinr=cinr,
                rsrq=rsrq_db(rsrp, self.noise_dbm)[everyone, serving],
                prev_cinr=cinr if prev_cinr is None else prev_cinr,
                distance=distance[everyone, serving],
                speed=users.speed,
                serving=serving.copy(),
                target=target,
                target_rsrp=rsrp[everyone, target],
                candidate=connected & (rsrp[everyone, target] > rsrp[everyone, serving]),
                rsrp_all=rsrp,
            )
            if recorder is not None:
                recorder(obs, sinr)
            prev_cinr = cinr

            # --- Execute requested handovers ---
            wants_ho = policy.decide(obs) & obs.candidate
            for u in np.flatnonzero(wants_ho):
                count["attempts"] += 1
                if sinr_now[u, target[u]] < cfg.q_out_db:       # handover failure
                    count["hof"] += 1
                    outage_left[u] = steps_of["reest_steps"]
                    continue
                count["handovers"] += 1
                if target[u] == previous_cell[u] and step - last_ho_step[u] <= steps_of["pp_steps"]:
                    count["ping_pong"] += 1
                previous_cell[u], serving[u] = serving[u], target[u]
                last_ho_step[u] = step
                interruption_left[u] = steps_of["ho_steps"]
                below_qout[u] = 0

            # --- Radio link monitoring (RLF after T310 below Qout) ---
            serving_sinr = sinr_now[everyone, serving]
            monitored = (outage_left == 0) & ~wants_ho
            below_qout = np.where(monitored & (serving_sinr < cfg.q_out_db), below_qout + 1,
                                  np.where(serving_sinr > cfg.q_in_db, 0, below_qout))
            rlf = monitored & (below_qout >= steps_of["t310_steps"])
            count["rlf"] += int(rlf.sum())
            outage_left[rlf] = steps_of["reest_steps"]
            below_qout[rlf] = 0

            # --- Throughput: Shannon, bandwidth shared by users of the same cell ---
            active = (outage_left == 0) & (interruption_left == 0)
            load = np.bincount(serving[active], minlength=len(self.cells))
            share_hz = cfg.bandwidth_hz / np.maximum(load[serving], 1)
            throughput_sum += np.where(active, shannon_mbps(serving_sinr, share_hz, cfg), 0).sum()
            count["outage_steps"] += int((~active).sum())

            # --- Advance timers; re-establish users whose outage ended ---
            interruption_left = np.maximum(interruption_left - 1, 0)
            ending = outage_left == 1
            serving[ending] = rsrp[ending].argmax(axis=1)
            previous_cell[ending] = -1
            outage_left = np.maximum(outage_left - 1, 0)

        policy.end_episode()
        return self._metrics(count, throughput_sum, num_users)

    def _metrics(self, count: dict, throughput_sum: float, num_users: int) -> dict:
        cfg = self.cfg
        user_minutes = num_users * cfg.duration_s / 60
        return {
            "handovers": count["handovers"],
            "ho_per_user_min": count["handovers"] / user_minutes,
            "hof": count["hof"],
            "rlf": count["rlf"],
            "failure_ratio": count["hof"] / max(count["attempts"], 1),
            "rlf_per_user_min": count["rlf"] / user_minutes,
            "ping_pong_ratio": count["ping_pong"] / max(count["handovers"], 1),
            "interruption_ms_per_user_min": count["outage_steps"] * cfg.dt_s * 1000 / user_minutes,
            "throughput_mbps": throughput_sum / (num_users * cfg.steps),
        }
