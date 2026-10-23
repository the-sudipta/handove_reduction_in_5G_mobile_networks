"""Sanity checks for the corrected simulator and the baseline policies."""

import numpy as np

from handover.channel import Channel, path_loss_db
from handover.config import Config, scale_config
from handover.policies import A3Policy, Observation, RewardGreedyPolicy
from handover.policies.rewards import paper_reward
from handover.radio import L3Filter
from handover.simulator import Simulator, make_layout


def observation(rsrp, target_rsrp):
    n = len(rsrp)
    zeros = np.zeros(n)
    return Observation(step=0, rsrp=np.array(rsrp), cinr=zeros, rsrq=zeros, prev_cinr=zeros,
                       distance=zeros, speed=zeros, serving=np.zeros(n, int),
                       target=np.ones(n, int), target_rsrp=np.array(target_rsrp),
                       candidate=np.array(target_rsrp) > np.array(rsrp), rsrp_all=None)


def test_a3_fires_only_after_time_to_trigger():
    cfg = Config()                                   # 3 dB offset, 160 ms = 4 steps
    a3 = A3Policy(cfg)
    a3.reset(2)
    obs = observation(rsrp=[-100, -100], target_rsrp=[-96, -98])   # +4 dB and +2 dB
    fired = [a3.decide(obs) for _ in range(4)]
    assert [f[0] for f in fired] == [False, False, False, True]
    assert not any(f[1] for f in fired)             # +2 dB never satisfies the offset


def test_paper_reward_matches_eq4():
    rsrp = np.array([-100, -85, -100, -92, -85])
    handover = np.array([True, True, False, False, False])
    assert paper_reward(rsrp, handover).tolist() == [4, -3, -2, 2, 2]


def test_reward_greedy_policy_is_the_eq4_threshold():
    obs = observation(rsrp=[-95, -85], target_rsrp=[-90, -80])
    assert RewardGreedyPolicy().decide(obs).tolist() == [True, False]


def test_l3_filter_converges_with_k4_coefficient():
    f = L3Filter(k=4)
    f.update(np.array([0.0]))
    assert np.isclose(f.update(np.array([10.0]))[0], 5.0)   # a = 1/2^(4/4) = 0.5


def test_path_loss_grows_with_distance_and_nlos_is_worse():
    cfg = Config()
    d = np.array([50.0, 100.0, 300.0])
    los = path_loss_db(d, np.ones(3, bool), cfg)
    nlos = path_loss_db(d, np.zeros(3, bool), cfg)
    assert np.all(np.diff(los) > 0) and np.all(nlos >= los)


def test_channel_is_identical_for_every_policy():
    cfg = scale_config(1, duration_s=2.0)
    sim = Simulator(cfg, make_layout(cfg, 1))
    traces = []
    for policy in (A3Policy(cfg), RewardGreedyPolicy()):
        rsrp = []
        sim.run(policy, 20, seed=7, recorder=lambda obs, sinr: rsrp.append(obs.rsrp_all.copy()))
        traces.append(np.stack(rsrp))
    assert np.allclose(traces[0], traces[1])
