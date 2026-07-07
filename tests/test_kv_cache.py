import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.kv_cache import EvictionSimulator


def test_none_policy_keeps_all():
    sim = EvictionSimulator(budget=2, policy="none")
    rate = None
    for t in range(6):
        row = [1.0 / (t + 1)] * (t + 1)
        rate = sim.step(row)
    assert abs(rate - 1.0) < 1e-9
    assert len(sim.resident) == 6


def test_fifo_evicts_oldest():
    sim = EvictionSimulator(budget=3, policy="fifo")
    for t in range(5):
        sim.step([1.0] * (t + 1))
    # after 5 inserts with budget 3, positions 0 and 1 are gone
    assert set(sim.resident.keys()) == {2, 3, 4}


def test_lru_keeps_attended_position():
    sim = EvictionSimulator(budget=2, policy="lru")
    # position 0 keeps getting the most attention, so LRU must retain it
    for t in range(5):
        row = [0.0] * (t + 1)
        row[0] = 1.0  # all mass on position 0
        sim.step(row)
    assert 0 in sim.resident
    assert len(sim.resident) == 2


def test_lru_beats_fifo_with_sink():
    from src.bench import sink_stress_test

    r = sink_stress_test(gen_tokens=200, budget=16, sink_mass=0.3)
    assert r["lru_hit"] > r["fifo_hit"] + 0.1


def test_hit_rate_within_bounds():
    sim = EvictionSimulator(budget=4, policy="lru")
    for t in range(20):
        row = [1.0 / (t + 1)] * (t + 1)
        r = sim.step(row)
        assert 0.0 <= r <= 1.0 + 1e-9
