import random
import unittest

from src.dgisim.agents import RandomAgent
from src.dgisim.env.linear_env import LinearEnv
from src.dgisim.state.enums import Pid

from dgisim_ext.modes import ForkMode, UpstreamCompatMode
from dgisim_ext.payoff import TerminalReason, per_seat_payoff
from dgisim_ext.step_guard import step_checked


class TestExtModes(unittest.TestCase):
    def test_fork_mode_defaults(self):
        self.assertEqual(ForkMode().timeout_penalty(), -1.0)

    def test_upstream_compat_mode(self):
        self.assertEqual(UpstreamCompatMode().timeout_penalty(), 0.0)


class TestPayoffMatrix(unittest.TestCase):
    """B4：超时双方都输；击杀仍是零和。"""

    def test_timeout_both_lose(self):
        self.assertEqual(
            per_seat_payoff(TerminalReason.TIMEOUT, None, -1.0),
            {Pid.P1: -1.0, Pid.P2: -1.0},
        )

    def test_kill_is_zero_sum(self):
        p1_wins = per_seat_payoff(TerminalReason.KILL, Pid.P1)
        self.assertEqual(p1_wins, {Pid.P1: 1.0, Pid.P2: -1.0})
        p2_wins = per_seat_payoff(TerminalReason.KILL, Pid.P2)
        self.assertEqual(p2_wins, {Pid.P1: -1.0, Pid.P2: 1.0})

    def test_timeout_is_not_zero_sum(self):
        payoff = per_seat_payoff(TerminalReason.TIMEOUT, None)
        self.assertEqual(sum(payoff.values()), -2.0, "超时不是零和：两边都 -1")
        self.assertNotEqual(payoff[Pid.P2], -payoff[Pid.P1])


class TestStepGuard(unittest.TestCase):
    def test_zero_vector_rejected(self):
        env = LinearEnv()
        with self.assertRaises(ValueError):
            step_checked(env, [0] * 81)

    def test_normal_step_produces_transition(self):
        random.seed(11)
        env = LinearEnv()
        agent = RandomAgent()
        pid = env.full_view().waiting_for()
        self.assertIsNotNone(pid)
        outcome = step_checked(env, agent.choose_action([env.full_view()], pid))
        self.assertFalse(outcome.noop, "正常决策不应被判为 noop")
        self.assertIsNone(outcome.error)
