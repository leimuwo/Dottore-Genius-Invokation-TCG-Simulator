import itertools
import random
import unittest
from unittest.mock import patch

from dgisim_ext.env_api import CardEnv, DEFAULT_PLAN
from src.dgisim.dice import ActualDice, AbstractDice
from src.dgisim.element import Element
from src.dgisim.card.cards import Cards
from src.dgisim.card.card import Paimon, Liben
from src.dgisim.state.enums import Pid


class TestCardEnvM1(unittest.TestCase):
    def test_fast_plan_matches_native_mapping_and_perspective(self):
        from dgisim_ext.encoding import FastEncodingPlan
        from src.dgisim.encoding.encoding_plan import EncodingPlan
        self.assertIsInstance(DEFAULT_PLAN, FastEncodingPlan)
        for seat in Pid:
            plan = DEFAULT_PLAN.perspective_version(seat)
            for mapping in plan._TYPED_MAPPING.values():
                for item in mapping:
                    self.assertEqual(plan.encode_item(item), EncodingPlan.encode_item(plan, item))

    def test_noop_illegal_and_exception_are_distinct(self):
        from src.dgisim.action.action import EndRoundAction
        from src.dgisim.action.action_generator import ActionGenerator
        for category in ('noop', 'illegal', 'exception'):
            env = CardEnv()
            env.reset(seed=0)
            env._gen = ActionGenerator(game_state=env.state, pid=env.seat, action=EndRoundAction())
            env._stage_changed()
            kwargs = ({'side_effect': RuntimeError('sentinel')} if category == 'exception'
                      else {'return_value': env.state if category == 'noop' else None})
            with patch.object(type(env.state), 'action_step', **kwargs):
                with self.assertRaises(RuntimeError) as caught:
                    env.step(0)
            self.assertEqual(caught.exception.category, category)
            self.assertEqual(env.stats[category], 1)
            self.assertTrue(env.failed)

    def test_private_choice_prefix_is_not_shared_with_opponent(self):
        env = CardEnv()
        env.reset(seed=0)
        actor = env.seat
        env.step(1)
        self.assertTrue(env.obs(actor).meta['choice_prefix'])
        self.assertFalse(env.obs(actor.other).meta['choice_prefix'])
        self.assertEqual(env.obs(actor.other).stage_index, 0)

    def test_terminal_reward_is_for_actor_not_next_seat(self):
        env = CardEnv()
        env.reset(seed=2)
        policy = random.Random(95)
        while not env.done:
            obs = env.obs()
            actor = obs.seat
            out = env.step(policy.randrange(len(obs.choices)))
            self.assertEqual(out.info['actor'], actor.name)
            self.assertEqual(out.reward, out.info['rewards'][actor])
        self.assertIn(out.terminal_reason.split(':')[0], ('kill', 'timeout'))

    def test_payment_pruning_for_same_element_and_free_costs(self):
        from dgisim_ext.selection import Selection
        raw = ActualDice({Element.PYRO: 2, Element.HYDRO: 1, Element.OMNI: 1})
        for cost in (AbstractDice({}), AbstractDice({Element.OMNI: 2}),
                     AbstractDice({Element.ANY: 3}), AbstractDice({Element.HYDRO: 2})):
            leaves = self.leaves(Selection.payment(raw, cost))
            expected = {ActualDice(dict(zip((Element.PYRO, Element.HYDRO, Element.OMNI), ns)))
                        for ns in itertools.product(range(3), range(2), range(2))
                        if ActualDice(dict(zip((Element.PYRO, Element.HYDRO, Element.OMNI), ns))).just_satisfy(cost)}
            self.assertEqual(set(leaves), expected)

    def test_all_dice_subsets_reachable(self):
        from dgisim_ext.selection import Selection
        raw = ActualDice({Element.PYRO: 2, Element.HYDRO: 1})
        leaves = self.leaves(Selection.dice(raw))
        self.assertEqual(len(leaves), 6)
        self.assertEqual({(x[Element.PYRO], x[Element.HYDRO]) for x in leaves},
                         set(itertools.product(range(3), range(2))))

    @staticmethod
    def leaves(selection):
        if selection.done:
            return [selection.result()]
        return [leaf for n in selection.counts()
                for leaf in TestCardEnvM1.leaves(selection.choose(n))]

    def test_payment_covers_all_and_only_valid_subsets(self):
        from dgisim_ext.selection import Selection
        raw = ActualDice({Element.PYRO: 2, Element.HYDRO: 1, Element.OMNI: 1})
        cost = AbstractDice({Element.PYRO: 1, Element.ANY: 1})
        leaves = self.leaves(Selection.payment(raw, cost))
        expected = {ActualDice(dict(zip((Element.PYRO, Element.HYDRO, Element.OMNI), ns)))
                    for ns in itertools.product(range(3), range(2), range(2))
                    if ActualDice(dict(zip((Element.PYRO, Element.HYDRO, Element.OMNI), ns))).just_satisfy(cost)}
        self.assertEqual(set(leaves), expected)

    def test_exchange_can_select_none_many_and_duplicates(self):
        from dgisim_ext.selection import Selection
        leaves = self.leaves(Selection.cards(Cards({Paimon: 2, Liben: 1})))
        self.assertEqual({(x[Paimon], x[Liben]) for x in leaves},
                         set(itertools.product(range(3), range(2))))

    def test_repeated_and_intermediate_observation_reuses_encoding(self):
        env = CardEnv()
        obs = env.reset(seed=2)
        with patch.object(DEFAULT_PLAN, 'encode', wraps=DEFAULT_PLAN.encode) as encode:
            self.assertEqual(env.obs().enc, obs.enc)
            out = env.step(0)
            self.assertEqual(out.obs.enc, obs.enc)
            self.assertEqual(encode.call_count, 0)
            self.assertNotEqual(out.obs.meta['choice_prefix'], obs.meta['choice_prefix'])

    def test_invalid_choice_fails_closed_and_is_counted(self):
        env = CardEnv()
        env.reset(seed=1)
        with self.assertRaises((ValueError, IndexError)):
            env.step(-1)
        self.assertEqual(env.stats['illegal'], 1)
        self.assertTrue(env.failed)
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_fork_does_not_consume_external_random_stream(self):
        random.seed(203)
        before = random.getstate()
        env = CardEnv()
        env.reset(seed=12)
        self.assertEqual(before, random.getstate())
        fork = env.fork()
        for _ in range(120):
            if env.done:
                break
            a, b = env.obs(), fork.obs()
            self.assertEqual(a, b)
            idx = len(a.choices) - 1
            x, y = env.step(idx), fork.step(idx)
            self.assertEqual(x, y)
        self.assertEqual(before, random.getstate())

    def test_cold_encoding_equals_native_for_both_seats(self):
        env = CardEnv()
        env.reset(seed=1)
        for _ in range(25):
            for seat in Pid:
                actual = env.obs(seat).enc
                expected = tuple(DEFAULT_PLAN.encode(env.state.perspective_view(seat), seat))
                self.assertEqual(actual, expected)
            if env.done:
                break
            env.step(0)


if __name__ == '__main__':
    unittest.main()
