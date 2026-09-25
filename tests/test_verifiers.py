import numpy as np

from vdyn.policies import bernoulli
from vdyn.verifiers import toy


def test_gold_rewards_correctness_only():
    np.testing.assert_array_equal(toy.reward_table(toy.gold), [0, 0, 1, 1])


def test_y_verifier_accepts_correct_or_exploit():
    np.testing.assert_array_equal(toy.reward_table(toy.y_false_positive), [0, 1, 1, 1])


def test_r_expected_reward_enumerates_the_flip():
    p = 0.2
    np.testing.assert_allclose(toy.reward_table(toy.r_expected(p)), [p, p, 1 - p, 1 - p])


def test_constant_verifier_ignores_response():
    np.testing.assert_array_equal(toy.reward_table(toy.constant(1.0)), [1, 1, 1, 1])
    np.testing.assert_array_equal(toy.reward_table(toy.constant(0.0)), [0, 0, 0, 0])


def test_fixed_flip_of_incorrect_exploit_outcome_is_the_y_verifier():
    flipped = toy.reward_table(toy.fixed_flip([(0, 1)]))
    np.testing.assert_array_equal(flipped, toy.reward_table(toy.y_false_positive))


def test_r_sample_flips_at_rate_p(rng):
    p, n = 0.3, 400_000
    corr = rng.integers(0, 2, size=n)
    z = rng.integers(0, 2, size=n)
    rewards = toy.r_sample(corr, z, p, rng)
    flip_rate = np.mean(rewards != toy.gold(corr, z))
    assert abs(flip_rate - p) < 5 * np.sqrt(p * (1 - p) / n)


def test_multi_prompt_table_stacks_prompt_tables_in_prompt_major_order():
    table = toy.multi_prompt_table([toy.gold, toy.constant(1.0)])
    np.testing.assert_array_equal(table, [0, 0, 1, 1, 1, 1, 1, 1])
    assert len(table) == 2 * len(bernoulli.OUTCOMES)
