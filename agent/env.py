import random
from enum import Enum
from typing import final, override

import mlx.core as mx


class Action(Enum):
    FeedOther = 0
    OpenMouth = 1

    @override
    def __repr__(self) -> str:
        match self:
            case Action.FeedOther:
                return "FeedOther"
            case Action.OpenMouth:
                return "OpenMouth"


@final
class FeedMeEnv:
    """
    Observation is each player's action over the previous 10 turns.
    The most recent turn is index 0, the 10th most recent is index 9.
    """

    def __init__(self, max_steps: int = 35, termination_prob: float = 0.04):
        """
        Args:
            max_steps: Maximum episode length
            termination_prob: Probability of episode ending at each step
        """
        self.max_steps = max_steps
        self.termination_prob = termination_prob
        self.reset()

    def obs_space_shape(self) -> tuple[int, int]:
        return self.obs.shape  # pyright: ignore[reportReturnType]

    def action_space_n(self) -> int:
        return len(Action)

    def reset(self) -> tuple[mx.array, mx.array]:
        self.t = 0
        # Observation: [10, 3] = [my_action, opponent_action, is_episode_start]
        self.obs = mx.zeros([10, 3], dtype=mx.float32) - 1.0

        self.obs[0, 2] = 1.0

        # Get hidden termination step by sampling from geometric distribution
        # E[T] = 1/termination_prob
        self.hidden_termination_step = min(
            self.max_steps, int(random.expovariate(self.termination_prob)) + 1
        )

        return self.observation()

    def observation(self) -> tuple[mx.array, mx.array]:
        # Agent A sees: [my_action, opponent_action, is_start]
        a = self.obs
        # Agent B sees swapped actions: [opponent_action, my_action, is_episode_start]
        b = mx.concatenate([self.obs[:, [1, 0]], self.obs[:, [2]]], axis=1)
        return a, b

    def step(
        self,
        action_a: int,
        action_b: int,
    ) -> tuple[tuple[mx.array, mx.array], tuple[int, int], bool]:
        assert self.t < self.hidden_termination_step
        self.obs = mx.roll(self.obs, shift=1, axis=0)  # pyright: ignore[reportCallIssue]
        self.obs[0, 0] = float(action_a)
        self.obs[0, 1] = float(action_b)
        self.obs[0, 2] = -1.0  # Clear the episode start marker for new timestep

        reward_a = self.get_reward(action_b)
        reward_b = self.get_reward(action_a)
        self.t += 1
        reward = (reward_a, reward_b)

        # Episode ends when reaching the hidden termination step
        done = self.t >= self.hidden_termination_step

        return self.observation(), reward, done

    def get_reward(self, opponent_action: int) -> int:
        assert opponent_action == 0 or opponent_action == 1
        a = Action(opponent_action)
        match a:
            case Action.FeedOther:
                return 1
            case Action.OpenMouth:
                return 0


def test_single_step():
    """Test that a single step has the intended effect on self.obs"""
    env = FeedMeEnv(max_steps=35)

    # After reset, should have:
    # obs[0] = [-1, -1, 1] (episode start marker)
    # obs[1:] = [-1, -1, -1] (all uninitialized)
    obs_a, obs_b = env.reset()
    assert obs_a.shape == (10, 3), f"Expected shape (10, 3), got {obs_a.shape}"
    assert float(obs_a[0, 0]) == -1.0, "Initial obs[0, 0] should be -1"
    assert float(obs_a[0, 1]) == -1.0, "Initial obs[0, 1] should be -1"
    assert float(obs_a[0, 2]) == 1.0, "Initial obs[0, 2] should be 1 (episode start)"
    assert float(obs_a[1, 2]) == -1.0, "obs[1, 2] should be -1"

    # Take first step with action_a=0 (FeedOther), action_b=1 (OpenMouth)
    (next_obs_a, next_obs_b), (reward_a, reward_b), done = env.step(0, 1)

    # After step, the observation should be rolled:
    # obs[0] = [0, 1, -1] (most recent: a=0, b=1, not episode start)
    # obs[1] = [-1, -1, 1] (previous: the episode start marker rolled down)
    # obs[2:] = [-1, -1, -1] (older history)
    assert float(env.obs[0, 0]) == 0.0, (
        f"After step, obs[0, 0] should be 0 (action_a), got {float(env.obs[0, 0])}"
    )
    assert float(env.obs[0, 1]) == 1.0, (
        f"After step, obs[0, 1] should be 1 (action_b), got {float(env.obs[0, 1])}"
    )
    assert float(env.obs[0, 2]) == -1.0, (
        f"After step, obs[0, 2] should be -1 (rolled from obs[1]), got {float(env.obs[0, 2])}"
    )
    assert float(env.obs[1, 2]) == 1.0, (
        f"Episode start marker should roll to obs[1, 2], got {float(env.obs[1, 2])}"
    )

    # Rewards: agent A gets reward for opponent action (1=OpenMouth), agent B for action A (0=FeedOther)
    assert reward_a == 0, (
        f"Agent A should get 0 reward (opponent opened mouth), got {reward_a}"
    )
    assert reward_b == 1, f"Agent B should get 1 reward (opponent fed), got {reward_b}"

    # Take second step with action_a=1, action_b=0
    (next_obs_a, next_obs_b), (reward_a, reward_b), done = env.step(1, 0)

    # After second step:
    # obs[0] = [1, 0, -1] (most recent)
    # obs[1] = [0, 1, -1] (previous step rolled down)
    # obs[2] = [-1, -1, 1] (episode start marker rolled to position 2)
    assert float(env.obs[0, 0]) == 1.0, (
        f"After step 2, obs[0, 0] should be 1, got {float(env.obs[0, 0])}"
    )
    assert float(env.obs[0, 1]) == 0.0, (
        f"After step 2, obs[0, 1] should be 0, got {float(env.obs[0, 1])}"
    )
    assert float(env.obs[1, 0]) == 0.0, (
        f"Previous step should be at obs[1, 0], got {float(env.obs[1, 0])}"
    )
    assert float(env.obs[1, 1]) == 1.0, (
        f"Previous step should be at obs[1, 1], got {float(env.obs[1, 1])}"
    )
    assert float(env.obs[2, 2]) == 1.0, (
        f"Episode start marker should roll to obs[2, 2], got {float(env.obs[2, 2])}"
    )

    print("✓ All tests passed!")


if __name__ == "__main__":
    test_single_step()
