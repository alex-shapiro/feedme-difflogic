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
        self.obs = mx.roll(self.obs, shift=1)  # pyright: ignore[reportCallIssue]
        self.obs[0, 0] = float(action_a)
        self.obs[0, 1] = float(action_b)

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
