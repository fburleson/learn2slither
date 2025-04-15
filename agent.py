import random
from collections import deque
import torch
import numpy as np
from torch import nn
from torch import optim
from env import Env, Action, SnakeGame


def _get_dist_to_wall(game: SnakeGame, direction: np.ndarray) -> int:
    current: np.ndarray = np.array(game.snake[0], copy=True)
    while (current[0] >= 0 and current[0] < game.env.shape[0]) and (
        current[1] >= 0 and current[1] < game.env.shape[1]
    ):
        current += direction
    return np.linalg.norm(current - game.snake[0])


def _get_dist_to_obj(game: SnakeGame, direction: np.ndarray, obj: Env) -> int:
    current: np.ndarray = np.array(game.snake[0], copy=True)
    while game.env[current[0], current[1]] != obj.value:
        current += direction
        if (current[0] < 0 or current[0] >= game.env.shape[0]) or (
            current[1] < 0 or current[1] >= game.env.shape[1]
        ):
            return 0
    return np.linalg.norm(current - game.snake[0])


def _get_state_dir(game: SnakeGame, direction: np.ndarray):
    return [
        _get_dist_to_obj(game, direction, Env.APPLE_GREEN),
        _get_dist_to_obj(game, direction, Env.APPLE_RED),
        _get_dist_to_obj(game, direction, Env.SNAKE_BODY),
        _get_dist_to_wall(game, direction),
    ]


def get_state(game: SnakeGame) -> np.ndarray:
    return torch.as_tensor(
        [
            *_get_state_dir(game, np.array([0, -1])),
            *_get_state_dir(game, np.array([0, 1])),
            *_get_state_dir(game, np.array([-1, 0])),
            *_get_state_dir(game, np.array([1, 0])),
        ],
        dtype=torch.float32,
    )


class DQN(nn.Module):
    def __init__(self, state_size: int, n_actions: int):
        super().__init__()
        self.actions: tuple[Action] = (
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
        )
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Agent:
    def __init__(self, replay_size: int = 10_000, lr: float = 0.01):
        self._actions: tuple[Action] = (
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
        )
        self._experiences: deque = deque(maxlen=replay_size)
        self._net: DQN = DQN(state_size=16, n_actions=len(self._actions))
        self._target_net: DQN = DQN(state_size=16, n_actions=len(self._actions))
        self._optimizer: optim.Optimizer = optim.SGD(self._net.parameters(), lr=lr)
        self.sync()

    def get_buffer_size(self) -> int:
        return len(self._experiences)

    def sync(self):
        self._target_net.load_state_dict(self._net.state_dict())

    def policy_greedy(self, state: torch.Tensor) -> Action:
        return self._actions[torch.argmax(self._net.forward(state))]

    def policy_epsilon_greedy(
        self, state: torch.Tensor, epsilon: float = 0.1
    ) -> Action:
        if random.random() < epsilon:
            return random.choice(self._actions)
        return self.policy_greedy(state)

    def add_transition(
        self,
        state: torch.Tensor,
        action: Action,
        reward: int,
        next_state: torch.Tensor,
        terminal: bool = False,
    ) -> None:
        self._experiences.appendleft(
            [state, action.value, reward, next_state, terminal]
        )

    def update_step(
        self,
        batch_size: int = 32,
        discount: float = 0.99,
    ) -> None:
        states, actions, rewards, next_states, is_terminal = zip(
            *random.sample(self._experiences, batch_size)
        )
        batch = (
            torch.stack(states),
            torch.LongTensor(actions),
            torch.IntTensor(rewards),
            torch.stack(next_states),
            torch.IntTensor(is_terminal),
        )
        q_target: torch.Tensor = self._target_net.forward(batch[3]).max(dim=1)[0]
        targets: torch.Tensor = batch[2] + discount * q_target * (1 - batch[4])
        action_q_values: torch.Tensor = torch.gather(
            input=self._net.forward(batch[0]), dim=0, index=batch[1].unsqueeze(-1)
        )
        loss = nn.functional.smooth_l1_loss(action_q_values.squeeze(), targets)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
