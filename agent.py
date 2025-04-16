import random
from collections import deque
import torch
from torch import nn
from torch import optim
import numpy as np
from env import Environment
from defs import Action, QReward


class DQN(nn.Module):
    def __init__(self, state_size: int, n_actions: int):
        super().__init__()
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
    def __init__(self, env: Environment, memory_size: int = 10_000, lr: float = 0.01):
        self._actions: tuple[Action] = (
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
        )
        self._reward_map: dict = {
            QReward.OK: -1,
            QReward.DEAD: -100,
            QReward.LOSE: -2,
            QReward.GAIN: 2,
        }
        self._transitions: deque = deque(maxlen=memory_size)
        state_size: int = env.w + env.h + 4
        self._net: DQN = DQN(state_size=state_size, n_actions=len(self._actions))
        self._target_net: DQN = DQN(state_size=state_size, n_actions=len(self._actions))
        self._optimizer: optim.Optimizer = optim.SGD(self._net.parameters(), lr=lr)
        self.sync()

    def sync(self):
        self._target_net.load_state_dict(self._net.state_dict())

    def observe(self, env: Environment) -> torch.Tensor:
        state: np.ndarray = env.get_env()
        head: np.ndarray = env.snake[0]
        horizontal: np.ndarray = state[head[1]]
        vertical: np.ndarray = np.array(
            [env.get_cell(head[0], y) for y in range(env.h + 2)]
        )
        return torch.as_tensor([*horizontal, *vertical], dtype=torch.float32)

    def policy_greedy(self, obs: torch.Tensor) -> Action:
        return self._actions[torch.argmax(self._net.forward(obs))]

    def policy_epsilon_greedy(self, obs: torch.Tensor, epsilon: float = 0.1) -> Action:
        if random.random() < epsilon:
            return random.choice(self._actions)
        return self.policy_greedy(obs)

    def store_transition(
        self,
        obs: torch.Tensor,
        action: Action,
        reward: QReward,
        next_obs: torch.Tensor,
        terminal: bool,
    ) -> None:
        self._transitions.append(
            [obs, action.value, self._reward_map[reward], next_obs, terminal]
        )

    def update_step(
        self,
        batch_size: int = 32,
        discount: float = 0.99,
    ) -> None:
        obs, actions, rewards, next_obs, is_terminal = zip(
            *random.sample(self._transitions, batch_size)
        )
        batch = (
            torch.stack(obs),
            torch.LongTensor(actions),
            torch.IntTensor(rewards),
            torch.stack(next_obs),
            torch.IntTensor(is_terminal),
        )
        q_target: torch.Tensor = self._target_net.forward(batch[3]).max(
            dim=1, keepdims=True
        )[0]
        targets: torch.Tensor = batch[2] + discount * q_target * (1 - batch[4])
        action_q_values: torch.Tensor = torch.gather(
            self._net.forward(batch[0]), 1, batch[1].unsqueeze(-1)
        )
        loss = nn.functional.smooth_l1_loss(action_q_values, targets)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
