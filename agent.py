import random
from collections import deque
import torch
from torch import nn
from torch import optim
import numpy as np
from env import Environment
from defs import Action, QReward, EnvID


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
    def __init__(self, memory_size: int = 10_000, lr: float = 0.01):
        self._actions: tuple[Action] = (
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
        )
        self._reward_map: dict = {
            QReward.OK: -2.5,
            QReward.DEAD: -100,
            QReward.LOSE: -90,
            QReward.GAIN: 1000,
        }
        self._obs_map: dict = {
            EnvID.WALL.value: 0,
            EnvID.APPLE_RED.value: 1,
            EnvID.APPLE_GREEN.value: 2,
            EnvID.SNAKE_BODY.value: 3,
        }
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._transitions: deque = deque(maxlen=memory_size)
        state_size: int = len(self._obs_map) * 4
        self._net: DQN = DQN(state_size=state_size, n_actions=len(self._actions)).to(
            self._device
        )
        self._target_net: DQN = DQN(
            state_size=state_size, n_actions=len(self._actions)
        ).to(self._device)
        self._optimizer: optim.Optimizer = optim.Adam(self._net.parameters(), lr=lr)
        self.sync()

    def sync(self):
        self._target_net.load_state_dict(self._net.state_dict())

    def _observe_dir(self, dir: np.ndarray) -> np.ndarray:
        dir_obs: np.ndarray = np.zeros(len(self._obs_map), dtype=int)
        for cell_idx in range(len(dir)):
            cell: int = dir[cell_idx]
            if cell != EnvID.EMPTY.value:
                dir_obs[self._obs_map[cell]] = cell_idx + 1
                return dir_obs

    def observe(self, env: Environment) -> torch.Tensor:
        state: np.ndarray = env.get_env()
        head: np.ndarray = env.snake[0]
        left: np.ndarray = state[head[1]][: head[0]][::-1]
        right: np.ndarray = state[head[1]][head[0] + 1 :]
        up: np.ndarray = state[:, head[0]][: head[1]][::-1]
        down: np.ndarray = state[:, head[0]][head[1] + 1 :]
        obs_left: np.ndarray = self._observe_dir(left)
        obs_right: np.ndarray = self._observe_dir(right)
        obs_up: np.ndarray = self._observe_dir(up)
        obs_down: np.ndarray = self._observe_dir(down)
        return torch.as_tensor(
            np.concatenate((obs_left, obs_right, obs_up, obs_down), axis=None),
            dtype=torch.float32,
        ).to(self._device)

    def policy_greedy(self, obs: torch.Tensor) -> Action:
        return self._actions[torch.argmax(self._net.forward(obs))]

    def policy_epsilon_greedy(self, obs: torch.Tensor, epsilon: float = 0.1) -> Action:
        if random.random() < epsilon:
            return random.choice(self._actions)
        return self.policy_greedy(obs)

    def n_transitions(self) -> int:
        return len(self._transitions)

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
            torch.LongTensor(actions).to(self._device),
            torch.IntTensor(rewards).to(self._device),
            torch.stack(next_obs),
            torch.IntTensor(is_terminal).to(self._device),
        )
        q_target: torch.Tensor = self._target_net.forward(batch[3]).max(
            dim=1, keepdims=True
        )[0]
        targets: torch.Tensor = batch[2].unsqueeze(-1) + discount * q_target * (
            1 - batch[4].unsqueeze(-1)
        )
        action_q_values: torch.Tensor = torch.gather(
            self._net.forward(batch[0]), 1, batch[1].unsqueeze(-1)
        )
        loss = nn.functional.smooth_l1_loss(action_q_values, targets)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
