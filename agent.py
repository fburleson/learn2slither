import numpy as np
from torch import nn
from game import Env, SnakeGame


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
    return np.array(
        [
            _get_dist_to_obj(game, direction, Env.APPLE_GREEN),
            _get_dist_to_obj(game, direction, Env.APPLE_RED),
            _get_dist_to_obj(game, direction, Env.SNAKE_BODY),
            _get_dist_to_wall(game, direction),
        ]
    )


def get_state(game: SnakeGame) -> np.ndarray:
    return np.array(
        [
            _get_state_dir(game, np.array([0, -1])),
            _get_state_dir(game, np.array([0, 1])),
            _get_state_dir(game, np.array([-1, 0])),
            _get_state_dir(game, np.array([1, 0])),
        ]
    )


class Agent(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
