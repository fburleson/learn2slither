import numpy as np
import pygame
from pygame import Surface
from defs import EnvID


def _env_to_color(env: int):
    if env == EnvID.EMPTY.value:
        return np.array([0, 0, 0])
    if env == EnvID.WALL.value:
        return np.array([80, 80, 80])
    if env == EnvID.APPLE_RED.value:
        return np.array([255, 0, 0])
    if env == EnvID.APPLE_GREEN.value:
        return np.array([0, 255, 0])
    if env == EnvID.SNAKE_HEAD.value:
        return np.array([0, 0, 255])
    if env == EnvID.SNAKE_BODY.value:
        return np.array([0, 0, 180])


def render_game(scale: int, env: np.ndarray) -> Surface:
    render: Surface = pygame.Surface((env.shape[0], env.shape[1]))
    pixels: np.ndarray = np.apply_along_axis(
        lambda x: [_env_to_color(x_i) for x_i in x], 1, env
    )
    pygame.surfarray.blit_array(render, pixels)
    render = pygame.transform.scale(
        render, (env.shape[0] * scale, env.shape[1] * scale)
    )
    return render
