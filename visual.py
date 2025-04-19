import pygame
import numpy as np
from env import Environment
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


def _render_env(scale: int, env: Environment) -> pygame.Surface:
    render: pygame.Surface = pygame.Surface((env.w, env.h))
    pixels: np.ndarray = np.apply_along_axis(
        lambda x: [_env_to_color(x_i) for x_i in x], 1, env.get_env()
    )
    pygame.surfarray.blit_array(render, pixels)
    render = pygame.transform.scale(render, (env.h * scale, env.w * scale))
    return render


def init_display(env: Environment, scale: int = 10) -> pygame.Surface:
    pygame.init()
    return pygame.display.set_mode((env.w * scale, env.h * scale))


def render_env_to_screen(
    screen: pygame.Surface,
    env: Environment,
    refresh_rate: int = 200,
    scale: int = 10,
    stepbystep: bool = False,
) -> bool:
    screen.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    render: pygame.Surface = _render_env(scale, env)
    screen.blit(render, (0, 0))
    pygame.display.flip()
    if stepbystep:
        paused: bool = True
        while paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        paused = False
                        break
    else:
        pygame.time.wait(refresh_rate)
    return True
