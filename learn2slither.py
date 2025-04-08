import numpy as np
import pygame
from pygame import Surface
from game import SnakeGame, GameState, Action, Env


def env_to_color(env: int):
    if env == Env.EMPTY.value:
        return np.array([0, 0, 0])
    if env == Env.WALL.value:
        return np.array([80, 80, 80])
    if env == Env.APPLE_RED.value:
        return np.array([255, 0, 0])
    if env == Env.APPLE_GREEN.value:
        return np.array([0, 255, 0])
    if env == Env.SNAKE_HEAD.value:
        return np.array([0, 0, 255])
    if env == Env.SNAKE_BODY.value:
        return np.array([0, 0, 180])


def render_game(screen: Surface, scale: int, env: np.ndarray):
    render: Surface = pygame.Surface((env.shape[0], env.shape[1]))
    pixels: np.ndarray = np.apply_along_axis(
        lambda x: [env_to_color(x_i) for x_i in x], 1, env
    )
    pygame.surfarray.blit_array(render, pixels)
    render = pygame.transform.scale(
        render, (env.shape[0] * scale, env.shape[1] * scale)
    )
    screen.blit(render, (0, 0))


def main():
    w: int = 10
    h: int = 10
    scale: int = 10
    game: SnakeGame = SnakeGame(w, h)
    screen: Surface = pygame.display.set_mode((w * scale, h * scale))

    pygame.init()
    run: bool = True
    while run:
        screen.fill((0, 0, 0))
        action = Action.IDLE
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_r:
                    game.reset(w, h)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    action = Action.UP
                if event.key == pygame.K_s:
                    action = Action.DOWN
                if event.key == pygame.K_a:
                    action = Action.LEFT
                if event.key == pygame.K_d:
                    action = Action.RIGHT
        if game.update(action) == GameState.GAMEOVER:
            game.reset(w, h)
        render_game(screen, scale, game.env)
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
