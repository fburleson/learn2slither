import pygame
import numpy as np
from pygame import Surface
from env import SnakeGame, GameState, Env, state_to_reward
from agent import Agent, get_state


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


def train(
    w: int = 10,
    h: int = 10,
    n_sessions: int = 10_000,
    target_update_freq: int = 1000,
    min_replay_size: float = 500,
    visual: bool = False,
    visual_scale: float = 20,
):
    game: SnakeGame = SnakeGame(w, h)
    agent: Agent = Agent(lr=0.001, replay_size=100_000)
    next_state: np.ndarray = get_state(game)
    transition_count: int = 0
    session_count: int = 0
    running: bool = True
    total_reward: int = 0
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.99
    epsilon: float = epsilon_start
    if visual:
        pygame.init()
        screen: Surface = pygame.display.set_mode((w * visual_scale, h * visual_scale))
    while running:
        state: np.ndarray = next_state
        action = agent.policy_epsilon_greedy(state, epsilon=epsilon)
        epsilon = epsilon * epsilon_decay
        if epsilon < epsilon_end:
            epsilon = epsilon_end
        event: GameState = game.update(action)
        reward: int = state_to_reward(event)
        total_reward += reward
        if event == GameState.GAMEOVER:
            game.reset(game.env.shape[0], game.env.shape[1])
            next_state: np.ndarray = get_state(game)
            agent.add_transition(state, action, reward, next_state, True)
            session_count = session_count + 1
            if session_count == n_sessions:
                running = False
            total_reward = 0
            epsilon = epsilon_start
        else:
            next_state: np.ndarray = get_state(game)
            agent.add_transition(state, action, reward, next_state)
        if agent.get_buffer_size() >= min_replay_size:
            transition_count = transition_count + 1
            agent.update_step(discount=0.7, batch_size=100)
        if transition_count % target_update_freq == 0:
            agent.sync()
        if visual:
            screen.fill((0, 0, 0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            render_game(screen, visual_scale, game.env)
            pygame.display.flip()
            pygame.time.wait(100)
    pygame.quit()
    return agent


if __name__ == "__main__":
    trained_agent: Agent = train(10, 10, n_sessions=10_000, visual=True)
