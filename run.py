import sys
import pygame
from defs import RunModes, QReward
from agent import load_agent, Agent
from env import Environment
from train import train
from visual import init_display, render_env_to_screen


def play(
    env: Environment,
    agent: Agent,
    visual_refresh_rate: int = 200,
    visual_scale: int = 10,
):
    screen: pygame.Surface = init_display(env, visual_scale)
    run: bool = True
    max_length: int = env.snake.shape[0]
    while run:
        obs = agent.observe(env)
        action = agent.policy_greedy(obs)
        if env.snake.shape[0] > max_length:
            max_length = env.snake.shape[0]
        if env.step(action) == QReward.DEAD:
            print(f"game over - max length={max_length} - length={env.snake.shape[0]}")
            env.reset(env.w, env.h)
            max_length = env.snake.shape[0]
        if not render_env_to_screen(
            screen, env, refresh_rate=visual_refresh_rate, scale=visual_scale
        ):
            run = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset(env.w, env.h)
                    max_length = env.snake.shape[0]


def run(
    mode: RunModes = RunModes.PLAY,
    w: int = 10,
    h: int = 10,
    load_from: str = None,
    save_to: str = None,
    n_episodes: int = 2000,
    epsilon_decay: float = 0.999,
    lr: float = 0.001,
    verbose: bool = True,
    visual: bool = False,
    visual_refresh_rate: float = 200,
):
    print(f"running in {mode.name.lower()} mode")
    env: Environment = Environment(w, h)
    if load_from is None:
        agent: Agent = Agent(lr=lr, memory_size=1_000_000)
    else:
        agent: Agent = load_agent(load_from)
    if mode == RunModes.PLAY:
        play(env, agent, visual_refresh_rate=visual_refresh_rate)
    elif mode == RunModes.TRAIN:
        try:
            train(
                env,
                agent,
                n_episodes=n_episodes,
                epsilon_decay=epsilon_decay,
                epsilon_end=0.2,
                discount=0.9,
                batch_size=32,
                min_memory_size=1024,
                target_update_freq=100,
                verbose=verbose,
                visual=visual,
                visual_refresh_rate=visual_refresh_rate,
            )
        except KeyboardInterrupt:
            should_save: bool = True if input("save model? (y/n): ") == "y" else False
            if not should_save:
                return
        if save_to is not None:
            agent.save(save_to)
    else:
        print("wrong argument")


def _get_arg_value(arg: str, fallback=None):
    return sys.argv[sys.argv.index(arg) + 1] if arg in sys.argv else fallback


def _arg_to_value(arg: str, value, fallback):
    if arg in sys.argv:
        return value
    return fallback


def main():
    run(
        mode=_arg_to_value("-train", RunModes.TRAIN, RunModes.PLAY),
        w=int(_get_arg_value("-w", 10)),
        h=int(_get_arg_value("-h", 10)),
        load_from=_get_arg_value("-load", None),
        save_to=_get_arg_value("-save", None),
        n_episodes=int(_get_arg_value("-sessions", 2000)),
        epsilon_decay=float(_get_arg_value("-decay", 0.995)),
        lr=float(_get_arg_value("-lr", 0.001)),
        verbose=_arg_to_value("-verbose", True, False),
        visual=_arg_to_value("-visual", True, False),
        visual_refresh_rate=int(_get_arg_value("-fpms", 50)),
    )


if __name__ == "__main__":
    main()
