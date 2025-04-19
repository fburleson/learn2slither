import pygame
from agent import Agent
from env import Environment
from visual import render_env_to_screen, init_display
from defs import Action, QReward


def transition(env: Environment, agent: Agent, epsilon: float = 0.9):
    obs = agent.observe(env)
    action: Action = agent.policy_epsilon_greedy(obs, epsilon=epsilon)
    reward: QReward = env.step(action)
    if reward == QReward.DEAD:
        env.reset(env.w, env.h)
    return obs, action, reward, agent.observe(env), reward == QReward.DEAD


def train(
    env: Environment,
    agent: Agent,
    n_episodes: int = 1000,
    min_memory_size: int = 1000,
    target_update_freq: int = 100,
    epsilon: float = 1,
    epsilon_decay: float = 0.999,
    epsilon_end: float = 0.2,
    batch_size: int = 1000,
    discount: float = 0.95,
    visual: bool = False,
    visual_refresh_rate: int = 200,
    visual_scale: int = 10,
    stepbystep: bool = False,
    verbose: bool = False,
):
    if visual:
        screen: pygame.Surface = init_display(env, visual_scale)
    iteration: int = 0
    current_episode: int = 0
    while current_episode < n_episodes:
        obs, action, reward, next_obs, is_terminal = transition(env, agent, epsilon)
        agent.store_transition(obs, action, reward, next_obs, is_terminal)
        if agent.n_transitions() >= min_memory_size:
            if visual:
                if not render_env_to_screen(
                    screen,
                    env,
                    refresh_rate=visual_refresh_rate,
                    scale=visual_scale,
                    stepbystep=stepbystep,
                ):
                    return
            agent.update_step(batch_size, discount)
            if iteration % target_update_freq == 0:
                agent.sync()
            epsilon *= epsilon_decay
            if epsilon < epsilon_end:
                epsilon = epsilon_end
            if is_terminal:
                current_episode += 1
            iteration += 1
            if verbose:
                print(
                    f"episode={current_episode} iteration={iteration} epsilon={epsilon}"
                )
