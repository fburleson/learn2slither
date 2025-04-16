import torch
import pygame
from pygame import Surface
from visual import render_game
from agent import Agent
from env import Environment
from defs import Action, QReward


def train(
    env: Environment,
    agent: Agent,
    batch_size: int = 200,
    min_transitions: int = 5000,
    target_update_freq: int = 1000,
    epsilon_start: float = 0.9,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.99999,
    visual: bool = False,
    screen: Surface = None,
    refresh_rate: float = 200,
) -> None:
    iteration: int = 0
    episode_total_reward: int = 0
    epsilon: float = epsilon_start
    next_obs: torch.Tensor = agent.observe(env)
    while True:
        obs = next_obs
        action: Action = agent.policy_epsilon_greedy(obs, epsilon=epsilon)
        reward: QReward = env.step(action)
        if reward == QReward.DEAD:
            env.reset(env.w, env.h)
            print(episode_total_reward)
            episode_total_reward = 0
        next_obs = agent.observe(env)
        agent.store_transition(obs, action, reward, next_obs, reward == QReward.DEAD)
        episode_total_reward += agent._transitions[-1][2]
        epsilon *= epsilon_decay
        if epsilon < epsilon_end:
            epsilon = epsilon_end
        if len(agent._transitions) >= min_transitions:
            agent.update_step(batch_size, 0.99)
            if iteration % min_transitions == 0:
                agent.sync()
            iteration += 1
            if visual:
                screen.fill((0, 0, 0))
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                render: Surface = render_game(10, env.get_env())
                screen.blit(render, (0, 0))
                pygame.display.flip()
                pygame.time.wait(refresh_rate)


if __name__ == "__main__":
    pygame.init()
    screen: Surface = pygame.display.set_mode((120, 120))
    env: Environment = Environment(10, 10)
    agent: Agent = Agent(env, lr=0.001, memory_size=100_000)
    train(env, agent, visual=True, screen=screen, refresh_rate=100)
    pygame.quit()
