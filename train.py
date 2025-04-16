import torch
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


if __name__ == "__main__":
    env: Environment = Environment(10, 10)
    agent: Agent = Agent(env, lr=0.001, memory_size=100_000)
    train(env, agent)
