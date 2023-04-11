from typing import Dict

import gym
import numpy as np

from jaxrl5.wrappers.wandb_video import WANDBVideo


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action, agent = agent.eval_actions(observation)
            observation, _, done, _ = env.step(action)
    return {"return": np.mean(env.return_queue)}

def implicit_evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action, agent = agent.sample_implicit_policy(observation)
            observation, _, done, _ = env.step(action)
    return {"return": np.mean(env.return_queue)}
