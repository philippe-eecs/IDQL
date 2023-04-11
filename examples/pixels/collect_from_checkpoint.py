#! /usr/bin/env python
import os
import pickle

import dmcgym
import gym
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from jaxrl5.agents import DrQLearner
from jaxrl5.data import MemoryEfficientReplayBuffer
from jaxrl5.evaluation import evaluate
from jaxrl5.wrappers import wrap_pixels

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "jaxrl5_collect_pixels", "wandb project name.")
flags.DEFINE_string("env_name", "cheetah-run-v0", "Environment name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(5e5), "Number of training steps.")
flags.DEFINE_integer("image_size", 64, "Image size.")
flags.DEFINE_integer("num_stack", 3, "Stack frames.")
flags.DEFINE_integer(
    "replay_buffer_size", None, "Number of training steps to start training."
)
flags.DEFINE_integer(
    "action_repeat", None, "Action repeat, if None, uses 2 or PlaNet default values."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_string("load_dir", None, "Directory to save checkpoints.")
flags.DEFINE_string("save_dir", None, "Directory to save buffers.")
config_flags.DEFINE_config_file(
    "config",
    "configs/drq_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

PLANET_ACTION_REPEAT = {
    "cartpole-swingup-v0": 8,
    "reacher-easy-v0": 4,
    "cheetah-run-v0": 4,
    "finger-spi-n-0": 2,
    "ball_in_cup-catch-v0": 4,
    "walker-walk-v0": 2,
}


def main(_):
    wandb.init(project=FLAGS.project_name)
    wandb.config.update(FLAGS)

    action_repeat = FLAGS.action_repeat or PLANET_ACTION_REPEAT.get(FLAGS.env_name, 2)

    def wrap(env):
        if "quadruped" in FLAGS.env_name:
            camera_id = 2
        else:
            camera_id = 0
        return wrap_pixels(
            env,
            action_repeat=action_repeat,
            image_size=FLAGS.image_size,
            num_stack=FLAGS.num_stack,
            camera_id=camera_id,
        )

    env = gym.make(FLAGS.env_name)
    env, pixel_keys = wrap(env)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)

    eval_env = gym.make(FLAGS.env_name)
    eval_env, _ = wrap(eval_env)
    eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed,
        env.observation_space,
        env.action_space,
        pixel_keys=pixel_keys,
        **kwargs,
    )

    if FLAGS.load_dir is not None:
        from flax.training import checkpoints

        latest_checkpoint = checkpoints.latest_checkpoint(FLAGS.load_dir)
        agent = checkpoints.restore_checkpoint(latest_checkpoint, target=agent)

    replay_buffer_size = FLAGS.replay_buffer_size or FLAGS.max_steps // action_repeat
    replay_buffer = MemoryEfficientReplayBuffer(
        env.observation_space, env.action_space, replay_buffer_size
    )
    replay_buffer.seed(FLAGS.seed)

    observation, done = env.reset(), False
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps // action_repeat + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        action, agent = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i * action_repeat)

        if i % FLAGS.eval_interval == 0:
            eval_info, video = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
                camera_id=2 if "quadruped" in FLAGS.env_name else 0,
            )
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i * action_repeat)

            if FLAGS.save_video:
                video = wandb.Video(video, fps=20, format="mp4")
                wandb.log({"video": video}, step=i * action_repeat)

            if FLAGS.save_dir is not None:
                os.makedirs(FLAGS.save_dir, exist_ok=True)
                buffer_file = os.path.join(FLAGS.save_dir, FLAGS.env_name)
                with open(buffer_file, "wb") as f:
                    pickle.dump(replay_buffer, f)


if __name__ == "__main__":
    app.run(main)
