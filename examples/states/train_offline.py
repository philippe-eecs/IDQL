import d4rl
import gym
import jax
import optax
import wandb
from absl import app, flags
from ml_collections import config_flags
from tqdm import tqdm

from jaxrl5.agents import BCLearner, IQLLearner, DDPMIQLLearner
from jaxrl5.data.d4rl_datasets import D4RLDataset
from jaxrl5.evaluation import evaluate
from jaxrl5.wrappers import wrap_gym
from jaxrl5.wrappers.wandb_video import WANDBVideo

def call_main(details):
    wandb.init(project=details['project'], name=details['group'])
    #wandb.config.update(FLAGS)

    env = gym.make(details['env_name'])
    ds = D4RLDataset(env)
    env = wrap_gym(env)
    if details['save_video']:
        env = WANDBVideo(env)

    if details['take_top'] is not None or details['filter_threshold'] is not None:
        ds.filter(take_top=details['take_top'], threshold=details['filter_threshold'])

    config_dict = details['rl_config']
    cosine_decay = config_dict.pop("cosine_decay", False)
    if cosine_decay:
        config_dict["actor_lr"] = optax.cosine_decay_schedule(
            config_dict["actor_lr"], details['max_steps']
        )

    model_cls = config_dict.pop("model_cls")

    if "BC" in model_cls:
        agent = globals()[model_cls].create(
            details['seed'], env.observation_space, env.action_space, **config_dict
        )
        keys = ["observations", "actions"]
    else:
        agent = globals()[model_cls].create(
            details['seed'], env.observation_space, env.action_space, **config_dict
        )
        keys = None

        if "antmaze" in details['env_name']:
            ds.dataset_dict["rewards"] -= 1.0
            # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
            # but I found no difference between (x - 0.5) * 4 and x - 1.0
        elif (
            "halfcheetah" in details['env_name']
            or "walker2d" in details['env_name']
            or "hopper" in details['env_name']
        ):
            ds.normalize_returns()

    for i in tqdm(range(details['max_steps']), smoothing=0.1):
        sample = ds.sample_jax(details['batch_size'], keys=keys)
        agent, info = agent.update(sample)

        if i % details['log_interval'] == 0:
            info = jax.device_get(info)
            wandb.log({f"train/{k}": v for k, v in info.items()}, step=i)

        if i % details['eval_interval'] == 0 and i > 0:
            eval_info = evaluate(
                agent, env, details['eval_episodes'], save_video=details['save_video']
            )
            eval_info["return"] = env.get_normalized_score(eval_info["return"]) * 100.0
            wandb.log({f"eval/{k}": v for k, v in eval_info.items()}, step=i)
