import os

import numpy as np
from absl import app, flags

from examples.states.train_offline import call_main
from launcher.hyperparameters import set_hyperparameters


FLAGS = flags.FLAGS
flags.DEFINE_integer('variant', 0, 'Logging interval.')


def main(_):
    constant_parameters = dict(project='iql_one_param',
                               experiment_name='iql',
                               max_steps=1500001,
                               batch_size=256,
                               eval_episodes=50,
                               log_interval=1000,
                               eval_interval=100000,
                               save_video = False,
                               filter_threshold=None,
                               take_top=None,
                               normalize_returns=True,
                               rl_config=dict(
                                   model_cls='IQLLearner',
                                   actor_lr=3e-4,
                                   critic_lr=3e-4,
                                   value_lr=3e-4,
                               ))

    sweep_parameters = dict(
                            env_name=['walker2d-medium-expert-v2', 'halfcheetah-medium-replay-v2', 'hopper-medium-expert-v2', 
                            'antmaze-medium-diverse-v0', 'antmaze-large-diverse-v0', 'antmaze-umaze-v0', 'antmaze-umaze-diverse-v0',
                            'halfcheetah-medium-v2', 'hopper-medium-v2', 'walker2d-medium-v2', 'antmaze-medium-play-v0', 
                            'antmaze-large-play-v0', 'walker2d-medium-replay-v2', 'halfcheetah-medium-expert-v2', 'hopper-medium-replay-v2',
                            'maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1'],
                            temperature = [3, 6.5, 10],
                            seed=list(range(5)),
                            )
    

    variants = [constant_parameters]
    name_keys = ['experiment_name', 'env_name', 'temperature']
    variants = set_hyperparameters(sweep_parameters, variants, name_keys)

    filtered_variants = []
    for variant in variants:
        variant['rl_config']['temperature'] = variant['temperature']
            
        if 'antmaze' in variant['env_name']:
            variant['rl_config']['expectile'] = 0.9
        else:
            variant['rl_config']['expectile'] = 0.7

        filtered_variants.append(variant)

    print(len(filtered_variants))
    variant = filtered_variants[FLAGS.variant]
    print(FLAGS.variant)
    call_main(variant)


if __name__ == '__main__':
    app.run(main)
