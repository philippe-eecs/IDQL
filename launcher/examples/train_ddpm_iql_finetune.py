import os

import numpy as np
from absl import app, flags

from examples.states.train_diffusion_offline import call_main
from launcher.hyperparameters import set_hyperparameters


FLAGS = flags.FLAGS
flags.DEFINE_integer('variant', 0, 'Logging interval.')


def main(_):
    constant_parameters = dict(project='final_finetune_2',
                               experiment_name='finetune_iql',
                               max_steps=1000001,
                               batch_size=2048, #Actor batch size, critic is fixed to 256
                               eval_episodes=50,
                               log_interval=1000,
                               eval_interval=1000000,
                               save_video = False,
                               filter_threshold=None,
                               take_top=None,
                               online_start_training = 5000, #How many transitions are taken before finetuning starts
                               online_max_steps=1000001,
                               online_eval_episodes=25,
                               online_eval_interval = 25000,
                               unsquash_actions=False,
                               normalize_returns=True,
                               sample_implicit_policy=False, #Parameters for (1) finetuning, change both to true for our (2) Fine tuning version
                               train_actor_finetuning=False,
                               training_time_inference_params=dict(
                                N = 64,
                                clip_sampler = True,
                                M = 0,),
                               rl_config=dict(
                                   model_cls='DDPMIQLLearner',
                                   actor_lr=3e-4,
                                   critic_lr=3e-4,
                                   value_lr=3e-4,
                                   T=5,
                                   N=64,
                                   M=0,
                                   actor_dropout_rate=0.1, 
                                   actor_num_blocks=3,
                                   actor_weight_decay=None,
                                   actor_tau=0.001,
                                   actor_architecture='ln_resnet',
                                   critic_objective='expectile',
                                   beta_schedule='vp',
                                   actor_objective='bc',
                                   decay_steps=int(2e6), #Change this to int(4e6) for (2) (because you are finetuning actor)
                                   actor_layer_norm=True,
                               ))

    sweep_parameters = dict(
                            seed=list(range(10)),
                            env_name=['antmaze-umaze-v0', 'antmaze-umaze-diverse-v0', 'antmaze-medium-diverse-v0',
                             'antmaze-medium-play-v0','antmaze-large-diverse-v0', 'antmaze-large-play-v0'],                        
                            )
    

    variants = [constant_parameters]
    name_keys = ['experiment_name', 'env_name']
    variants = set_hyperparameters(sweep_parameters, variants, name_keys)

    inference_sweep_parameters = dict(
                            N = [128],
                            clip_sampler=[True],
                            M = [0],
                            )
    
    inference_variants = [{}]
    name_keys = []
    inference_variants = set_hyperparameters(inference_sweep_parameters, inference_variants)

    filtered_variants = []
    for variant in variants:
        variant['inference_variants'] = inference_variants
            
        if 'antmaze' in variant['env_name']:
            variant['rl_config']['critic_hyperparam'] = 0.9
        elif 'binary' in variant['env_name']:
            variant['rl_config']['critic_hyperparam'] = 0.7
        else:
            variant['rl_config']['critic_hyperparam'] = 0.7

        filtered_variants.append(variant)

    print(len(filtered_variants))
    variant = filtered_variants[FLAGS.variant]
    print(FLAGS.variant)
    call_main(variant)


if __name__ == '__main__':
    app.run(main)
