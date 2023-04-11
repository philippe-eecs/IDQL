from configs import td_config


def get_config():
    config = td_config.get_config()

    config.model_cls = "TD3Learner"

    config.exploration_noise = 0.1
    config.target_policy_noise = 0.2
    config.target_policy_noise_clip = 0.5

    config.actor_delay = 2

    return config
