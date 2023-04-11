from configs import pixel_config
from ml_collections.config_dict import config_dict


def get_config():
    config = pixel_config.get_config()

    config.model_cls = "DrQLearner"

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.discount = 0.99

    config.num_qs = 2

    config.tau = 0.005
    config.init_temperature = 0.1
    config.target_entropy = config_dict.placeholder(float)

    return config
