from configs import sac_config


def get_config():
    config = sac_config.get_config()

    config.critic_dropout_rate = 0.01
    config.critic_layer_norm = True

    return config
