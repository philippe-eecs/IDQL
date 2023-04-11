from configs import pixel_config


def get_config():
    config = pixel_config.get_config()

    config.model_cls = "PixelBCLearner"

    config.actor_lr = 3e-4

    config.cosine_decay = True
    config.use_layer_norm = False
    config.dropout_rate = 0.1

    return config
