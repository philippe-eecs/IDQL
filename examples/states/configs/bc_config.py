import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.model_cls = "BCLearner"

    config.actor_lr = 1e-3
    config.hidden_dims = (256, 256)
    config.cosine_decay = True
    config.use_layer_norm = False
    config.dropout_rate = 0.1
    config.weight_decay = config_dict.placeholder(float)
    config.entropy_bonus = config_dict.placeholder(float)

    return config
