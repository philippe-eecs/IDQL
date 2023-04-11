from configs import iql_antmaze_config


def get_config():
    config = iql_antmaze_config.get_config()

    config.expectile = 0.7  # The actual tau for expectiles.
    config.temperature = 3.0

    return config
