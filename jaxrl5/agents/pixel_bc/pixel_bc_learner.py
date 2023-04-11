"""Implementations of algorithms for continuous control."""

from functools import partial
from itertools import zip_longest
from typing import Callable, Optional, Sequence, Tuple, Union

import gym
import jax
import optax
from flax import struct
from flax.training.train_state import TrainState

from jaxrl5.agents.bc.bc_learner import BCLearner
from jaxrl5.agents.drq.augmentations import batched_random_crop
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.distributions import TanhNormal
from jaxrl5.networks import MLP, PixelMultiplexer
from jaxrl5.networks.encoders import D4PGEncoder, ResNetV2Encoder


class PixelBCLearner(BCLearner):
    data_augmentation_fn: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: Union[float, optax.Schedule] = 1e-3,
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        encoder: str = "d4pg",
        hidden_dims: Sequence[int] = (256, 256),
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
        distr_name: str = "TanhNormal",
        entropy_bonus: Optional[float] = None,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        depth_keys: Tuple[str, ...] = (),
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng)

        if encoder == "d4pg":
            encoder_cls = partial(
                D4PGEncoder,
                features=cnn_features,
                filters=cnn_filters,
                strides=cnn_strides,
                padding=cnn_padding,
            )
        elif encoder == "resnet":
            encoder_cls = partial(ResNetV2Encoder, stage_sizes=(2, 2, 2, 2))

        actor_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
        )
        actor_cls = partial(
            globals()[distr_name], base_cls=actor_base_cls, action_dim=action_dim
        )
        actor_cls = PixelMultiplexer(
            encoder_cls=encoder_cls,
            network_cls=actor_cls,
            latent_dim=latent_dim,
            pixel_keys=pixel_keys,
            depth_keys=depth_keys,
        )
        actor_params = actor_cls.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_cls.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        def data_augmentation_fn(rng, observations):
            for pixel_key, depth_key in zip_longest(pixel_keys, depth_keys):
                key, rng = jax.random.split(rng)
                observations = batched_random_crop(key, observations, pixel_key)
                if depth_key is not None:
                    observations = batched_random_crop(key, observations, depth_key)
            return observations

        return cls(
            rng=rng,
            actor=actor,
            entropy_bonus=entropy_bonus,
            data_augmentation_fn=data_augmentation_fn,
        )

    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self

        rng, key = jax.random.split(new_agent.rng)
        observations = self.data_augmentation_fn(key, batch["observations"])
        batch = batch.copy(add_or_replace={"observations": observations})

        new_agent = new_agent.replace(rng=rng)

        return BCLearner.update(new_agent, batch)
