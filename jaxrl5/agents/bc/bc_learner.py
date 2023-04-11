"""Implementations of algorithms for continuous control."""

import math
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union

import flax
import gym
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.distributions import Normal
from jaxrl5.networks import MLP


def get_weight_decay_mask(params):
    flattened_params = flax.traverse_util.flatten_dict(
        flax.core.frozen_dict.unfreeze(params)
    )

    def decay(k, v):
        if any([(key == "bias") for key in k]):
            return False
        else:
            return True

    return flax.core.frozen_dict.freeze(
        flax.traverse_util.unflatten_dict(
            {k: decay(k, v) for k, v in flattened_params.items()}
        )
    )


class BCLearner(Agent):
    entropy_bonus: Optional[float]

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_lr: Union[float, optax.Schedule] = 1e-3,
        hidden_dims: Sequence[int] = (256, 256),
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        entropy_bonus: Optional[float] = None,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng)

        action_dim = action_space.shape[0]
        base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
        )

        actor_def = Normal(
            base_cls,
            action_dim,
            log_std_min=math.log(0.1),
            log_std_max=math.log(0.1),
            state_dependent_std=False,
        )

        if weight_decay is None:
            optimiser = optax.adam(learning_rate=actor_lr)
        else:
            optimiser = optax.adamw(
                learning_rate=actor_lr,
                weight_decay=weight_decay,
                mask=get_weight_decay_mask,
            )

        observations = observation_space.sample()
        params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(apply_fn=actor_def.apply, params=params, tx=optimiser)

        return cls(actor=actor, rng=rng, entropy_bonus=entropy_bonus)

    @jax.jit
    def update(self, batch: DatasetDict):
        rng, key1, key2 = jax.random.split(self.rng, 3)

        def loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = self.actor.apply_fn(
                {"params": actor_params},
                batch["observations"],
                training=True,
                rngs={"dropout": key1},
            )
            nll = -dist.log_prob(batch["actions"]).mean()

            actor_loss = nll
            action = dist.sample(seed=key2)
            eps = 1e-5
            action = jax.lax.stop_gradient(jnp.clip(action, -1 + eps, 1 - eps))
            log_prob = dist.log_prob(action)

            if self.entropy_bonus is not None:
                entropy_grad = (-log_prob * jax.lax.stop_gradient(log_prob)).mean()

                actor_loss -= self.entropy_bonus * entropy_grad

            return actor_loss, {"nll": nll, "entropy": -log_prob.mean()}

        grads, info = jax.grad(loss_fn, has_aux=True)(self.actor.params)
        new_actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=new_actor, rng=rng), info
