"""Implementations of algorithms for continuous control."""

import math
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union

import gym
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.distributions import Normal
from jaxrl5.networks import MLP, Ensemble, StateActionValue, StateValue


def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class IQLLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    discount: float
    tau: float
    expectile: float
    temperature: float

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_lr: Union[float, optax.Schedule] = 1e-3,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.8,
        temperature: float = 0.1,
        num_qs: int = 2,
    ):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)
        actions = action_space.sample()
        action_dim = action_space.shape[0]
        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)

        actor_def = Normal(
            actor_base_cls,
            action_dim,
            log_std_min=math.log(0.1),
            log_std_max=math.log(0.1),
            state_dependent_std=False,
        )

        observations = observation_space.sample()
        actor_params = actor_def.init(actor_key, observations)["params"]

        actor_optimiser = optax.adam(learning_rate=actor_lr)
        actor = TrainState.create(
            apply_fn=actor_def.apply, params=actor_params, tx=actor_optimiser
        )

        critic_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic_optimiser = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params, tx=critic_optimiser
        )
        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        value_def = StateValue(base_cls=critic_base_cls)
        value_params = value_def.init(value_key, observations)["params"]

        value_optimiser = optax.adam(learning_rate=value_lr)
        value = TrainState.create(
            apply_fn=value_def.apply, params=value_params, tx=value_optimiser
        )

        return cls(
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            value=value,
            tau=tau,
            discount=discount,
            expectile=expectile,
            temperature=temperature,
            rng=rng,
        )

    def update_v(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        q1, q2 = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        q = jnp.minimum(q1, q2)

        def value_loss_fn(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v = agent.value.apply_fn({"params": value_params}, batch["observations"])
            value_loss = loss(q - v, agent.expectile).mean()
            return value_loss, {"value_loss": value_loss, "v": v.mean()}

        grads, info = jax.grad(value_loss_fn, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=grads)

        agent = agent.replace(value=value)

        return agent, info

    def update_q(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        next_v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["next_observations"]
        )

        target_q = batch["rewards"] + agent.discount * batch["masks"] * next_v

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            q1, q2 = agent.critic.apply_fn(
                {"params": critic_params}, batch["observations"], batch["actions"]
            )
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return critic_loss, {
                "critic_loss": critic_loss,
                "q1": q1.mean(),
                "q2": q2.mean(),
            }

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)

        agent = agent.replace(critic=critic)

        target_critic_params = optax.incremental_update(
            critic.params, agent.target_critic.params, agent.tau
        )
        target_critic = agent.target_critic.replace(params=target_critic_params)

        new_agent = agent.replace(critic=critic, target_critic=target_critic)
        return new_agent, info

    def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        v = agent.value.apply_fn({"params": agent.value.params}, batch["observations"])

        q1, q2 = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        q = jnp.minimum(q1, q2)
        exp_a = jnp.exp((q - v) * agent.temperature)
        exp_a = jnp.minimum(exp_a, 100.0)

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = agent.actor.apply_fn(
                {"params": actor_params}, batch["observations"], training=True
            )

            log_probs = dist.log_prob(batch["actions"])
            actor_loss = -(exp_a * log_probs).mean()

            return actor_loss, {"actor_loss": actor_loss, "adv": q - v}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.actor.params)
        actor = agent.actor.apply_gradients(grads=grads)

        agent = agent.replace(actor=actor)

        return agent, info

    @jax.jit
    def update(self, batch: DatasetDict):

        new_agent = self

        new_agent, critic_info = new_agent.update_v(batch)
        new_agent, actor_info = new_agent.update_actor(batch)
        new_agent, value_info = new_agent.update_q(batch)

        return new_agent, {**actor_info, **critic_info, **value_info}
