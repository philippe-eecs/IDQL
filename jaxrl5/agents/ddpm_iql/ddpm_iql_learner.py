"""Implementations of algorithms for continuous control."""
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax import struct
import numpy as np
from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import MLP, Ensemble, StateActionValue, StateValue, DDPM, FourierFeatures, cosine_beta_schedule, ddpm_sampler, MLPResNet, get_weight_decay_mask, vp_beta_schedule

def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

def quantile_loss(diff, quantile=0.6):
    weight = jnp.where(diff > 0, quantile, (1 - quantile))
    return weight * jnp.abs(diff)

def exp_w_clip(x, x0, mode='zero'):
  if mode == 'zero':
    return jnp.where(x < x0, jnp.exp(x), jnp.exp(x0))
  elif mode == 'first':
    return jnp.where(x < x0, jnp.exp(x), jnp.exp(x0) + jnp.exp(x0) * (x - x0))
  elif mode == 'second':
    return jnp.where(x < x0, jnp.exp(x), jnp.exp(x0) + jnp.exp(x0) * (x - x0) + (jnp.exp(x0) / 2) * ((x - x0)**2))
  else:
    raise ValueError()

def exponential_loss(diff, beta, clip=jnp.log(6), mode='zero'):
    exp_diff = exp_w_clip(diff * beta, clip, mode=mode)
    exp_diff = jax.lax.stop_gradient(exp_diff)
    return (exp_diff - 1) * (diff)

@partial(jax.jit, static_argnames=('critic_fn'))
def compute_q(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({'params': critic_params}, observations, actions)
    q_values = q_values.min(axis=0)
    return q_values

@partial(jax.jit, static_argnames=('value_fn'))
def compute_v(value_fn, value_params, observations):
    v_values = value_fn({'params': value_params}, observations)
    return v_values

def mish(x):
    return x * jnp.tanh(nn.softplus(x))

class DDPMIQLLearner(Agent):
    score_model: TrainState
    target_score_model: TrainState
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    discount: float
    tau: float
    actor_tau: float
    critic_hyperparam: float
    critic_objective: str = struct.field(pytree_node=False)
    actor_objective: str = struct.field(pytree_node=False)
    act_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    N: int #How many samples per observation
    M: int = struct.field(pytree_node=False) #How many repeat last steps
    clip_sampler: bool = struct.field(pytree_node=False)
    ddpm_temperature: float
    policy_temperature: float
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_architecture: str = 'mlp',
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        critic_hyperparam: float = 0.7,
        ddpm_temperature: float = 1.0,
        num_qs: int = 2,
        actor_num_blocks: int = 2,
        actor_weight_decay: Optional[float] = None,
        actor_tau: float = 0.001,
        actor_dropout_rate: Optional[float] = None,
        actor_layer_norm: bool = False,
        policy_temperature: float = 3.0,
        T: int = 5,
        time_dim: int = 64,
        N: int = 64,
        M: int = 0,
        clip_sampler: bool = True,
        actor_objective: str = 'bc',
        critic_objective: str = 'expectile',
        beta_schedule: str = 'vp',
        decay_steps: Optional[int] = int(2e6),
    ):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]

        preprocess_time_cls = partial(FourierFeatures,
                                      output_size=time_dim,
                                      learnable=True)

        cond_model_cls = partial(MLP,
                                hidden_dims=(128, 128),
                                activations=mish,
                                activate_final=False)
        
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if actor_architecture == 'mlp':
            base_model_cls = partial(MLP,
                                    hidden_dims=tuple(list(actor_hidden_dims) + [action_dim]),
                                    activations=mish,
                                    use_layer_norm=actor_layer_norm,
                                    activate_final=False)
            
            actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                             cond_encoder_cls=cond_model_cls,
                             reverse_encoder_cls=base_model_cls)

        elif actor_architecture == 'ln_resnet':

            base_model_cls = partial(MLPResNet,
                                     use_layer_norm=actor_layer_norm,
                                     num_blocks=actor_num_blocks,
                                     dropout_rate=actor_dropout_rate,
                                     out_dim=action_dim,
                                     activations=mish)
            
            actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                             cond_encoder_cls=cond_model_cls,
                             reverse_encoder_cls=base_model_cls)

        else:
            raise ValueError(f'Invalid actor architecture: {actor_architecture}')
        
        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis = 0)
        actions = jnp.expand_dims(actions, axis = 0)
        actor_params = actor_def.init(actor_key, observations, actions,
                                        time)['params']

        score_model = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params,
                                        tx=optax.adamw(learning_rate=actor_lr, 
                                                       weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
                                                       mask=get_weight_decay_mask,))
        
        target_score_model = TrainState.create(apply_fn=actor_def.apply,
                                               params=actor_params,
                                               tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None))

        critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True)
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

        value = TrainState.create(apply_fn=value_def.apply,
                                  params=value_params,
                                  tx=value_optimiser)

        if beta_schedule == 'cosine':
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == 'linear':
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == 'vp':
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f'Invalid beta schedule: {beta_schedule}')

        alphas = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas[:i + 1]) for i in range(T)])

        return cls(
            actor=None,
            score_model=score_model,
            target_score_model=target_score_model,
            critic=critic,
            target_critic=target_critic,
            value=value,
            tau=tau,
            discount=discount,
            rng=rng,
            betas=betas,
            alpha_hats=alpha_hat,
            act_dim=action_dim,
            T=T,
            N=N,
            M=M,
            alphas=alphas,
            ddpm_temperature=ddpm_temperature,
            actor_tau=actor_tau,
            actor_objective=actor_objective,
            critic_objective=critic_objective,
            critic_hyperparam=critic_hyperparam,
            clip_sampler=clip_sampler,
            policy_temperature=policy_temperature,
        )

    def update_v(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        q = qs.min(axis=0)

        def value_loss_fn(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v = agent.value.apply_fn({"params": value_params}, batch["observations"])

            if agent.critic_objective == 'expectile':
                value_loss = expectile_loss(q - v, agent.critic_hyperparam).mean()
            elif agent.critic_objective == 'quantile':
                value_loss = quantile_loss(q - v, agent.critic_hyperparam).mean()
            elif agent.critic_objective == 'exponential':
                value_loss = exponential_loss(q - v, agent.critic_hyperparam).mean()
            else:
                raise ValueError(f'Invalid critic objective: {agent.critic_objective}')

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
        batch_size = batch["observations"].shape[0]

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn(
                {"params": critic_params}, batch["observations"], batch["actions"]
            )
            critic_loss = ((qs - target_q) ** 2).mean()

            return critic_loss, {
                "critic_loss": critic_loss,
                "q": qs.mean(),
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
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)
        time = jax.random.randint(key, (batch['actions'].shape[0], ), 0, agent.T)
        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(
            key, (batch['actions'].shape[0], agent.act_dim))
        
        alpha_hats = agent.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng, 2)

        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )

        q = qs.min(axis=0)

        v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["observations"]
        )

        adv = q - v

        if agent.actor_objective == "soft_adv":
            weights = jnp.where(adv > 0, agent.critic_hyperparam, 1 - agent.critic_hyperparam)
        elif agent.actor_objective == "hard_adv":
            weights = jnp.where(adv >= (-0.01), 1, 0)
        elif agent.actor_objective == "exp_adv":
            weights = jnp.exp(adv * agent.policy_temperature)
            weights = jnp.minimum(weights, 100) # clip weights
        elif agent.actor_objective == "bc":
            weights = jnp.ones(adv.shape)
        else:
            raise ValueError(f'Invalid actor objective: {agent.actor_objective}')

        def actor_loss_fn(
                score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            eps_pred = agent.score_model.apply_fn({'params': score_model_params},
                                       batch['observations'],
                                       noisy_actions,
                                       time,
                                       rngs={'dropout': key},
                                       training=True)
            
            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis = -1) * weights).mean()

            return actor_loss, {'actor_loss': actor_loss, 'weights' : weights.mean()}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)

        agent = agent.replace(score_model=score_model)

        target_score_params = optax.incremental_update(
            score_model.params, agent.target_score_model.params, agent.actor_tau
        )

        target_score_model = agent.target_score_model.replace(params=target_score_params)

        new_agent = agent.replace(score_model=score_model, target_score_model=target_score_model, rng=rng)

        return new_agent, info

    def eval_actions(self, observations: jnp.ndarray):
        rng = self.rng

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis = 0).repeat(self.N, axis = 0)

        score_params = self.target_score_model.params
        actions, rng = ddpm_sampler(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)
        rng, key = jax.random.split(rng, 2)
        qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations, actions)
        idx = jnp.argmax(qs)
        action = actions[idx]
        new_rng = rng

        return np.array(action.squeeze()), self.replace(rng=new_rng)
    
    def sample_implicit_policy(self, observations: jnp.ndarray):
        rng = self.rng

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis = 0).repeat(self.N, axis = 0)

        score_params = self.target_score_model.params
        actions, rng = ddpm_sampler(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)
        rng, key = jax.random.split(rng, 2)
        qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations, actions)
        vs = compute_v(self.value.apply_fn, self.value.params, observations)
        adv = qs - vs

        if self.critic_objective == 'expectile':
            tau_weights = jnp.where(adv > 0, self.critic_hyperparam, 1 - self.critic_hyperparam)
            sample_idx = jax.random.choice(key, self.N, p = tau_weights/tau_weights.sum())
            action = actions[sample_idx]
        elif self.critic_objective == 'quantile':
            tau_weights = jnp.where(adv > 0, self.critic_hyperparam, 1 - self.critic_hyperparam)
            tau_weights = tau_weights / adv
            sample_idx = jax.random.choice(key, self.N, p = tau_weights/tau_weights.sum())
            action = actions[sample_idx]
        elif self.critic_objective == 'exponential':
            weights = self.critic_hyperparam * jnp.abs(adv * self.critic_hyperparam)/jnp.abs(adv)
            sample_idx = jax.random.choice(key, self.N, p = weights)
            action = actions[sample_idx]
        else:
            raise ValueError(f'Invalid critic objective: {self.critic_objective}')

        new_rng = rng

        return np.array(action.squeeze()), self.replace(rng=new_rng)
    
    def actor_loss_no_grad(agent, batch: DatasetDict):
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)
        time = jax.random.randint(key, (batch['actions'].shape[0], ), 0, agent.T)
        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(
            key, (batch['actions'].shape[0], agent.act_dim))
        
        alpha_hats = agent.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng, 2)

        def actor_loss_fn(score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            eps_pred = agent.score_model.apply_fn({'params': score_model_params},
                                       batch['observations'],
                                       noisy_actions,
                                       time,
                                       training=False)
            
            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis = -1)).mean()

            return actor_loss, {'actor_loss': actor_loss}

        _, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        new_agent = agent.replace(rng=rng)

        return new_agent, info
    
    @jax.jit
    def actor_update(self, batch: DatasetDict):
        new_agent = self
        new_agent, actor_info = new_agent.update_actor(batch)
        return new_agent, actor_info
    
    @jax.jit
    def eval_loss(self, batch: DatasetDict):
        new_agent = self
        new_agent, actor_info = new_agent.actor_loss_no_grad(batch)
        return new_agent, actor_info
    
    @jax.jit
    def critic_update(self, batch: DatasetDict):
        def slice(x):
            return x[:256]

        new_agent = self
        
        mini_batch = jax.tree_util.tree_map(slice, batch)
        new_agent, critic_info = new_agent.update_v(mini_batch)
        new_agent, value_info = new_agent.update_q(mini_batch)

        return new_agent, {**critic_info, **value_info}

    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self
        batch_size = batch['observations'].shape[0]

        def first_half(x):
            return x[:batch_size]
        
        def second_half(x):
            return x[batch_size:]
        
        first_batch = jax.tree_util.tree_map(first_half, batch)
        second_batch = jax.tree_util.tree_map(second_half, batch)

        new_agent, _ = new_agent.update_actor(first_batch)
        new_agent, actor_info = new_agent.update_actor(second_batch)

        def slice(x):
            return x[:256]
        
        mini_batch = jax.tree_util.tree_map(slice, batch)
        new_agent, critic_info = new_agent.update_v(mini_batch)
        new_agent, value_info = new_agent.update_q(mini_batch)

        return new_agent, {**actor_info, **critic_info, **value_info}