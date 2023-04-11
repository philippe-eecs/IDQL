from typing import Callable, Optional, Sequence
import flax.linen as nn
import jax.numpy as jnp
import flax

default_init = nn.initializers.xavier_uniform

def get_weight_decay_mask(params):
    flattened_params = flax.traverse_util.flatten_dict(
        flax.core.frozen_dict.unfreeze(params))

    def decay(k, v):
        if any([(key == 'bias' or 'Input' in key or 'Output' in key)
                for key in k]):
            return False
        else:
            return True

    return flax.core.frozen_dict.freeze(
        flax.traverse_util.unflatten_dict(
            {k: decay(k, v)
             for k, v in flattened_params.items()}))

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                x = self.activations(x)
        return x

class MLPResNetV3Block(nn.Module):
    """MLPResNet block."""
    features: int
    act: Callable
    dropout_rate: float = None
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x, training: bool = False):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(
                x, deterministic=not training)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.features * 4)(x)
        x = self.act(x)
        x = nn.Dense(self.features)(x)

        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)

        return residual + x

class MLPResNetV3(nn.Module):
    """MLPResNetV2."""
    num_blocks: int
    out_dim: int
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 256
    activations: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        for _ in range(self.num_blocks):
            x = MLPResNetV3Block(self.hidden_dim, act=self.activations, use_layer_norm=self.use_layer_norm, dropout_rate=self.dropout_rate)(x, training=training)
            
        x = self.activations(x)
        x = nn.Dense(self.out_dim, kernel_init=default_init())(x)
        return x

class MlpBlock(nn.Module):

    @nn.compact
    def __call__(self, x):
        dim = x.shape[-1]

        x = nn.Dense(dim)(x)
        x = nn.swish(x)
        output = nn.Dense(dim)(x)
        return output
        
class EncoderBlock(nn.Module):

    @nn.compact
    def __call__(self, inputs, cond):
        x = inputs + cond
        y = nn.LayerNorm()(x)
        y = MlpBlock()(y)

        return x + y

class Encoder(nn.Module):
    num_blocks: int = 3

    @nn.compact
    def __call__(self, x, cond):

        for _ in range(self.num_blocks):
            x = EncoderBlock()(x, cond)

        return x
