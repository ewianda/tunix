# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backbone-agnostic OmicsLM adapter using composition + ContextVar.

Single affine projector (Wv + b) matching the OmicsLM paper.
No dynamic normalization — data must be pre-normalized in the pipeline.
"""

import contextvars
import dataclasses
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

_omics_ctx: contextvars.ContextVar = contextvars.ContextVar(
    'omics_ctx', default=None
)


@dataclasses.dataclass(frozen=True)
class OmicsConfig:
  omics_dim: int = 20006
  omics_token_id: int = -1
  embed_dim: int = -1
  neftune_alpha: float = 0.0
  debug: bool = False
  dtype: jnp.dtype = jnp.bfloat16
  param_dtype: jnp.dtype = jnp.bfloat16


class OmicsEmbedderAdapter(nnx.Module):
  """Wraps any Tunix embedder to inject omics at <omics> token positions.

  Single affine projection (Wv + b), matching the paper.
  Data must be pre-normalized before reaching this adapter.
  """

  def __init__(
      self,
      base_embedder: nnx.Module,
      projector: nnx.Linear,
      omics_config: OmicsConfig,
  ):
    self.base_embedder = base_embedder
    self.projector = projector
    self.omics_config = omics_config

    if hasattr(base_embedder, 'input_embedding'):
      self.input_embedding = base_embedder.input_embedding
    if hasattr(base_embedder, 'shd_config'):
      self.shd_config = base_embedder.shd_config
    if hasattr(base_embedder, 'dtype'):
      self.dtype = base_embedder.dtype

  def encode(self, tokens, *args, **kwargs):
    x = self.base_embedder.encode(tokens, *args, **kwargs)

    # NEFTune: add uniform noise to embeddings (paper uses α=5)
    neftune_alpha = self.omics_config.neftune_alpha
    if neftune_alpha > 0:
      seq_len = x.shape[1]
      embed_dim = x.shape[2]
      noise_scale = neftune_alpha / jnp.sqrt(jnp.float32(seq_len * embed_dim))
      token_hash = jnp.sum(tokens).astype(jnp.int32)
      key = jax.random.fold_in(jax.random.PRNGKey(42), token_hash)
      noise = jax.random.uniform(
          key, x.shape, dtype=x.dtype,
          minval=-noise_scale, maxval=noise_scale,
      )
      x = x + noise

    omics = _omics_ctx.get()
    if omics is not None:
      omics = jnp.nan_to_num(omics, nan=0.0, posinf=0.0, neginf=0.0)

      if self.omics_config.embed_dim != x.shape[-1]:
        raise ValueError(
            f'Projector embed_dim={self.omics_config.embed_dim}, '
            f'but base embedder dim={x.shape[-1]}'
        )

      # Check for signal (skip all-zero vectors)
      omics_energy = jnp.sum(omics ** 2, axis=-1, keepdims=True)
      has_signal = omics_energy > 1e-6

      # Single affine projection — data must be pre-normalized
      omics = omics.astype(self.omics_config.dtype)
      omics_emb = self.projector(omics)

      # Merge at placeholder positions (only for samples with signal)
      any_signal = jnp.any(has_signal, axis=1)  # (B, 1)
      mask = (tokens == self.omics_config.omics_token_id) & any_signal

      if self.omics_config.debug:
        jax.debug.print(
            'omics_inject: placeholder_count={} omics_shape={} '
            'embed_dim={} omics_emb_absmax={}',
            jnp.sum(mask, axis=-1),
            omics.shape,
            x.shape[-1],
            jnp.max(jnp.abs(omics_emb.astype(jnp.float32))),
        )

      def _merge(text_e, omics_e, m):
        n = omics_e.shape[0]
        indices = jnp.cumsum(m.astype(jnp.int32), axis=-1) - 1
        indices = jnp.clip(indices, 0, n - 1)
        return jnp.where(m[:, None], omics_e[indices], text_e)

      x = jax.vmap(_merge)(x, omics_emb, mask)

    return x

  def decode(self, x):
    return self.base_embedder.decode(x)

  def encode_vision(self, *args, **kwargs):
    if hasattr(self.base_embedder, 'encode_vision'):
      return self.base_embedder.encode_vision(*args, **kwargs)
    raise AttributeError('Base embedder does not support encode_vision')

  def encode_per_layer_input(self, *args, **kwargs):
    if hasattr(self.base_embedder, 'encode_per_layer_input'):
      return self.base_embedder.encode_per_layer_input(*args, **kwargs)
    raise AttributeError('Base embedder does not support encode_per_layer_input')


class OmicsAgnosticLM(nnx.Module):
  """Backbone-agnostic multimodal LLM with single-layer omics projector."""

  def __init__(
      self,
      base_model: nnx.Module,
      projector_state: dict[str, jnp.ndarray],
      omics_config: OmicsConfig,
  ):
    self.base_model = base_model
    self.omics_config = omics_config

    def _zeros(key, shape, dtype):
      return jnp.zeros(shape, dtype)

    self.projector = nnx.Linear(
        omics_config.omics_dim, omics_config.embed_dim,
        dtype=omics_config.dtype, param_dtype=omics_config.param_dtype,
        rngs=nnx.Rngs(0), kernel_init=_zeros, bias_init=_zeros,
    )
    self.projector.kernel = nnx.Param(projector_state['kernel'])
    self.projector.bias = nnx.Param(projector_state['bias'])

    self.base_model.embedder = OmicsEmbedderAdapter(
        self.base_model.embedder,
        self.projector,
        omics_config,
    )

  def __call__(self, tokens, *args, omics_vectors=None, **kwargs):
    kwargs.pop('omics_vectors', None)
    token = _omics_ctx.set(omics_vectors)
    try:
      return self.base_model(tokens, *args, **kwargs)
    finally:
      _omics_ctx.reset(token)


def load_omics_model(
    model_path: str,
    base_model_loader,
    base_config: Any,
    omics_config: OmicsConfig,
    mesh: jax.sharding.Mesh,
    dtype: jnp.dtype = jnp.bfloat16,
) -> OmicsAgnosticLM:
  """Load any Tunix backbone and wrap with single-layer omics projector."""
  base_model = base_model_loader(model_path, base_config, mesh, dtype=dtype)

  shd = base_config.shd_config

  @jax.jit
  def init_projector(key):
    k = jax.random.normal(
        key, (omics_config.omics_dim, omics_config.embed_dim), dtype=dtype
    ) * jnp.sqrt(1.0 / omics_config.omics_dim).astype(dtype) * 0.01
    b = jnp.zeros((omics_config.embed_dim,), dtype=dtype)
    k = jax.lax.with_sharding_constraint(
        k, NamedSharding(mesh, PartitionSpec(*shd.ffw_weight_df)))
    b = jax.lax.with_sharding_constraint(
        b, NamedSharding(mesh, PartitionSpec(*shd.rms_norm_weight)))
    return {'kernel': k, 'bias': b}

  projector_state = init_projector(jax.random.PRNGKey(42))
  return OmicsAgnosticLM(base_model, projector_state, omics_config)
