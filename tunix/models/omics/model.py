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

"""Compatibility helpers for omics-conditioned model loading.

Prefer the native Qwen3 omics path in `tunix.models.qwen3.model`.
This module keeps a lightweight compatibility layer for older examples.
"""

import dataclasses
import inspect
from typing import Any

from absl import logging
from flax import nnx
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class OmicsConfig:
  omics_dim: int = 20541
  omics_token_id: int = -1
  embed_dim: int = -1
  neftune_alpha: float = 0.0
  debug: bool = False
  dtype: jnp.dtype = jnp.bfloat16
  param_dtype: jnp.dtype = jnp.bfloat16


class OmicsEmbedderAdapter(nnx.Module):
  """Deprecated no-op adapter kept for API compatibility."""

  def __init__(
      self,
      base_embedder: nnx.Module,
      projector: nnx.Linear | None = None,
      omics_config: OmicsConfig | None = None,
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
    kwargs.pop('omics_vectors', None)
    return self.base_embedder.encode(tokens, *args, **kwargs)

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
  """Thin wrapper that forwards `omics_vectors` to the base model."""

  def __init__(
      self,
      base_model: nnx.Module,
      omics_config: OmicsConfig,
  ):
    self.base_model = base_model
    self.omics_config = omics_config

  def __call__(self, tokens, *args, omics_vectors=None, **kwargs):
    kwargs.pop('omics_vectors', None)
    return self.base_model(
        tokens, *args, omics_vectors=omics_vectors, **kwargs
    )


def load_omics_model(
    model_path: str,
    base_model_loader,
    base_config: Any,
    omics_config: OmicsConfig,
    mesh: Any,
    dtype: jnp.dtype = jnp.bfloat16,
) -> OmicsAgnosticLM:
  """Load a model and add native omics support.

  Loads the base model WITHOUT omics_dim (so safetensors loading
  succeeds), then sets omics config and creates the projector.
  """
  import dataclasses
  from flax import nnx
  import jax

  if omics_config.neftune_alpha:
    logging.warning(
        'OmicsConfig.neftune_alpha is ignored in compatibility loader; '
        'configure noise in the training pipeline instead.'
    )

  # Load base model without omics (avoids safetensors tree mismatch)
  base_model = base_model_loader(model_path, base_config, mesh, dtype=dtype)

  # Now set omics config and create projector on the loaded model
  if hasattr(base_config, 'omics_dim'):
    base_config = dataclasses.replace(
        base_config,
        omics_dim=omics_config.omics_dim,
        omics_token_placeholder=omics_config.omics_token_id,
    )
    base_model.config = base_config

    # Import the projector class from the model module
    projector_cls = type(base_model).mro()[0]
    model_module = type(base_model).__module__
    import importlib
    mod = importlib.import_module(model_module)
    if hasattr(mod, 'OmicsProjector'):
      base_model.omics_projector = mod.OmicsProjector(
          omics_dim=omics_config.omics_dim,
          embed_dim=base_config.embed_dim,
          rngs=nnx.Rngs(params=42),
          dtype=base_config.dtype,
          param_dtype=base_config.param_dtype,
          shd_config=base_config.shd_config,
      )

  model_params = inspect.signature(base_model.__call__).parameters
  has_var_keyword = any(
      p.kind == inspect.Parameter.VAR_KEYWORD for p in model_params.values()
  )
  if 'omics_vectors' not in model_params and not has_var_keyword:
    raise ValueError(
        f'Base model {type(base_model).__name__!r} does not support '
        '`omics_vectors` in __call__. Use a model with native omics support.'
    )
  return OmicsAgnosticLM(base_model, omics_config)
