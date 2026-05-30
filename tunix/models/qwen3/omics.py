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

"""Utilities for Qwen3 OmicsLM-style token/embedding augmentation."""

import dataclasses
from typing import Any, Sequence

import jax.numpy as jnp
from tunix.models.qwen3 import model as qwen3_model


DEFAULT_OMICS_DIM = 20541
DEFAULT_OMICS_TOKEN = '<omics>'


@dataclasses.dataclass(frozen=True)
class GeneTokenAugmentation:
  """Result of adding gene-symbol tokens to a tokenizer.

  `gene_token_ids` only contains symbols newly added by this call.
  """

  base_vocab_size: int
  added_symbols: tuple[str, ...]
  gene_token_ids: dict[str, int]
  decomposition_token_ids: dict[str, tuple[int, ...]]


def _unwrap_tokenizer(tokenizer: Any) -> Any:
  return getattr(tokenizer, 'tokenizer', tokenizer)


def _check_tokenizer(tokenizer: Any) -> None:
  required = (
      'add_tokens',
      'convert_tokens_to_ids',
      'get_vocab',
      'encode',
      '__len__',
  )
  missing = [name for name in required if not hasattr(tokenizer, name)]
  if missing:
    raise ValueError(
        'Tokenizer must support HuggingFace-like dynamic vocabulary APIs. '
        f'Missing methods: {missing}'
    )


def ensure_token(tokenizer: Any, token: str) -> int:
  """Ensures a token exists in tokenizer vocabulary and returns its id.

  Args:
    tokenizer: A tokenizer or tokenizer adapter with dynamic-vocab APIs.
    token: Token text to ensure in the vocabulary.

  Returns:
    Integer token id for `token`.

  Side effects:
    Mutates `tokenizer` by adding `token` when it does not already exist.
  """
  tokenizer = _unwrap_tokenizer(tokenizer)
  _check_tokenizer(tokenizer)
  vocab = tokenizer.get_vocab()
  if token not in vocab:
    tokenizer.add_tokens([token])
  return int(tokenizer.convert_tokens_to_ids(token))


def add_gene_tokens(
    tokenizer: Any, gene_symbols: Sequence[str]
) -> GeneTokenAugmentation:
  """Extends tokenizer with gene symbols and stores original subword IDs."""
  tokenizer = _unwrap_tokenizer(tokenizer)
  _check_tokenizer(tokenizer)

  vocab = tokenizer.get_vocab()
  base_vocab_size = len(tokenizer)
  added_symbols = []
  decomposition_token_ids = {}

  for symbol in gene_symbols:
    if symbol in vocab:
      continue
    token_ids = tuple(
        int(i)
        for i in tokenizer.encode(symbol, add_special_tokens=False)
        if i is not None
    )
    if not token_ids:
      continue
    decomposition_token_ids[symbol] = token_ids
    added_symbols.append(symbol)

  if added_symbols:
    tokenizer.add_tokens(added_symbols)

  # Only include newly added symbols in gene_token_ids to keep it consistent
  # with decomposition_token_ids. Pre-existing vocab tokens were not
  # re-initialized, so including them would mix "freshly added, mean-initialized"
  # with "already existed, untouched" tokens.
  gene_token_ids = {
      symbol: int(tokenizer.convert_tokens_to_ids(symbol))
      for symbol in added_symbols
  }

  return GeneTokenAugmentation(
      base_vocab_size=base_vocab_size,
      added_symbols=tuple(added_symbols),
      gene_token_ids=gene_token_ids,
      decomposition_token_ids=decomposition_token_ids,
  )


def resize_qwen3_token_embeddings(
    model: qwen3_model.Qwen3, new_vocab_size: int
) -> None:
  """Resizes Qwen3 embeddings/lm_head in-place when vocab grows.

  Args:
    model: Qwen3 model instance to mutate.
    new_vocab_size: Target vocabulary size.

  Side effects:
    Extends `model.embedder.input_embedding` and, when untied, `model.lm_head`.
    No-op when `new_vocab_size` is not greater than current vocab size.
  """
  old_vocab_size = model.embedder.input_embedding.value.shape[0]
  if new_vocab_size <= old_vocab_size:
    return
  delta = new_vocab_size - old_vocab_size
  embed = model.embedder.input_embedding.value
  embed_pad = jnp.zeros((delta, embed.shape[1]), dtype=embed.dtype)
  model.embedder.input_embedding.value = jnp.concatenate(
      [embed, embed_pad], axis=0
  )
  if not model.config.use_tied_embedding and hasattr(model, 'lm_head'):
    lm_head = model.lm_head.w.value
    lm_head_pad = jnp.zeros((lm_head.shape[0], delta), dtype=lm_head.dtype)
    model.lm_head.w.value = jnp.concatenate([lm_head, lm_head_pad], axis=1)
  model.config.vocab_size = new_vocab_size


def initialize_gene_token_embeddings(
    model: qwen3_model.Qwen3, augmentation: GeneTokenAugmentation
) -> None:
  """Initializes gene-token embeddings from mean subword embeddings.

  Args:
    model: Qwen3 model instance to mutate.
    augmentation: Output metadata from `add_gene_tokens`.

  Side effects:
    Updates newly added gene-token rows in input embeddings in-place. For untied
    models, also updates corresponding lm_head columns.
  """
  embed = model.embedder.input_embedding.value
  base_embed = embed[: augmentation.base_vocab_size]
  for symbol, token_ids in augmentation.decomposition_token_ids.items():
    target_id = augmentation.gene_token_ids.get(symbol)
    if target_id is None:
      continue
    mean_embed = jnp.mean(
        base_embed[jnp.asarray(token_ids, dtype=jnp.int32)], axis=0
    )
    embed = embed.at[target_id].set(mean_embed)

  model.embedder.input_embedding.value = embed
  if not model.config.use_tied_embedding and hasattr(model, 'lm_head'):
    lm_head = model.lm_head.w.value
    for symbol, token_ids in augmentation.decomposition_token_ids.items():
      target_id = augmentation.gene_token_ids.get(symbol)
      if target_id is None:
        continue
      mean_embed = jnp.mean(
          lm_head[:, jnp.asarray(token_ids, dtype=jnp.int32)],
          axis=1,
      )
      lm_head = lm_head.at[:, target_id].set(mean_embed)
    model.lm_head.w.value = lm_head


def augment_qwen3_with_gene_symbols(
    *,
    model: qwen3_model.Qwen3,
    tokenizer: Any,
    gene_symbols: Sequence[str],
) -> GeneTokenAugmentation:
  """Adds gene tokens to tokenizer and initializes their trainable embeddings."""
  augmentation = add_gene_tokens(tokenizer=tokenizer, gene_symbols=gene_symbols)
  tokenizer = _unwrap_tokenizer(tokenizer)
  resize_qwen3_token_embeddings(model=model, new_vocab_size=len(tokenizer))
  initialize_gene_token_embeddings(model=model, augmentation=augmentation)
  return augmentation
