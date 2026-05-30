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

"""Minimal OmicsLM-style Qwen3 training/inference example."""

import jax
import jax.numpy as jnp
from flax import nnx
from tunix.models.qwen3 import model as qwen3_model
from tunix.models.qwen3 import omics as omics_lib
from tunix.sft import peft_trainer


class _ToyTokenizer:
  """Tiny tokenizer-like object with dynamic vocab APIs."""

  def __init__(self):
    self._vocab = {'<pad>': 0, '<omics>': 1, 'gene': 2, 'state': 3}
    self._encode_map = {'BRCA1': [2, 3]}

  def get_vocab(self):
    return dict(self._vocab)

  def add_tokens(self, tokens):
    for token in tokens:
      if token not in self._vocab:
        self._vocab[token] = len(self._vocab)

  def convert_tokens_to_ids(self, token):
    return self._vocab[token]

  def encode(self, text, add_special_tokens=False):
    del add_special_tokens
    return self._encode_map.get(text, [self._vocab['gene']])

  def __len__(self):
    return len(self._vocab)


def main():
  tokenizer = _ToyTokenizer()
  omics_token_id = omics_lib.ensure_token(
      tokenizer, omics_lib.DEFAULT_OMICS_TOKEN
  )
  model_config = qwen3_model.ModelConfig(
      num_layers=2,
      vocab_size=len(tokenizer),
      embed_dim=16,
      hidden_dim=32,
      num_heads=2,
      head_dim=8,
      num_kv_heads=2,
      rope_theta=10000,
      norm_eps=1e-6,
      use_tied_embedding=True,
      omics_dim=omics_lib.DEFAULT_OMICS_DIM,
      omics_token_placeholder=omics_token_id,
  )
  model = qwen3_model.Qwen3(model_config, rngs=nnx.Rngs(params=0))

  # Optional gene-aware vocabulary augmentation.
  omics_lib.augment_qwen3_with_gene_symbols(
      model=model, tokenizer=tokenizer, gene_symbols=['BRCA1']
  )

  # Prompt with 2 interleaved omics slots: "gene <omics> state <omics>".
  input_tokens = jnp.array(
      [[2, omics_token_id, 3, omics_token_id]], dtype=jnp.int32
  )
  input_mask = jnp.array([[1, 1, 1, 1]], dtype=jnp.int32)
  positions = jnp.arange(input_tokens.shape[1], dtype=jnp.int32)[None, :]
  attention_mask = jnp.ones(
      (1, input_tokens.shape[1], input_tokens.shape[1]), dtype=jnp.bool_
  )
  omics_vectors = jnp.ones(
      (1, 2, omics_lib.DEFAULT_OMICS_DIM), dtype=jnp.float32
  )

  loss = peft_trainer._default_loss_fn(
      model=model,
      input_tokens=input_tokens,
      input_mask=input_mask,
      positions=positions,
      attention_mask=attention_mask,
      omics_vectors=omics_vectors,
  )
  print('loss:', float(jax.device_get(loss)))


if __name__ == '__main__':
  main()
