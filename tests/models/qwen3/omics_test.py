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

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tunix.models.qwen3 import model as qwen3_model
from tunix.models.qwen3 import omics as omics_lib


class _FakeTokenizer:

  def __init__(self):
    self._vocab = {
        '<pad>': 0,
        'B': 1,
        'RCA': 2,
        '1': 3,
        'TP': 4,
        '53': 5,
    }
    self._encode_map = {
        'BRCA1': [1, 2, 3],
        'TP53': [4, 5],
    }

  def add_tokens(self, tokens):
    for token in tokens:
      if token not in self._vocab:
        self._vocab[token] = len(self._vocab)

  def convert_tokens_to_ids(self, token):
    return self._vocab[token]

  def get_vocab(self):
    return dict(self._vocab)

  def encode(self, text, add_special_tokens=False):
    del add_special_tokens
    return self._encode_map[text]

  def __len__(self):
    return len(self._vocab)


class Qwen3OmicsTest(absltest.TestCase):

  def _tiny_config(self, **kwargs):
    base = dict(
        num_layers=2,
        vocab_size=8,
        embed_dim=6,
        hidden_dim=16,
        num_heads=2,
        head_dim=3,
        num_kv_heads=2,
        rope_theta=10000,
        norm_eps=1e-6,
        use_tied_embedding=True,
    )
    base.update(kwargs)
    return qwen3_model.ModelConfig(**base)

  def test_omics_projector_initialization(self):
    model = qwen3_model.Qwen3(
        self._tiny_config(omics_dim=4, omics_token_placeholder=7),
        rngs=nnx.Rngs(params=0),
    )
    # Single affine projector uses nnx.Linear with xavier_uniform * 0.01
    kernel = np.asarray(model.omics_projector.linear.kernel.value)
    bias = np.asarray(model.omics_projector.linear.bias.value)
    self.assertEqual(kernel.shape, (4, 6))
    self.assertTrue(np.max(np.abs(kernel)) < 0.1)
    np.testing.assert_array_equal(bias, np.zeros_like(bias))

  def test_merge_omics_embeddings_replaces_placeholder_positions(self):
    model = qwen3_model.Qwen3(
        self._tiny_config(omics_dim=4, omics_token_placeholder=7),
        rngs=nnx.Rngs(params=0),
    )
    # Set projector to identity-like mapping for first 4 dims
    model.omics_projector.linear.kernel.value = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    model.omics_projector.linear.bias.value = jnp.zeros((6,), dtype=jnp.float32)

    tokens = jnp.array([[1, 7, 2, 7]], dtype=jnp.int32)
    omics_vectors = jnp.array(
        [[[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]]],
        dtype=jnp.float32,
    )
    text_embeddings = model.embedder.encode(tokens)
    merged = model._encode_and_get_inputs(
        tokens=tokens, omics_vectors=omics_vectors
    )

    # Non-placeholder positions should be unchanged
    np.testing.assert_array_equal(
        np.asarray(merged[:, 0, :]), np.asarray(text_embeddings[:, 0, :])
    )
    np.testing.assert_array_equal(
        np.asarray(merged[:, 2, :]), np.asarray(text_embeddings[:, 2, :])
    )
    # Placeholder positions should have projected omics (identity * input)
    np.testing.assert_array_equal(
        np.asarray(merged[0, 1, :4]),
        np.asarray(jnp.array([10.0, 11.0, 12.0, 13.0])),
    )
    np.testing.assert_array_equal(
        np.asarray(merged[0, 3, :4]),
        np.asarray(jnp.array([20.0, 21.0, 22.0, 23.0])),
    )

  def test_gene_token_augmentation_initializes_from_subword_mean(self):
    tokenizer = _FakeTokenizer()
    model = qwen3_model.Qwen3(
        self._tiny_config(vocab_size=len(tokenizer)),
        rngs=nnx.Rngs(params=0),
    )
    base_embeddings = jnp.arange(
        len(tokenizer) * model.config.embed_dim, dtype=jnp.float32
    ).reshape(len(tokenizer), model.config.embed_dim)
    model.embedder.input_embedding.value = base_embeddings

    augmentation = omics_lib.augment_qwen3_with_gene_symbols(
        model=model,
        tokenizer=tokenizer,
        gene_symbols=['BRCA1', 'TP53'],
    )

    self.assertEqual(model.config.vocab_size, len(tokenizer))
    self.assertSequenceEqual(
        list(augmentation.added_symbols), ['BRCA1', 'TP53']
    )

    brca_id = augmentation.gene_token_ids['BRCA1']
    tp53_id = augmentation.gene_token_ids['TP53']
    brca_expected = np.asarray(
        jnp.mean(base_embeddings[jnp.array([1, 2, 3])], axis=0)
    )
    tp53_expected = np.asarray(
        jnp.mean(base_embeddings[jnp.array([4, 5])], axis=0)
    )
    np.testing.assert_allclose(
        np.asarray(model.embedder.input_embedding.value[brca_id]), brca_expected
    )
    np.testing.assert_allclose(
        np.asarray(model.embedder.input_embedding.value[tp53_id]), tp53_expected
    )


if __name__ == '__main__':
  absltest.main()
