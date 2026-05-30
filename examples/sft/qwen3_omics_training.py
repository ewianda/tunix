# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OmicsLM training using backbone-agnostic tunix.models.omics module.

Usage:
    python examples/sft/qwen3_omics_training.py \
        --data_pattern "gs://omicslm-batch-data/arrayrecord/omicslm-*.array_record" \
        --model_id Qwen/Qwen3-4B \
        --max_steps 10000
"""

import dataclasses
import glob
import os

from absl import app
from absl import flags
from flax import nnx
from google.cloud import storage
from grain import python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax import checkpoint as ocp
import tensorflow as tf
from transformers import AutoTokenizer

from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.omics import OmicsConfig, load_omics_model
from tunix.models.qwen3 import model as model_lib
from tunix.models.qwen3 import params as params_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer

OMICS_TOKEN = '<omics>'


def _gcs_glob(pattern: str) -> list[str]:
  """Glob GCS files using the storage client (no subprocess)."""
  # Parse gs://bucket/prefix/pattern
  without_gs = pattern[5:]  # remove 'gs://'
  bucket_name = without_gs.split('/')[0]
  glob_pattern = without_gs[len(bucket_name) + 1:]
  client = storage.Client()
  blobs = client.list_blobs(bucket_name, match_glob=glob_pattern)
  return [f'gs://{bucket_name}/{blob.name}' for blob in blobs]

_SEED = flags.DEFINE_integer('seed', 42, 'Random seed.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 4, 'Batch size per host.')
_MAX_SEQ_LEN = flags.DEFINE_integer('max_seq_len', 512, 'Max sequence length.')
_MAX_STEPS = flags.DEFINE_integer('max_steps', 10000, 'Training steps.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 1.5e-5, 'Peak LR.')
_WARMUP_STEPS = flags.DEFINE_integer('warmup_steps', 200, 'LR warmup steps.')
_SAVE_INTERVAL = flags.DEFINE_integer('save_interval', 2000, 'Checkpoint interval.')
_MODEL_ID = flags.DEFINE_string('model_id', 'Qwen/Qwen3-4B', 'HuggingFace model ID.')
_MODEL_CACHE = flags.DEFINE_string('model_cache', '/tmp/qwen3-4b', 'Local model cache.')
_DATA_PATTERN = flags.DEFINE_string('data_pattern', '', 'ArrayRecord GCS pattern.')
_OUTPUT_DIR = flags.DEFINE_string('output_dir',
    'gs://omicslm-batch-data/omicslm_training_v3', 'Output dir.')
_OMICS_DIM = flags.DEFINE_integer('omics_dim', 20006, 'Omics vector dimension.')
_NORM_STATS = flags.DEFINE_string('norm_stats', '/tmp/omics_norm_stats.npz',
    'Path to pre-computed normalization stats (gene_mean, global_std).')
_NEFTUNE_ALPHA = flags.DEFINE_float('neftune_alpha', 5.0, 'NEFTune noise alpha (0 to disable).')
_GRAD_ACCUM = flags.DEFINE_integer('grad_accum', 8, 'Gradient accumulation steps.')


# =============================================================================
# Data pipeline — plain iterator to avoid grain worker serialization issues
# =============================================================================

class _ParseArrayRecord(grain.MapTransform):
  def map(self, raw_record):
    example = tf.train.Example()
    example.ParseFromString(raw_record)
    f = example.features.feature
    return (
        f['prompt'].bytes_list.value[0].decode('utf-8'),
        f['completion'].bytes_list.value[0].decode('utf-8'),
        np.frombuffer(f['omics'].bytes_list.value[0], dtype=np.float32).copy(),
    )


class _TokenizeAndBuild(grain.MapTransform):
  def __init__(self, tokenizer, max_seq_len, omics_dim, gene_mean=None, global_std=1.0):
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len
    self.omics_dim = omics_dim
    self.gene_mean = gene_mean  # (20006,) or None
    self.global_std = global_std

  def map(self, element):
    prompt, completion, omics = element
    src = np.array(self.tokenizer.encode(prompt), dtype=np.int32)
    dst = np.array(
        list(self.tokenizer.encode(completion)) + [self.tokenizer.eos_id()],
        dtype=np.int32,
    )
    tokens = np.concatenate([src, dst])
    q_mask = np.zeros_like(src, dtype=np.bool_)
    a_mask = np.ones_like(dst, dtype=np.bool_)
    mask = np.concatenate([q_mask, a_mask])
    tokens = self._pad(tokens, self.tokenizer.pad_id())
    mask = self._pad(mask, False)
    if omics.ndim == 1:
      omics = omics.reshape(-1, self.omics_dim)
    omics = np.nan_to_num(omics, nan=0.0, posinf=0.0, neginf=0.0)
    # Normalize per paper: log1p → center per gene → scale by global std
    omics = np.log1p(np.abs(omics)) * np.sign(omics)
    if self.gene_mean is not None:
      omics = (omics - self.gene_mean) / max(self.global_std, 1e-6)
    return peft_trainer.TrainingInput(
        input_tokens=tokens,
        input_mask=mask,
        images=omics.astype(np.float32),
    )

  def _pad(self, x, val):
    pad = max(self.max_seq_len - len(x), 0)
    return np.pad(x[:self.max_seq_len], [[0, pad]], constant_values=val)


def load_dataset(data_pattern, tokenizer, batch_size, max_seq_len, omics_dim,
                 gene_mean=None, global_std=1.0, num_epochs=2, seed=42):
  """Load ArrayRecord dataset with grain. Uses worker_count=0 to avoid
  serialization issues with TrainingInput across process boundaries."""
  if data_pattern.startswith('gs://'):
    paths = sorted(_gcs_glob(data_pattern))
  else:
    paths = sorted(glob.glob(data_pattern))

  print(f'  Found {len(paths)} shards', flush=True)
  source = grain.ArrayRecordDataSource(paths)
  print(f'  Total records: {len(source):,}', flush=True)

  return grain.DataLoader(
      data_source=source,
      sampler=grain.IndexSampler(
          num_records=len(source), num_epochs=num_epochs,
          shard_options=grain.NoSharding(), shuffle=True, seed=seed,
      ),
      operations=[
          _ParseArrayRecord(),
          _TokenizeAndBuild(tokenizer, max_seq_len, omics_dim, gene_mean, global_std),
          grain.Batch(batch_size=batch_size, drop_remainder=True),
      ],
      worker_count=0,
  )


# =============================================================================
# Training
# =============================================================================

def train(train_ds, model, tokenizer, mesh, max_steps, output_dir, pid=0):
  """Training loop following vlm_training.py pattern."""

  def gen_model_input_fn(x):
    pad_mask = x.input_tokens != tokenizer.pad_id()
    n = x.input_tokens.shape[-1]
    positions = jnp.broadcast_to(jnp.arange(n), x.input_tokens.shape)
    attention_mask = (
        jnp.tril(jnp.ones((n, n), dtype=jnp.bool_))[None, :, :]
        & pad_mask[:, None, :]
    )
    result = {
        'input_tokens': x.input_tokens,
        'input_mask': x.input_mask,
        'positions': positions,
        'attention_mask': attention_mask,
    }
    # Map images → omics_vectors for the model's __call__
    if x.images is not None:
      result['omics_vectors'] = x.images
    return result

  ckpt_dir = os.path.join(output_dir, 'checkpoints')
  log_dir = os.path.join(output_dir, 'logs')

  lr_schedule = optax.warmup_cosine_decay_schedule(
      init_value=0.0, peak_value=_LEARNING_RATE.value,
      warmup_steps=_WARMUP_STEPS.value, decay_steps=max_steps, end_value=0.0,
  )
  optimizer = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.95,
                  weight_decay=0.01),
  )

  training_config = peft_trainer.TrainingConfig(
      eval_every_n_steps=100000,
      max_steps=max_steps,
      gradient_accumulation_steps=_GRAD_ACCUM.value,
      checkpoint_root_directory=ckpt_dir,
      checkpointing_options=ocp.CheckpointManagerOptions(
          save_interval_steps=_SAVE_INTERVAL.value, max_to_keep=3,
          enable_async_checkpointing=True,
      ),
      metrics_logging_options=metrics_logger.MetricsLoggerOptions(
          log_dir=log_dir, flush_every_n_steps=50,
      ),
      data_sharding_axis=('fsdp',),
      pbar_description='OmicsLM Training',
  )

  trainer = peft_trainer.PeftTrainer(
      model, optimizer, training_config,
  ).with_gen_model_input_fn(gen_model_input_fn)

  with jax.set_mesh(mesh):
    trainer.train(train_ds)


# =============================================================================
# Main
# =============================================================================

def main(argv):
  del argv

  coordinator = os.environ.get('JAX_COORDINATOR_ADDRESS')
  if coordinator:
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=int(os.environ['JAX_NUM_PROCESSES']),
        process_id=int(os.environ['JAX_PROCESS_ID']),
    )

  pid = jax.process_index()
  n_devices = jax.device_count()
  n_local = len(jax.local_devices())
  if pid == 0:
    print(f'Devices: {n_devices} ({n_local} local)', flush=True)

  mesh = jax.make_mesh(
      (n_devices, 1), ('fsdp', 'tp'),
      axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
  )

  # Tokenizer
  from huggingface_hub import snapshot_download
  if pid == 0:
    print('Downloading model...', flush=True)
  model_path = snapshot_download(
      repo_id=_MODEL_ID.value, local_dir=_MODEL_CACHE.value,
      local_dir_use_symlinks=False, ignore_patterns=['*.pth'],
  )
  hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
  hf_tokenizer.pad_token = hf_tokenizer.pad_token or hf_tokenizer.eos_token
  if OMICS_TOKEN not in hf_tokenizer.get_vocab():
    hf_tokenizer.add_tokens([OMICS_TOKEN])
  tokenizer = tokenizer_lib.TokenizerAdapter(hf_tokenizer)
  omics_token_id = hf_tokenizer.get_vocab()[OMICS_TOKEN]
  # Verify <omics> encodes to a single token with the expected ID
  omics_ids = hf_tokenizer.encode(OMICS_TOKEN, add_special_tokens=False)
  if pid == 0:
    print(f'Omics token ID: {omics_token_id}', flush=True)
    print(f'Omics encode test: "{OMICS_TOKEN}" -> {omics_ids}', flush=True)
    print(f'Vocab size: {len(hf_tokenizer)}', flush=True)
    print(f'Embed dim (base_config): will be set after config load', flush=True)
  if len(omics_ids) != 1 or omics_ids[0] != omics_token_id:
    raise ValueError(
        f'<omics> token mismatch! encode("{OMICS_TOKEN}")={omics_ids}, '
        f'but vocab lookup={omics_token_id}. The tokenizer is not encoding '
        f'<omics> as a single token.'
    )

  # Barrier: ensure all hosts have tokenizer ready before proceeding
  if coordinator:
    jax.experimental.multihost_utils.sync_global_devices('tokenizer_ready')

  # Load normalization stats
  gene_mean, global_std = None, 1.0
  if os.path.exists(_NORM_STATS.value):
    stats = np.load(_NORM_STATS.value)
    gene_mean = stats['gene_mean']
    global_std = float(stats['global_std'])
    if pid == 0:
      print(f'Norm stats: gene_mean shape={gene_mean.shape}, global_std={global_std:.4f}', flush=True)
  else:
    if pid == 0:
      print('WARNING: no norm stats found, using raw log1p values', flush=True)

  # Data
  if pid == 0:
    print('Loading data...', flush=True)
  train_ds = load_dataset(
      _DATA_PATTERN.value, tokenizer,
      _BATCH_SIZE.value * n_local, _MAX_SEQ_LEN.value,
      _OMICS_DIM.value, gene_mean=gene_mean, global_std=global_std,
      seed=_SEED.value + pid,
  )

  # Model config
  config_name = _MODEL_ID.value.split('/')[-1].lower().replace('-', '_')
  if pid == 0:
    print(f'Config: {config_name}', flush=True)
  base_config = getattr(model_lib.ModelConfig, config_name)()
  base_config = dataclasses.replace(
      base_config, remat_config=model_lib.RematConfig.BLOCK,
  )

  omics_config = OmicsConfig(
      omics_dim=_OMICS_DIM.value,
      omics_token_id=omics_token_id,
      embed_dim=base_config.embed_dim,
      neftune_alpha=_NEFTUNE_ALPHA.value,
      debug=True,
  )

  # Load model using backbone-agnostic loader
  with jax.set_mesh(mesh):
    if pid == 0:
      print('Loading model...', flush=True)
    model = load_omics_model(
        model_path=model_path,
        base_model_loader=params_lib.create_model_from_safe_tensors,
        base_config=base_config,
        omics_config=omics_config,
        mesh=mesh,
    )
    if pid == 0:
      print(f'Model: {type(model).__name__}', flush=True)
      print(f'Embedder: {type(model.base_model.embedder).__name__}', flush=True)
      # Verify projector is in the param tree
      params = nnx.state(model, nnx.Param)
      param_shapes = jax.tree.map(lambda x: x.shape, params)
      projector_found = False
      for path, shape in jax.tree_util.tree_leaves_with_path(param_shapes):
        path_str = '/'.join(str(p) for p in path)
        if 'projector' in path_str.lower():
          print(f'  Projector param: {path_str} = {shape}', flush=True)
          projector_found = True
      if not projector_found:
        raise ValueError('Projector params NOT found in model param tree!')
      print(f'  Omics config: dim={omics_config.omics_dim}, '
            f'embed_dim={omics_config.embed_dim}, '
            f'token_id={omics_config.omics_token_id}', flush=True)

  # Sanity check: verify first batch has <omics> tokens and omics data
  if pid == 0:
    print('Running data sanity check...', flush=True)
    for first_batch in train_ds:
      tokens = np.asarray(first_batch.input_tokens)
      images = np.asarray(first_batch.images)
      omics_count = np.sum(tokens == omics_token_id, axis=-1)
      print(f'  First batch tokens shape: {tokens.shape}', flush=True)
      print(f'  First batch omics shape: {images.shape}', flush=True)
      print(f'  <omics> tokens per sample: {omics_count}', flush=True)
      print(f'  Omics L2 per sample: {np.sqrt(np.sum(images**2, axis=-1)).flatten()}', flush=True)
      if np.all(omics_count == 0):
        raise ValueError(
            f'No <omics> tokens found in first batch! '
            f'Token ID {omics_token_id} not present in input_tokens. '
            f'Check that prompts contain "{OMICS_TOKEN}" and the tokenizer '
            f'encodes it as a single token.'
        )
      break
    # Re-create dataset since we consumed one batch
    train_ds = load_dataset(
        _DATA_PATTERN.value, tokenizer,
        _BATCH_SIZE.value * n_local, _MAX_SEQ_LEN.value,
        _OMICS_DIM.value, gene_mean=gene_mean, global_std=global_std,
        seed=_SEED.value + pid,
    )

  # Train
  if pid == 0:
    print(f'Training: {_MAX_STEPS.value} steps', flush=True)
  train(train_ds, model, tokenizer, mesh, _MAX_STEPS.value, _OUTPUT_DIR.value, pid=pid)

  if pid == 0:
    print('Training complete.', flush=True)


if __name__ == '__main__':
  app.run(main)
