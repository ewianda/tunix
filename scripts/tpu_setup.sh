#!/bin/bash
set -e

VENV=/tmp/tunix_venv
REPO=/tmp/tunix
BRANCH=copilot/implement-omicslm-style-architecture
NORM_STATS=gs://omicslm-batch-data/omics_norm_stats.npz

echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== Creating venv ==="
rm -rf $VENV
uv venv $VENV --python 3.12
source $VENV/bin/activate

echo "=== Cloning repo ==="
rm -rf $REPO
git clone --depth 1 -b $BRANCH https://github.com/ewianda/tunix.git $REPO

echo "=== Installing tunix + deps ==="
cd $REPO
uv pip install -e ".[prod]"
uv pip install transformers huggingface_hub google-cloud-storage tensorflow array-record grain-nightly

echo "=== Copying norm stats ==="
gsutil cp $NORM_STATS /tmp/omics_norm_stats.npz 2>/dev/null || echo "WARN: norm stats not copied"

echo "=== Verifying ==="
python3 -c "
import jax
from tunix.models.omics import OmicsConfig
print(f'JAX OK: {jax.device_count()} devices, platform={jax.devices()[0].platform}')
print(f'OmicsConfig OK: dim={OmicsConfig().omics_dim}')
"

echo "=== Writing run script ==="
cat > /tmp/run_train.sh << 'EOF'
#!/bin/bash
source /tmp/tunix_venv/bin/activate
exec python3 -u /tmp/tunix/examples/sft/qwen3_omics_training.py \
  --data_pattern "gs://omicslm-batch-data/arrayrecord/omicslm-*.array_record" \
  --model_id Qwen/Qwen3-4B --max_steps 10000 --batch_size 4 --max_seq_len 512 \
  --learning_rate 1.5e-5 --warmup_steps 200 --grad_accum 8 \
  --neftune_alpha 5.0 --omics_dim 20006 \
  --output_dir gs://omicslm-batch-data/omicslm_training_v4
EOF
chmod +x /tmp/run_train.sh

echo "=== SETUP COMPLETE ==="
