#!/bin/bash
#SBATCH --job-name=arcgym_test
#SBATCH --account=def-arashmoh_gpu
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/gkianfar/scratch/Amin/Robotic/outputs/%x-%j.out
#SBATCH --output=/home/gkianfar/scratch/Amin/Robotic/outputs/%x-%j.err

# ---------- paths ----------
ROOT=/home/gkianfar/scratch/Amin/Robotic
REPO=$ROOT/Robo
VENV=$ROOT/venv
mkdir -p $ROOT/outputs

# ---------- environment ----------
module load python/3.10
source $VENV/bin/activate

# manylinux shim (the cluster Python disables manylinux tags)
export PYTHONPATH=$ROOT/shim:$PYTHONPATH

# Isaac Sim licence / telemetry prompts must be answered non-interactively
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

# keep Isaac Sim caches out of $HOME (small quota); safe to remove if it causes trouble
export XDG_CACHE_HOME=$ROOT/.cache
export XDG_DATA_HOME=$ROOT/.local_share
mkdir -p $XDG_CACHE_HOME $XDG_DATA_HOME

cd $REPO

echo "=== NODE / GPU"
hostname
date
nvidia-smi

echo "=== PYTHON CHECK"
python - <<'EOF'
import torch, numpy
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("numpy", numpy.__version__)
EOF

echo "=== SCRIPT HELP"
timeout 600 python scripts/arc_env_rl.py --help
echo "HELP EXIT CODE: $?"

echo "=== TEST 1: 1 env, headless, no training"
timeout 900 python scripts/arc_env_rl.py --num_envs 1 --headless
RC1=$?
echo "TEST 1 EXIT CODE: $RC1"

# Only try the camera flag if the plain run failed (H100 has no RT cores, so rendering may fail)
if [ $RC1 -ne 0 ]; then
    echo "=== TEST 2: 1 env, headless, --enable_cameras"
    timeout 900 python scripts/arc_env_rl.py --num_envs 1 --headless --enable_cameras
    echo "TEST 2 EXIT CODE: $?"
fi

# Short training run only if test 1 passed
if [ $RC1 -eq 0 ]; then
    echo "=== TEST 3: 3 envs, short training"
    timeout 1200 python scripts/arc_env_rl.py --num_envs 3 --train --headless
    echo "TEST 3 EXIT CODE: $?"
fi

echo "=== DONE"
date
