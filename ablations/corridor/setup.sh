#!/usr/bin/env bash
# Dependencies for the corridor experiment, on top of the repository environment.
# MAGAT+ needs torch_geometric; the tokenizer is compiled from source by cppimport.
set -euo pipefail
cd "$(dirname "$0")"

CUDA_TAG="${CUDA_TAG:-cu126}"

uv pip install cppimport pybind11 pyarrow numpy
uv pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
uv pip install torch_geometric

python dataset.py
python dataset_graphs.py

# HMAGAT builds its training data with its own hypergraph and expert-data
# pipeline, which depends on the rest of its benchmark suite, so it is fetched
# rather than vendored. Skip this and the other three models still run.
if [ "${WITH_HMAGAT:-0}" = "1" ]; then
    # wandb is imported by one of the upstream argument modules, and its version
    # there still needs pkg_resources, which setuptools dropped in 81.
    uv pip install pogema pogema-toolbox scipy scikit-learn pyamg tqdm wandb "setuptools<81"
    if [ ! -d third_party/upstream_hmagat ]; then
        git clone https://github.com/proroklab/hmagat.git third_party/upstream_hmagat
    fi
    git -C third_party/upstream_hmagat checkout --quiet \
        f664ae816c7f926606c2b22eef2cb8d92878acf3
    python dataset_hmagat.py
fi

echo "Corridor datasets written to data/."
