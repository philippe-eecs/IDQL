# jaxrl5

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/ikostrikov/jaxrl5/tree/main.svg?style=svg&circle-token=668374ebe0f27c7ee70edbdfbbd1dd928725c01a)](https://dl.circleci.com/status-badge/redirect/gh/ikostrikov/jaxrl5/tree/main) [![codecov](https://codecov.io/gh/ikostrikov/jaxrl5/branch/main/graph/badge.svg?token=Q5QMIDZNZ3)](https://codecov.io/gh/ikostrikov/jaxrl5)

## Installation

Run
```bash
pip install --upgrade pip

pip install -e .
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

See instructions for other versions of CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

## Examples

[Here.](examples/)

## Tests

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= pytest tests
```
