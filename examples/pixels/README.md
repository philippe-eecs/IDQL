## Online

### DrQ
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl python train_online_pixels.py \
                --env_name=cheetah-run-v0 \
                --config=configs/drq_config.py
```

## Offline

### Collect data
```bash
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online_pixels.py \
                --env_name=cheetah-run-v0 \
                --save_dir=chckpnts
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python collect_from_checkpoint.py \
                --env_name=cheetah-run-v0 \
                --load_dir=chckpnts --save_dir=buffers
```

### PixelBC
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl python train_offline_pixels.py \
                --env_name=cheetah-run-v0 \
                --config=configs/pixel_bc_config.py \
                --load_dir=buffers
```