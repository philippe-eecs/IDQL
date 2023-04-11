## Online

### SAC
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --env_name=Hopper-v4 \
                --config=configs/sac_config.py \
                --notqdm
```
### DroQ
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --env_name=Hopper-v4 \
                --utd_ratio=20 \
                --start_training 5000 \
                --max_steps 300000 \
                --config=configs/droq_config.py \
                --notqdm
```
### RedQ
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --env_name=Hopper-v4 \
                --utd_ratio=20 \
                --start_training 5000 \
                --max_steps 300000 \
                --config=configs/redq_config.py \
                --notqdm
```

## Offline

###
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_offline.py --env_name=halfcheetah-expert-v2 \
                --config=configs/bc_config.py
```