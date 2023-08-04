#!/bin/bash
#SBATCH --job-name=diffusion_jax
#SBATCH --open-mode=append
#SBATCH --output=logs/out/%x_%j.txt
#SBATCH --error=logs/err/%x_%j.txt
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:TITAN:1
#SBATCH --account=co_rail
#SBATCH --partition=savio3_gpu
#SBATCH --qos=rail_gpu3_normal


ID=$((SLURM_ARRAY_TASK_ID-1))

module load gnu-parallel

run_singularity ()
{
    formalid=$(( $2 * 4 + $1 ))
    singularity exec --nv --userns --writable-tmpfs -B /usr/lib64 -B /var/lib/dcv-gl --overlay $SCRATCH/singularity/overlay-50G-10M.ext3:ro $SCRATCH/singularity/cuda11.4-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "
    source ~/.bashrc
    XLA_PYTHON_CLIENT_PREALLOCATE=false python ../launcher/iql/train_iql_bc_gaussian.py \
    --variant $formalid
    "
}
export -f run_singularity

parallel --delay 20 --linebuffer -j 4 run_singularity {} ::: 0 1 2 3 ::: $ID