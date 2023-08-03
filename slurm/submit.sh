#!/bin/bash

END=19
mkdir logs/out/ -p
mkdir logs/err/ -p

sbatch --array=1-${END}%16 sbatch.sh