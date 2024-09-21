#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=20:00:00
#SBATCH --account=m4746
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=sleeper
#SBATCH --output=out/sleeper.out
#SBATCH --constraint=cpu
#SBATCH --reservation=cdr_atlas2

for i in $(seq 1 24);
do
    echo ${i}
    sleep 3600
done



####SBATCH  --exclusive
###