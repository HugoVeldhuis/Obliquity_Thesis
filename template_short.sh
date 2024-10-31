#!/bin/sh

 # Request nodes from the “all” partition # How many nodes to ask for
#SBATCH --partition=all
#SBATCH --nodes 1
#SBATCH --ntasks 1 # Number of tasks (MPI processes)
#SBATCH --cpus-per-task 1 # Number of logical CPUS (threads) per task
#SBATCH --time 0-3:00:00 
#SBATCH --mem 16gb 
#SBATCH -J myjob

module purge
module load anaconda3/2021-05
conda activate xpsi_py3

module load intel/PSXE/2019u4 # in case the program was compiled with the intel compiler 

./Small_fit_test.py # run your program from the directory you submitted your job from
