#!/bin/sh
#$ -N SIR_test
#$ -cwd
#$ -e ../logs/$JOB_ID_errors.txt
#$ -o ../logs/$JOB_ID_outputs.txt
#$ -M andrew-pensoneault@uiowa.edu
#$ -m e
module load python/2.7.14_openmpi-2.1.2
cd ..
cpunum=$1
jsonfile=$2
mpiexec -n $cpunum python -m mpi4py asynchdist_assim_parallel.py $jsonfile
