#!/bin/bash

#SBATCH --job-name=concentration_C_0001_Si_4667
#SBATCH --account=auburn-mins
##SBATCH -p medium
##SBATCH --account=scinet
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jose.cortes2@mavs.uta.edu
#SBATCH --nodes=1
#SBATCH --exclusive

#SBATCH --cpus-per-task=48

##SBATH -p priority --qos=nsdl
#SBATCH --time=200:00:00

atlas=true
line=concentration_C_0001_Si_4667
n_tasks=96

echo ########
echo 'Running' $line
echo ########
echo ''
date

input_file=input/$line.inp

outp=output/outp/$line.outp
runtpe=output/runtpe/$line.runtpe
mctal=output/mctal/$line.mctal
ptrac=output/ptrac/$line.ptrac


rm $outp $mctal $runtpe #$ptrac
echo 'Starting' $line

if [ $atlas = true ]; then
    module load apptainer 
    apptainer exec --pem-path=/home/jose.cortes/.ssh/mkey-pub.pem /apps/licensed/mcnp/mcnp-encrypted.sif mcnp6 r i=$input_file o=$outp mctal=$mctal ru=$runtpe notek tasks $n_tasks
else
    mcnp6 r i=$input_file o=$outp mctal=$mctal ru=$runtpe notek tasks $n_tasks
fi

echo ''
date
echo ''