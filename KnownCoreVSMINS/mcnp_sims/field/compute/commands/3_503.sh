#!/bin/bash

#SBATCH --job-name=3_503
#SBATCH --account=auburn-mins
##SBATCH -p medium
##SBATCH --account=scinet
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jose.cortes2@mavs.uta.edu
#SBATCH --nodes=1
#SBATCH --exclusive

#SBATCH --cpus-per-task=48

##SBATH -p priority --qos=nsdl
#SBATCH --time=300:00:00

atlas=false
line=3_503
n_tasks=27 #96 threads on atlas # 32 on site
continue=false

echo ########
echo 'Running' $line
echo ########
echo ''
date

input_file=input/$line.txt

outp=output/outp/$line.outp
runtpe=output/runtpe/$line.runtpe
mctal=output/mctal/$line.mctal
ptrac=output/ptrac/$line.ptrac

if [ $continue = false ]; then
    rm $outp $mctal $runtpe #$ptrac
else
    rm $outp $mctal
fi

echo 'Starting' $line

start_time=$(date +%s)

if [ $atlas = true ]; then 
    module load apptainer 
    if [ $continue = false ]; then
        apptainer exec --pem-path=/home/jose.cortes/.ssh/mkey-pub.pem /apps/licensed/mcnp/mcnp-encrypted.sif mcnp6 r i=$input_file o=$outp mctal=$mctal ru=$runtpe notek tasks $n_tasks
    else
        apptainer exec --pem-path=/home/jose.cortes/.ssh/mkey-pub.pem /apps/licensed/mcnp/mcnp-encrypted.sif mcnp6 c i=$input_file o=$outp mctal=$mctal ru=$runtpe notek tasks $n_tasks
    fi
else
    if [ $continue = false ]; then
        mcnp6 r i=$input_file o=$outp mctal=$mctal ru=$runtpe notek tasks $n_tasks
    else
        mcnp6 c i=$input_file o=$outp mctal=$mctal ru=$runtpe notek tasks $n_tasks
    fi
fi

end_time=$(date +%s)
## diagnostics
# write the simulation time to a file
echo $line $(($end_time - $start_time))