echo $1
qsub -P aclab -o logs/$1.log -e logs/$1.log -N $1 -l gpus=1 -pe omp 32 -l gpu_c=7.0 -l h_rt=11:00:00 job.sh $1
