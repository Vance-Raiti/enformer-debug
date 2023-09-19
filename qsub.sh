echo $@
qsub -P aclab -o job.log -e job.log -l gpus=1 -pe omp 32 -l gpu_c=7.0 -l h_rt=11:00:00 job.sh $@
