from os.path import dirname, join
import os
import torch
from random import choice
import sys



chrs = [chr(a) for a in range(ord('a'),ord('z')+1)]
chrs.extend([chr(a) for a in range(ord('A'),ord('Z')+1)])
rid = choice(chrs)
chrs.extend([chr(a) for a in range(ord('1'),ord('9')+1)])
for _ in range(6):
    rid += choice(chrs)

sd = {
    'config': {
        'lr': 5e-5,
        'n_epochs': 20,
        'epoch': 0,
        'arch': 'h-enformer-2',
        'sequence_length': 196608,
        'debug': '-d' in sys.argv,
        'id': rid,
    },
}

torch.save(sd,join(dirname(__file__),'checkpoints',rid+'.pt'))

if '-d' in sys.argv:
    cmd = f'source job.sh {rid}'
else:
    cmd = f'qsub -P aclab -o logs/{rid}.log -e logs/{rid}.log -l gpus=1 -N {rid} -pe omp 32 -l gpu_c=7.0 -l h_rt=11:00:00 job.sh {rid}'
print(cmd)
os.system(cmd)
