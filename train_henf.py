import torch
from basenji import BasenjiDataset
import wandb

import os
from os.path import exists, join,dirname
import sys
from random import choice

import henf
from enformer.metrics import MeanPearsonCorrCoefPerChannel as MPCCPC
from torch.optim.lr_scheduler import LambdaLR

from torch import _dynamo
_dynamo.config.suppress_errors = True

from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()

Enformer = henf.Enformer

ename = dirname(__file__)
checkpoints = join(ename,'checkpoints')

rid = None
from_checkpoint = '-id' in sys.argv
epoch = 3
lr = 5e-5
for i,arg in enumerate(sys.argv):
    if arg == '-lr':
        lr = float(sys.argv[i+1])
    if arg == '-id':
        rid = sys.argv[i+1]
    if arg == '-ep':
        epoch = int(sys.argv[i+1])-1

if rid is None:
    chrs = [chr(a) for a in range(ord('a'),ord('z')+1)]
    chrs.extend([chr(a) for a in range(ord('A'),ord('Z')+1)])
    rid = choice(chrs)
    chrs.extend([chr(a) for a in range(ord('1'),ord('9')+1)])
    for _ in range(6):
        rid += choice(chrs)

if not exists(checkpoints):
    os.mkdir(checkpoints)
checkpoint = join(checkpoints,rid+'.pt')

n_epochs = 10
n_train = 34000
n_valid = 2000
model = Enformer.from_hparams(
    output_heads = dict(human = 5313),
)

loss = torch.nn.PoissonNLLLoss(log_input=False)

mpccpc = MPCCPC(n_channels = 5313).to('cuda')

train_data, valid_data = [
    torch.utils.data.DataLoader(
        BasenjiDataset(organism='human',split=split),
        batch_size=1,
        num_workers=0,
    )
    for split in ['train','valid']
]
if '-w' in sys.argv:
    os.environ['WANDB_MODE'] = 'disabled'
else:
    os.environ['WANDB_SILENT']='true'
    os.environ['WANDB_CONSOLE']='off'


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=4e-2,
)

class linear_warmup_cosine_decay(LambdaLR):
    def __init__(
            self,
            warmup: int,
            N: int,
            **kwargs,
        ):
        def lwcd(n):
            if n < warmup:
                return n/warmup
            theta = math.pi*(warmup-n)/(N-warmup)
            return 0.5*math.cos(theta)+0.5
        super().__init__(lr_lambda=lwcd,**kwargs)

lr_schedule = linear_warmup_cosine_decay(optimizer=optimizer,warmup=int(9e4),N=34000*n_epochs)
scaler = torch.cuda.amp.GradScaler()



model = torch.compile(model)

if from_checkpoint:
    sd = torch.load(join(checkpoints,rid+'.pt'))
    model.load_state_dict(sd['model'])
    optimizer.load_state_dict(sd['optimizer'])
    lr_schedule.load_state_dict(sd['lr_schedule'])

model = model.to('cuda')

wandb.init(
    project = 'enformer-vance',
    id=rid,
)



model.train()
for it, data in enumerate(train_data):
    if it>100:
        break
    x = data['features'].to('cuda')
    y = data['targets'].to('cuda')
    lr_schedule.step()
    with torch.autocast('cuda'):
        y_hat = model(x)
        l = loss(y_hat,y)
        scaler.scale(l).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        with torch.no_grad():
            mpccpc.update(y_hat,y)
            corr_coef = mpccpc.compute().mean()
    
    log = {
        'train/correlation': corr_coef.item(),
        'train/loss': l.item(),
        'iteration': it+n_train*epoch,
        'train/lr': lr_schedule.get_last_lr()[0],
    }
    print(log)
    wandb.log(log)
mpccpc.reset()

model.eval()
for it, data in enumerate(valid_data):
    if it>100:
        break
    x = data['features'].to('cuda')
    y = data['targets'].to('cuda')
    with torch.autocast('cuda'), torch.no_grad():
        y_hat = model(x)
        l = loss(y_hat,y)
        mpccpc.update(y_hat,y)
        corr_coef = mpccpc.compute().mean()
    
    log = {
        'valid/correlation': corr_coef.item(),
        'valid/loss': l.item(),
        'iteration': it+n_valid*epoch,
    }
    print(log)
    wandb.log(log)

wandb.finish()

model = model.to('cpu')

torch.save(
    {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_schedule': lr_schedule.state_dict(),
    },
    checkpoint
)

if epoch == 0:
    print('training done!')
    exit()

args = f'-lr {lr} -ep {epoch} -id {rid}'
print(args)
cmd = f'qsub -P aclab -o job.log -e job.log -l gpus=1 -N {rid} -pe omp 32 -l gpu_c=7.0 -l h_rt=11:00:00 job.sh {args}'
os.system(cmd)
