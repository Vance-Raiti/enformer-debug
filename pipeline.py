import torch
from basenji import BasenjiDataset
import wandb
import os
from enformer.metrics import MeanPearsonCorrCoefPerChannel as MPCCPC
from enformer.modeling_enformer import Enformer
from torch.optim.lr_scheduler import LambdaLR


n_epochs = 10
n_train = 34000
n_valid = 2000
model = Enformer.from_hparams(
    output_heads = dict(human = 5313),
).to('cuda')
model = torch.compile(model)
loss = torch.nn.PoissonNLLLoss(log_input=False)

mpccpc = MPCCPC(n_channels = 5313).to('cuda')

train_data, valid_data = [
    torch.utils.data.DataLoader(
        BasenjiDataset(organism='human',split='train'),
        batch_size=1,
        num_workers=0,
    )
    for split in ['train','valid']
]

#os.environ['WANDB_MODE'] = 'disabled'
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

schedule = linear_warmup_cosine_decay(optimizer=optimizer,warmup=int(9e4),N=34000*n_epochs)

scaler = torch.cuda.amp.GradScaler()
wandb.init(
    project = 'enformer-vance',
)
for epoch in range(n_epochs):
    
    model.train()
    for it, data in enumerate(train_data):
        x = data['features'].to('cuda')
        y = data['targets'].to('cuda')
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
        }
        print(log)
        wandb.log(log)
    mpccpc.reset()

    model.eval()
    for it, data in enumerate(valid_data):
        x = data['features'].to('cuda')
        y = torch.from_numpy(data['targets']).to('cuda')
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
    mpccpc.reset()


wandb.finish()
