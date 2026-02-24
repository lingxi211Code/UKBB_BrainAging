import os
import argparse
import omegaconf
import torch
from utils import *
from dataset import *
import torch.optim as optim
import inspect
from datetime import datetime
import monai
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR



import warnings
warnings.filterwarnings('ignore')



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--epochs', type=int, default=30)
    
    torch.backends.cudnn.enabled = False
    
    # --------------- change here -------------------------#
    yaml_file_path = './config/deseNet.yaml'
    current_date = datetime.now().strftime('%Y-%m-%d')
    cp_saving_name = yaml_file_path.split("/")[-1].split(".")[0] + "_" + current_date
    print(cp_saving_name)
    # --------------- change ending -----------------------#

    config = omegaconf.OmegaConf.load(yaml_file_path)
    
    
    config_type_ls = ['model','train','data']
    for ct in config_type_ls:
        model_config = config[ct]
        for params in model_config:
            for key, value in params.items():
                if isinstance(value, bool):
                    parser.add_argument(f"--{key}", type=bool, default=value, help=f"Set {key}")
                elif isinstance(value, int):
                    parser.add_argument(f"--{key}", type=int, default=value, help=f"Set {key}")
                elif isinstance(value, float):
                    parser.add_argument(f"--{key}", type=float,default=value,help=f"Set {key}")
                elif isinstance(value, str):
                    parser.add_argument(f"--{key}", type=str, default=value,help=f"Set {key}")
                elif isinstance(value, omegaconf.listconfig.ListConfig):
                    parser.add_argument(f"--{key}", nargs='+',default=value,help=f"List of {key}")
    
    # Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "sgd"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.001 ,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    # Training settring
    parser.add_argument('--epochs', type=int, default=130, metavar='N',
                    help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 0.01)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=20, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit')
    
    opt = parser.parse_args()
    args = vars(opt)
    print(args)
    model =  monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args["device_ids"])
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")
    model = torch.nn.DataParallel(model, device_ids=args["device_ids"])
    
    ds_train,ds_val,ds_test,dl_train,dl_val,dl_test = create_dataset(args)
    print(f"Training set size: {len(ds_train)}")
    print(f"Validation set size: {len(ds_val)}")
    print(f"Test set size: {len(ds_test)}")


    total_epochs = args["epochs"]
    optimizer = torch.optim.Adam(model.parameters(),lr=args["lr"],weight_decay=args["weight_decay"])
    best_mae = 10.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args["lr_decay_priod"], gamma=args["lr_decay_gamma"])

    
    for epoch in range(args["epochs"]):
        train_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=dl_train,
                                    device=device,
                                    epoch=epoch,
                                    distribution=args["distribution"],
                                    corr=args["corr"])
 
        corr, mae = evaluate(model=model,
                             data_loader=dl_val,
                             device=device,
                             epoch=epoch,
                             distribution=args["distribution"],
                             corr=args["corr"])
    
        scheduler.step(mae)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args['epochs']}, Current LR: {current_lr}")

        if mae < best_mae:
            torch.save(model.state_dict(), "./weights/{}-{}.pth".format(cp_saving_name,epoch))
            best_mae = mae
            
    
    
    
    
    
