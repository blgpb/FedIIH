import os
from myparser import Parser
from datetime import datetime

from misc.utils import *
from modules.multiprocs import ParentProcess

def main(args):

    args = set_config(args)

    if args.model == 'fedavg':    
        from models.fedavg.server import Server
        from models.fedavg.client import Client
    elif args.model == 'fedhvae':
        from models.fedhvae.server import Server
        from models.fedhvae.client import Client
    else:
        print('incorrect model was given: {}'.format(args.model))
        os._exit(0)

    pp = ParentProcess(args, Server, Client)
    pp.start()

def set_config(args):

    args.base_lr = 1e-3
    args.min_lr = 1e-3
    args.momentum_opt = 0.9
    args.weight_decay = 1e-6
    args.warmup_epochs = 10
    args.base_momentum = 0.99
    args.final_momentum = 1.0

    if args.dataset == 'Cora':
        args.n_feat = 1433
        args.n_clss = 7
        args.n_clients = 10 if args.n_clients == None else args.n_clients
        if args.mode == 'disjoint': 
            args.base_lr = 0.02
            args.dropout = 0.3
            args.weight_decay = 0.005
            args.n_latentdims = 128
            args.n_layers = 4
            args.n_routit = 6
        elif args.mode == 'overlapping':
            args.base_lr = 0.01
    elif args.dataset == 'CiteSeer':
        args.n_feat = 3703
        args.n_clss = 6
        args.n_clients = 10 if args.n_clients == None else args.n_clients
        args.base_lr = 0.01 if args.lr == None else args.lr
    elif args.dataset == 'PubMed':
        args.n_feat = 500
        args.n_clss = 3
        args.n_clients = 10 if args.n_clients == None else args.n_clients
        if args.mode == 'disjoint': 
            args.base_lr = 0.01
            args.dropout = 0.25
            args.weight_decay = 0.0045
            args.n_latentdims = 256
            args.n_layers = 1
            args.n_routit = 6
        elif args.mode == 'overlapping':
            args.base_lr = 0.015
            args.dropout = 0.4
            args.weight_decay = 1e-6
            args.n_latentdims = 256
            args.n_layers = 1
            args.n_routit = 6
    elif args.dataset == 'Computers':
        args.n_feat = 767
        args.n_clss = 10
        args.n_clients = 10 if args.n_clients == None else args.n_clients
        if args.mode == 'disjoint' and args.n_clients in [10, 20]:
            args.base_lr = 0.015
            args.dropout = 0.4
            args.weight_decay = 1e-6
            args.n_latentdims = 128
            args.n_layers = 1
            args.n_routit = 6
        elif args.mode == 'disjoint' and args.n_clients == 5:
            args.base_lr = 0.015
            args.dropout = 0.35
            args.weight_decay = 1e-6
            args.n_latentdims = 128
            args.n_layers = 1
            args.n_routit = 6
        elif args.mode == 'overlapping':
            args.base_lr = 0.01
    elif args.dataset == 'Photo':
        args.n_feat = 745
        args.n_clss = 8
        args.n_clients = 10 if args.n_clients == None else args.n_clients
        if args.mode == 'disjoint':
            args.base_lr = 0.015
            args.dropout = 0.4
            args.weight_decay = 1e-6
            args.n_latentdims = 256
            args.n_layers = 1
            args.n_routit = 6
        elif args.mode == 'overlapping':
            args.base_lr = 0.01
            args.dropout = 0.35
            args.weight_decay = 1e-6
            args.n_latentdims = 128
            args.n_layers = 1
            args.n_routit = 5
    elif args.dataset == 'ogbn-arxiv':
        args.n_feat = 128
        args.n_clss = 40
        args.n_clients = 10 if args.n_clients == None else args.n_clients
        args.base_lr = 0.01 if args.lr == None else args.lr
    elif args.dataset == 'Roman-empire':
        args.n_feat = 300
        args.n_clss = 18
        args.n_clients = 10 if args.n_clients == None else args.n_clients
        if args.mode == 'disjoint':
            args.base_lr = 0.015
            args.dropout = 0.35
            args.weight_decay = 1e-6
            args.n_latentdims = 128
            args.n_layers = 1
            args.n_routit = 6
        elif args.mode == 'overlapping':
            args.base_lr = 0.015
            args.dropout = 0.35
            args.weight_decay = 1e-6
            args.n_latentdims = 128
            args.n_layers = 1
            args.n_routit = 6
    elif args.dataset == 'Amazon-ratings':
        args.n_feat = 300
        args.n_clss = 5
        args.n_clients = 10 if args.n_clients == None else args.n_clients
        if args.mode == 'disjoint':
            args.base_lr = 0.01
            args.dropout = 0.35
            args.weight_decay = 1e-6
            args.n_latentdims = 128
            args.n_layers = 3
            args.n_routit = 7
        elif args.mode == 'overlapping':
            args.base_lr = 0.01
    elif args.dataset == 'Minesweeper':
        args.n_feat = 7
        args.n_clss = 1
        args.n_clients = 10 if args.n_clients == None else args.n_clients
        args.base_lr = 0.01 if args.lr == None else args.lr
    elif args.dataset == 'Tolokers':
        args.n_feat = 10
        args.n_clss = 1
        args.n_clients = 10 if args.n_clients == None else args.n_clients
        if args.mode == 'disjoint':
            args.base_lr = 0.01
            args.dropout = 0.35
            args.weight_decay = 0.0045
            args.n_latentdims = 128
            args.n_layers = 1
            args.n_routit = 2
        elif args.mode == 'overlapping':
            args.base_lr = 0.01
            args.dropout = 0.35
            args.weight_decay = 0.0045
            args.n_latentdims = 128
            args.n_layers = 1
            args.n_routit = 2
    elif args.dataset == 'Questions':
        args.n_feat = 301
        args.n_clss = 1
        args.n_clients = 10 if args.n_clients == None else args.n_clients
        if args.mode == 'disjoint':
            args.base_lr = 0.01
            args.n_latentdims = 128
        elif args.mode == 'overlapping':
            args.base_lr = 0.01
            args.n_latentdims = 128


    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial = f'{args.dataset}_{args.mode}/clients_{args.n_clients}/{now}_{args.model}'

    args.data_path = f'{args.base_path}/datasets' 
    args.checkpt_path = f'{args.base_path}/checkpoints/{trial}'
    args.log_path = f'{args.base_path}/logs/{trial}'

    if args.debug == True:
        args.checkpt_path = f'{args.base_path}/debug/checkpoints/{trial}'
        args.log_path = f'{args.base_path}/debug/logs/{trial}'

    return args

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main(Parser().parse())










