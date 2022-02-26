#coding=utf-8
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import math
import logging
import tqdm

def str_to_list(aug_str:str):
    return [e.strip() for e in aug_str[1:-1].split(',')]


def get_pse_portion_list(pseudo_update_policy, src_init, tgt_init, run_num, tgt_max=0.6):
    """
    ret:
    - src_portion_list
    - tgt_portion_list
    """
    if pseudo_update_policy == 'vanilla_pl':
        tgt_list = [1] * run_num
        src_list = [1] * run_num
        return src_list, tgt_list
    elif pseudo_update_policy == 'no_src':
        tgt_list = [min(0.6, tgt_init * math.pow(2, i)) for i in range(run_num - 1)] + [1]
        src_list = [0] * run_num
        return src_list, tgt_list
    elif pseudo_update_policy == 'no_select':
        tgt_list = [1] * run_num
        src_list = [src_init * math.pow(0.5, i) for i in range(run_num - 1)] + [0]
        return src_list, tgt_list
    elif pseudo_update_policy == 'src_half':
        src_list = [src_init * math.pow(0.5, i) for i in range(run_num - 1)] + [0]
        tgt_list = [min(tgt_max, tgt_init * math.pow(2, i)) for i in range(run_num - 1)] + [1]
        return src_list, tgt_list
    elif pseudo_update_policy == 'more_tgt':
        src_list = [src_init * math.pow(0.5, i) for i in range(run_num - 1)] + [0]
        tgt_list = [min(0.8, tgt_init * math.pow(1.5, i)) for i in range(run_num - 1)] + [1]
        return src_list, tgt_list
    elif pseudo_update_policy == 'debug_mae':
        tgt_list = [tgt_init, 0.3, 0.5, 1]
        src_list = [src_init, 0.1, 0.05, 0]
        return src_list, tgt_list
    else:
        raise Exception("invalid selection policy")


def get_scheduler(optimizer, cfg):
    tot_steps = cfg.tot_steps
    if cfg.SOLVER.SCHEDULER == 'lin_epoch':
        scheduler = get_linear_epoch_scheduler(
            optimizer, 
            num_training_steps=tot_steps,
            step_per_epoch=cfg.SOLVER.ITER_PER_EPOCH
        )
    elif cfg.SOLVER.SCHEDULER == 'cyc':
        multi_steps = [cfg.SOLVER.WARMUP_EPOCH + cfg.SOLVER.EPOCH_PER_ROUND * i for i in range(cfg.SOLVER.ROUND_NUM - 1)]
        logging.info(f"iterative scheme : {multi_steps} \n total epoch {cfg.epoch}")
        scheduler = get_multi_cyc_scheduler(
            optimizer=optimizer,
            num_training_steps=tot_steps,
            multi_steps = [e * cfg.SOLVER.ITER_PER_EPOCH for e in multi_steps],
            scale_rate=cfg.SOLVER.LR_DECAY_RATE,
        )
    elif cfg.SOLVER.SCHEDULER == 'multi_cos':
        multi_steps = [cfg.SOLVER.WARMUP_EPOCH + cfg.SOLVER.EPOCH_PER_ROUND * i for i in range(cfg.SOLVER.ROUND_NUM - 1)]
        logging.info(f"iterative cosine scheme : {multi_steps} \n total epoch {cfg.epoch}")
        scheduler = get_multi_cos_scheduler(
            optimizer=optimizer,
            num_training_steps=tot_steps,
            multi_steps = [e * cfg.SOLVER.ITER_PER_EPOCH for e in multi_steps],
            scale_rate=cfg.SOLVER.LR_DECAY_RATE,
        )
    elif cfg.SOLVER.SCHEDULER == 'cos':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0,
            num_training_steps=tot_steps
        )
    elif cfg.SOLVER.SCHEDULER == 'multi_step':
        multi_steps = [cfg.SOLVER.WARMUP_EPOCH + cfg.SOLVER.EPOCH_PER_ROUND * i for i in range(cfg.SOLVER.ROUND_NUM - 1)]
        logging.info(f"iterative scheme : {multi_steps} \n total epoch {cfg.epoch}")
        scheduler = get_multi_step_scheduler(
            optimizer=optimizer,
            multi_steps = [e * cfg.SOLVER.ITER_PER_EPOCH for e in multi_steps],
            scale_rate=cfg.SOLVER.LR_DECAY_RATE,
        )
    return scheduler


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16., # 1/2 -> 0
                                    last_epoch=-1):
    # current function have two group
    def _lr_lambda(current_step):
        
        if current_step < num_warmup_steps: # when warmup, go down first
            return float(current_step) / float(max(1, num_warmup_steps))
        
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        
        return max(0., math.cos(math.pi * num_cycles * no_progress))
    base_func = lambda cur_step: _lr_lambda(cur_step)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, base_func, last_epoch)


def get_linear_scheduler(optimizer,
                        num_training_steps,
                        last_epoch=-1):
    # current function have two group
    def _lr_lambda(current_step):
        return (1 - abs((current_step + 1) / (num_training_steps + 1) * 2 - 1))
    base_func = lambda cur_step: _lr_lambda(cur_step)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, base_func, last_epoch)


def get_linear_epoch_scheduler(optimizer,
                        num_training_steps,
                        step_per_epoch,
                        last_epoch=-1):
    # current function have two group
    tot_epoch = num_training_steps / step_per_epoch
    def _lr_lambda(current_step):
        cur_epoch = current_step // step_per_epoch
        return (1 - abs((cur_epoch + 1) / (tot_epoch + 1) * 2 - 1))
    base_func = lambda cur_step: _lr_lambda(cur_step)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, base_func, last_epoch)


def get_multi_cos_scheduler(optimizer,
                        num_training_steps,
                        multi_steps,
                        scale_rate, 
                        num_cycles=7./16.,
                        last_epoch=-1):
    # current function have two group
    def _lr_lambda(current_step): 
        r = len(multi_steps)
        for i, s in enumerate(multi_steps):
            if current_step < s:
                r = i
                break
        cur_cyc_start = 0 if r == 0 else multi_steps[r - 1]
        cur_cyc_end = num_training_steps if r == len(multi_steps) else multi_steps[r]
        cur_cyc_step = current_step - cur_cyc_start
        cur_cyc_tot = cur_cyc_end - cur_cyc_start
        no_progress = float(cur_cyc_step) / float(max(1, cur_cyc_tot))
        return math.pow(scale_rate, r) * max(0., math.cos(math.pi * num_cycles * no_progress))
    base_func = lambda cur_step: _lr_lambda(cur_step)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, base_func, last_epoch)


def get_multi_cyc_scheduler(optimizer,
                        num_training_steps,
                        multi_steps,
                        scale_rate, 
                        last_epoch=-1):
    # current function have two group
    def _lr_lambda(current_step): 
        r = len(multi_steps)
        for i, s in enumerate(multi_steps):
            if current_step < s:
                r = i
                break
        cur_cyc_start = 0 if r == 0 else multi_steps[r - 1]
        cur_cyc_end = num_training_steps if r == len(multi_steps) else multi_steps[r]
        cur_cyc_step = current_step - cur_cyc_start
        cur_cyc_tot = cur_cyc_end - cur_cyc_start
        return math.pow(scale_rate, r) * (1 - abs((cur_cyc_step + 1) / (cur_cyc_tot + 1) * 2 - 1))
    base_func = lambda cur_step: _lr_lambda(cur_step)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, base_func, last_epoch)

def get_multi_step_scheduler(optimizer,
                        multi_steps,
                        scale_rate, 
                        last_epoch=-1):
    # current function have two group
    def _lr_lambda(current_step): 
        r = len(multi_steps)
        for i, s in enumerate(multi_steps):
            if current_step < s:
                r = i
                break
        return math.pow(scale_rate, r)
    func = lambda cur_step: _lr_lambda(cur_step)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, func, last_epoch)

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None

def get_config_str(args, sep='\n'):
    res_strs = []
    args = vars(args)
    for tup in args.items():
        res_strs.append(f'{tup[0]} : {tup[1]}')
    return sep.join(res_strs)

def get_exp_name(cfg, extra = ''):
    kv = {
        'lr':  cfg.SOLVER.LR if 'BASE_LR' not in cfg.SOLVER  else f'h{cfg.SOLVER.LR}b{cfg.SOLVER.BASE_LR}',
        'bn': cfg.DATA.TRAIN_BATCH_SIZE,
        'eps': f'{cfg.SOLVER.WARMUP_EPOCH}-{cfg.SOLVER.ROUND_NUM - 1}x{cfg.SOLVER.EPOCH_PER_ROUND}',
        'pse': cfg.SOLVER.PSEUDO_UPDATE_POLICY,
        'psrc': cfg.SOLVER.SRC_INIT_PORTION,
        'ptgt': f'{cfg.SOLVER.TGT_INIT_PORTION}t{cfg.SOLVER.TGT_MAX_PORTION}' if cfg.SOLVER.PSE_POLICY == 'portion' else f'var-{cfg.SOLVER.VAR_THRESHOLD}',
        'pfil': cfg.SOLVER.PSE_FILTER_PORTION,
        'sche': f'{cfg.SOLVER.SCHEDULER}-{cfg.SOLVER.LR_DECAY_RATE}',
        'lbsec': f'{cfg.SOLVER.AUG_TYPE}-{cfg.SOLVER.FLIP_NUM}-{cfg.SOLVER.FDA_NUM}-{cfg.SOLVER.RAND_SCALE_NUM}',
        'aug':  's' if cfg.DATA.STRONG_AUG else 'w',
        'oh': 't' if cfg.SOLVER.ONE_HOT else 'f',
        'pw': f'{cfg.SOLVER.PIXEL_WEIGHT}',
        'seed': cfg.SEED,
    }
    res = '_'.join([f'{k}={v}' for k, v in kv.items()])
    if extra:
        res += f"_{extra}"
    return res


def setup_tensorboard_writer(tb_base_dir, runs_name, comment, **kargs):
    from torch.utils.tensorboard import SummaryWriter
    tf_file_name = '_'.join(k + str(v) for k, v in kargs.items()) + comment
    tf_file_path = os.path.join(tb_base_dir, runs_name, tf_file_name)
    return SummaryWriter(
        log_dir=tf_file_path
    ), tf_file_name

def setup_logger(savepath, prefix_name = 'train'):
    logging.basicConfig(
        format='%(asctime)s {%(filename)s:%(lineno)d} - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(savepath, f'{prefix_name}_{get_time_str()}.log')),
            logging.StreamHandler()
        ],
        level=logging.INFO
    )
    # logging.info("start training! ")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def vis_pse_statis_line(name2metric, s_name2metric, name2gt_mae, fig_name):
    ys = [(name2metric[tup[0]], name2gt_mae[tup[0]]) for tup in s_name2metric]
    ents = [item[0] for item in ys]
    maes = [item[1] for item in ys]
    x = list(range(1, len(ys) + 1))
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(x, ents, color='olive')
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(x, maes, color='red')
    ax2.bar(x, maes, color='red', width=1)
    ax2 = fig.add_subplot(3, 1, 3)
    maes = np.array(maes)
    maes_cdf = [] 
    for i in range(1, len(maes) + 1):
        maes_cdf.append(np.mean(maes[:i]))
    # mean cdf
    ax2.plot(x, maes_cdf, color='blue')
    fig.suptitle(fig_name)
    return fig

def vis_pse_statis_scatter(name2metric, s_name2metric, name2gt_mae):
    ys = [(name2metric[tup[0]], name2gt_mae[tup[0]]) for tup in s_name2metric]
    ents = [item[0] for item in ys]
    maes = [item[1] for item in ys]
    x = list(range(1, len(ys) + 1))
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.scatter(x, ents, s=0.1, color='olive')
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.scatter(x, maes, s=0.1, color='red')
    ax2 = fig.add_subplot(3, 1, 3)
    maes = np.array(maes)
    maes_cdf = [] 
    for i in range(1, len(maes) + 1):
        maes_cdf.append(np.mean(maes[:i]))
    # mean cdf
    ax2.scatter(x, maes_cdf, color='blue')
    return fig


def save_checkpoint(state, checkpoint, filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.benchmark = False # will this faster
    # torch.backends.cudnn.deterministic = True # torch 1.6
    # torch.set_deterministic(True) # torch 1.7 or higher
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)


def ratio_str(ds, rs):
    res_str = []
    for i, d in enumerate(ds):
        if 'DUTS' in d:
            res_str.append(f'{rs[i]}d')
        elif 'MSRA' in d:
            res_str.append(f'{rs[i]}m')
    return '-'.join(res_str)

def get_time_str():
    from datetime import datetime
    now = datetime.now() # current date and time
    date_time = now.strftime("%m-%d-%H-%M-%S")
    # print("date and time:",date_time)
    return date_time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval_exp_pse(pse_dir, cfg):
    pse_train_file = os.path.join(pse_dir, 'train.txt')
    gt_dir = os.path.join(cfg.DATA.DATAROOT, 'DUTS')

    pse_name_list = [l.strip() for l in open(pse_train_file)]
    name2gt_mask = {k: f'{os.path.join(gt_dir, "mask", k)}.png' for k in pse_name_list}
    name2pse_path = {k: f'{os.path.join(pse_dir, "mask", k)}.png' for k in pse_name_list}
    
    res_mae = []
    # maybe adaptive threshold
    for pse_name in tqdm.tqdm(pse_name_list): 
        pse = cv2.imread(name2pse_path[pse_name], 0) / 255
        pse = (pse > 0.5).astype(np.float32) # binarization
        gt = cv2.imread(name2gt_mask[pse_name], 0) / 255
        mae = np.mean(np.abs(pse - gt))
        res_mae.append(mae)
    ret = np.mean(np.array(res_mae))
    logging.info(f'pseudo label {pse_dir} ')
    logging.info(f'number:{len(pse_name_list)}, mae :{ret}')
    return ret
