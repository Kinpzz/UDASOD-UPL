#!/usr/bin/python3
#coding=utf-8

import datetime

import torchvision
from model_wrappers.base_wrapper import Base_Wrappepr
import shutil
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from apex import amp
import argparse
import os

from lib.misc import  get_config_str, setup_logger
from lib.exp_logging import  make_train_img_grid
from lib.misc import AverageMeter, ProgressMeter
import time
from datetime import datetime
from lib.misc import save_checkpoint, set_seed, get_exp_name

from pathlib import Path # python > 3.5
from lib.dataset import MixSTData
from lib.misc import get_pse_portion_list
from lib.evalualtion import get_models_name, test
from lib.pipeline_ops import update_dataset, setup_pse_test_loader, create_model

from config.defaults import get_cfg_defaults
import copy

import json
import logging

## NOTE: limit the thread number, limit cpu usage 
# torch.set_num_threads(1)
# os.environ["MKL_NUM_THREADS"] = "1" 
# os.environ["NUMEXPR_NUM_THREADS"] = "1" 
# os.environ["OMP_NUM_THREADS"] = "1"

def parse_aug():
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
    parser.add_argument('--exp_config', type=str, default='', help='exp config file')
    parser.add_argument('--extra', type=str, default='', help='exp config file')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args

def epoch2round(epoch, cfg):
    multi_steps = [cfg.SOLVER.WARMUP_EPOCH + cfg.SOLVER.EPOCH_PER_ROUND * i for i in range(cfg.SOLVER.ROUND_NUM - 1)]
    for idx in range(len(multi_steps)):
        if epoch <= multi_steps[idx]: # epoch start from 1, round start from 0
            return idx
    return len(multi_steps)

def train(train_loader, test_loaders, model_wrapper:Base_Wrappepr, cfg = None):

    global_step = cfg.global_step
    tb_writer:SummaryWriter = cfg.tb_writer
    best_mae = cfg.best_mae
    best_sm = cfg.best_sm
    # best_epoch = 0
    best_epoch_sm = 0
    best_epoch_mae = 0
    model = model_wrapper.model

    def get_next_batch(my_iter, loader, device):
        try: 
            bat = my_iter.next()
        except:
            logging.info(f"finish reading all {len(loader.dataset)} samples in dataset, reload iterator")
            my_iter = iter(loader)
            bat = my_iter.next()
        return [item.to(device) for item in bat], my_iter

    current_round = cfg.start_round
    data_iter = iter(train_loader)
    
    for epoch in range(cfg.start_epoch, cfg.epoch + 1):# epoch starts from 1
        model.train(True)
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        baselr_rec = AverageMeter('BaseLR', ':.4e')
        headlr_rec = AverageMeter('HeadLR', ':.4e')
        losses = AverageMeter('Loss', ':.4e')

        progress = ProgressMeter(
            cfg.SOLVER.ITER_PER_EPOCH,
            [batch_time, data_time, baselr_rec, baselr_rec ,losses],
            prefix="Epoch: [{}]".format(epoch)
        )
        end = time.time()
        
        for step in range(cfg.SOLVER.ITER_PER_EPOCH):

            bat, data_iter = get_next_batch(data_iter, train_loader, 'cuda')
            data_time.update(time.time() - end) # data time

            image, mask, body, detail, var = bat
            
            out_dict = model_wrapper.handle_batch(bat, global_step = global_step)
            loss = out_dict['loss']
            
            losses.update(loss)
            baselr_rec.update(model_wrapper.optimizer.param_groups[0]['lr'])
            headlr_rec.update(model_wrapper.optimizer.param_groups[1]['lr'])
            batch_time.update(time.time() - end)
            end = time.time()

            if global_step % cfg.SOLVER.IMG_RECORD_INTERVAL == 0:
                in_im = make_train_img_grid(image, un_norm=True, num_colum=1) # c,h,w
                in_gt = make_train_img_grid(mask, num_colum=1)
                out_im = make_train_img_grid(model_wrapper.output2img(out_dict['final_out']), num_colum=1)

                new_img_tensor = torch.cat([in_im, in_gt, out_im], 2)
                log_img_dir = os.path.join(cfg.savepath, 'log_img')
                if not os.path.exists(log_img_dir):
                    os.makedirs(log_img_dir)
                log_img_name = os.path.join(log_img_dir, f'{current_round}_{epoch}_{step}.png')
                logging.info(f"logging image into {log_img_name} at global iter {global_step}")
                torchvision.utils.save_image(new_img_tensor, log_img_name)
                tb_writer.add_image('tr/log_imgs', new_img_tensor, global_step)

            if step % cfg.SOLVER.PRINT_FREQ == 0:
                progress.display(step)
            
            global_step += 1


        if epoch % cfg.TEST.EVAL_INTERVAL ==  0:
            logging.info("doing evaluation on datasets")
            model.eval()
            for k in cfg.TEST.DS_EVAL_TRAIN:
                name, _ = k, test_loaders[k]
                if name in ['thur', 'hkuis'] and epoch % (3 * cfg.TEST.EVAL_INTERVAL) != 0:
                    # reduce the frequence of evaluation
                    continue
                
                # need a adapter to evaluation.test
                res = test(
                    test_loaders=test_loaders, 
                    model_wrapper=model_wrapper, 
                    ds_names=[name],
                    metrics=['mae', 'sm'], 
                    save_res=False,
                    cfg=cfg
                )
                mae = res[name]['mae']
                sm = res[name]['sm']
                if name == 'duts_te' and mae < best_mae:
                    best_mae = mae
                    torch.save(model.state_dict(), cfg.savepath+'/best_model_mae.pth')
                    best_epoch_mae = epoch
                if name == 'duts_te' and sm > best_sm:
                    best_sm = sm
                    torch.save(model.state_dict(), cfg.savepath+'/best_model_sm.pth')
                    best_epoch_sm = epoch
                logging.info(f'epoch{epoch}: {name}_mae_{mae}')
                logging.info(f'epoch{epoch}: {name}_sm_{sm}')
                tb_writer.add_scalar(f'eval_mae/{name}', mae, global_step=epoch)
                tb_writer.add_scalar(f'eval_sm/{name}', sm, global_step=epoch)

        save_checkpoint({  # order matters
            'epoch': epoch + 1,
            'cur_round': epoch2round(epoch + 1, cfg),
            'global_step': global_step,
            'state_dict': model.state_dict(),
            'best_mae': best_mae,
            'best_sm': best_sm,
            'optimizer': model_wrapper.optimizer.state_dict(),
            'scheduler': model_wrapper.scheduler.state_dict(),
            'amp': None if not cfg.SOLVER.AMP else amp.state_dict()
        },cfg.savepath)

        # update pseudo label
        if (epoch) == cfg.SOLVER.WARMUP_EPOCH or \
            ((epoch) > cfg.SOLVER.WARMUP_EPOCH and (epoch - cfg.SOLVER.WARMUP_EPOCH) % cfg.SOLVER.EPOCH_PER_ROUND == 0):
            if epoch == cfg.SOLVER.WARMUP_EPOCH:
                logging.info("warmup training done, generating pseudo label with cg4 model, start fine_tuning")
                ## save warmup model as checkpoint
                shutil.copyfile(os.path.join(cfg.savepath, 'checkpoint.pth'), os.path.join(cfg.savepath, 'warmup_checkpoint.pth'))
            else: 
                logging.info(f"fine-tuning update pseudo label in round {current_round}")
                shutil.copyfile(os.path.join(cfg.savepath, 'checkpoint.pth'), os.path.join(cfg.savepath, f'round_{current_round}_checkpoint.pth'))
            
            model.eval()
            data_iter = update_dataset(train_loader, test_loaders, model_wrapper, epoch, current_round, cfg)
            current_round += 1

    model_cpts = get_models_name(cfg.savepath)
    fin_res = {}
    out_json = os.path.join(cfg.savepath, 'fin_res.json')
    out_csv = os.path.join(cfg.savepath, 'fin_res.csv')
    for model_name in model_cpts.keys():
        model.eval()
        model.load_state_dict(model_cpts[model_name])
        logging.info(f"evaluation on model {model_name}")
        res = test(
            test_loaders, 
            model_wrapper=model_wrapper, 
            ds_names=cfg.TEST.EVAL_DATASET, 
            metrics=cfg.TEST.EVAL_METRICS, 
            save_res=False,
            cfg=cfg
        )
        fin_res[model_name] = res
    
    with open(out_json, 'a', encoding='utf-8') as f:
        json.dump(fin_res, f, ensure_ascii=False, indent=4)

    with open(out_csv, 'a') as f:
        for model_name in sorted(list(model_cpts.keys())):
            f.write(f'{model_name},')
            log_strs = []
            metrics_csv = []
            for ds in cfg.TEST.EVAL_DATASET:
                metrics = fin_res[model_name][ds]
                log_str = [','.join( f'{met}={metrics[met]}' for met in cfg.TEST.EVAL_REPORT_METRICS)]
                log_str = f'dataset {ds} --- {log_str}'
                log_strs.append(log_str)
                metrics_csv.append(','.join([str(metrics[met].round(4)) for met in cfg.TEST.EVAL_REPORT_METRICS]))
                f.write(','.join([str(metrics[m]) for m in cfg.TEST.EVAL_REPORT_METRICS]))
                f.write(',')
            
            ext = model_name
            if '_sm' in model_name:
                ext = f'{ext}_{best_epoch_sm}'
            if '_mae' in model_name:
                ext = f'{ext}_{best_epoch_mae}'
            
            logging.info(metrics_csv)
            tb_writer.add_text(f'metric/{ext}_eval_log', '\n\n'.join(log_strs), global_step=0)
            tb_writer.add_text(f'metric/{ext}_csv_res', ','.join(metrics_csv), global_step=0)
            f.write('\n')

if __name__=='__main__':
    args = parse_aug()

    cfg = get_cfg_defaults()
    set_seed(cfg.SEED)
    
    if args.exp_config != "":
        cfg.merge_from_file(args.exp_config)
    
    ## setup configuration
    cfg.merge_from_list(args.opts)
    cfg_for_dump = copy.deepcopy(cfg)
    cfg = argparse.Namespace()
    for attr_str in cfg_for_dump:
        cfg.__setattr__(attr_str, cfg_for_dump.__getattr__(attr_str))
    
    cfg.resume = args.resume

    exp_file_name = get_exp_name(cfg, args.extra)
    cfg.savepath = os.path.join(cfg.EXP_ROOT, cfg.RUNS_NAME, exp_file_name)
    cfg.tb_writer = SummaryWriter(log_dir= os.path.join(cfg.TB_ROOT, cfg.RUNS_NAME, exp_file_name))
    Path(cfg.savepath).mkdir(parents=True, exist_ok=True)

    setup_logger(cfg.savepath)

    cfg.src_portion_list, cfg.tgt_portion_list = get_pse_portion_list(
        cfg.SOLVER.PSEUDO_UPDATE_POLICY, 
        cfg.SOLVER.SRC_INIT_PORTION, 
        cfg.SOLVER.TGT_INIT_PORTION, 
        cfg.SOLVER.ROUND_NUM, 
        tgt_max=cfg.SOLVER.TGT_MAX_PORTION
    )


    ## training dataloaders
    mix_data = MixSTData(
        src_datapath= cfg.DATA.SRC_DATAPATH,
        src_file_list_path=os.path.join(cfg.DATA.SRC_DATAPATH,'train.txt'),
        tgt_datapath=cfg.DATA.TGT_DATAPATH,
        tgt_file_list_path=os.path.join(cfg.DATA.TGT_DATAPATH,'train.txt'),
        cfg=cfg,
    )

    train_loader_mix = DataLoader(
        mix_data,
        collate_fn=mix_data.collate, 
        batch_size=cfg.DATA.TRAIN_BATCH_SIZE,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        num_workers=cfg.DATA.NUM_WORKER,
    )

    ## setup test loader
    test_loaders = setup_pse_test_loader(cfg)
    

    ## setup training model 
    model_wrapper = create_model(cfg)
    model_wrapper.setup_optimizer()
    model_wrapper.setup_scheduler()
    model_wrapper.setup_tbwriter(cfg.tb_writer)
    net = model_wrapper.model # reference
    
    net.cuda()
    if cfg.SOLVER.AMP: # using amp
        net = model_wrapper.setup_amp()
    
    def load_from_cpt(cpt_path):
        checkpoint = torch.load(cpt_path)
        cfg.global_step = checkpoint['global_step']
        cfg.start_epoch = checkpoint['epoch']
        if 'cur_round' not in checkpoint: # backward 
            cfg.start_round = epoch2round(cfg.start_epoch, cfg)
        else:
            cfg.start_round = checkpoint['cur_round']
        cfg.best_mae = checkpoint['best_mae']  if 'best_mae' in checkpoint else 1.1
        cfg.best_sm = checkpoint['best_sm']  if 'best_sm' in checkpoint else 0
        if 'amp' in checkpoint and cfg.SOLVER.AMP:
            amp.load_state_dict(checkpoint['amp'])
        net.load_state_dict(checkpoint['state_dict'])
        model_wrapper.optimizer.load_state_dict(checkpoint['optimizer'])
        if model_wrapper.scheduler:
            model_wrapper.scheduler.load_state_dict(checkpoint['scheduler'])
        if cfg.start_round != 0: # if not in warmup round we need to update dataset
            net.eval()
            update_dataset(train_loader_mix, test_loaders, model_wrapper, cfg.start_epoch - 1, cfg.start_round - 1, cfg)


    if cfg.resume:
        assert args.exp_config != None, "resuming from checkpoint needs a checkpoint file"
        cpt_path = os.path.join(cfg.savepath, 'checkpoint.pth')
        logging.info(f"resume training : {cpt_path}")
        load_from_cpt(cpt_path)
    elif cfg.MODEL.DETECTOR_PATH != "": # only model state dict, no cpt
        # TODO: skip warmup, setup optmizer state and scheduler state
        logging.info(f"loading pretraining detector from {cfg.MODEL.DETECTOR_PATH}")
        model_cpt = torch.load(cfg.MODEL.DETECTOR_PATH)
        if 'state_dict' in model_cpt:
            model_cpt = model_cpt['state_dict']
        cfg.global_step = 0
        cfg.start_epoch = 1 # epoch start from 1
        cfg.best_mae = 1.1
        cfg.best_sm = 0
        cfg.start_round = 0
        net.load_state_dict(model_cpt) # load
    elif cfg.MODEL.WARMUP_PATH != "":
        cpt_path = cfg.MODEL.WARMUP_PATH
        logging.info(f"skip warmup trianing : {cpt_path}")
        load_from_cpt(cpt_path)
    else:
        cfg.global_step = 0
        cfg.start_epoch = 1 # epoch start from 1
        cfg.best_mae = 1.1
        cfg.best_sm = 0
        cfg.start_round = 0
        logging.info(f"train from scratch with resnet backbone pretrain {cfg.MODEL.BAKCBONE_PATH}")
        model_wrapper.init_model()

    ## create path
    logging.info(f'{get_config_str(cfg)}')
    cfg.tb_writer.add_text(f'config/all', get_config_str(cfg, '\n\n'), global_step=0)

    ## save config file and train file into exp_dir
    with open(os.path.join(cfg.savepath, 'config.yaml'), 'w') as f:
        f.write(cfg_for_dump.dump()) # use cfg for dump we can resotre the env 

    train(train_loader_mix,test_loaders, model_wrapper, cfg)
    
    logging.info(f'training done at {datetime.now().strftime("%Y%m%d_%H_%M_%S")}')
    logging.info(f'savepath {cfg.savepath}')
    logging.info(f'config \n {str(cfg)}')
    
