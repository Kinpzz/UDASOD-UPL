from model_wrappers.base_wrapper import Base_Wrappepr
import os
import torch
from model_wrappers.ldf_wrapper import LDF_Wrapper
from lib.dataset import TestData

from tqdm import tqdm
import argparse

from lib.evalualtion import test
import logging
from config.defaults import get_cfg_defaults
from lib.misc import get_time_str, setup_logger
from lib.pipeline_ops import setup_test_loader
import copy

"""
CUDA_VISIBLE_DEVICES=2 python test.py \
    --exp_config 'config/test.yaml'
"""

def parse_aug():
    parser = argparse.ArgumentParser(description='DA_USOD')
    parser.add_argument('--exp_config', type=str, default='', help='cofing file')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # prediction saving path
    parser.add_argument('--save_res', action='store_true', help='whether to store output saliency map')
    # parser.add_argument('--save_body_path', type=str, default='', help='result path')
    # parser.add_argument('--save_detail_path', type=str, default='', help='exp result path')
    # parser.add_argument('--save_mask_path', type=str, default='', help='exp result path')
    
    args = parser.parse_args()
    return args


def evaluate(model_wrapper:Base_Wrappepr, model_name, cfg):
    net = model_wrapper.model
    net.train(False)

    # test_loaders = {ds: test_config[ds]['loader'] for ds in ds_names}
    test_loaders = setup_test_loader(cfg)
    ds_names = cfg.TEST.EVAL_DATASET
    fin_res = test(
        test_loaders, 
        model_wrapper=model_wrapper, 
        ds_names=ds_names, 
        metrics=cfg.TEST.EVAL_METRICS, 
        save_res=cfg.save_res, 
        cfg=cfg
    )

    # header line
    def get_csv_header_line(datasets, metrics):
        """
        args:
        - 
        return:
        - head_lines: two header line
        """
        headerline1 = [''] + [ds for ds in datasets for _ in metrics]
        headerline2 = [''] + [met for _ in datasets for met in metrics]
        return headerline1, headerline2

    headerlines = get_csv_header_line(cfg.TEST.EVAL_DATASET,cfg.TEST.EVAL_REPORT_METRICS)
    with open(cfg.out_csv_res, 'a') as f:
        f.write('\n'.join([','.join(hl) for hl in headerlines]))
        f.write('\n')
        f.write(f'{model_name},')
        for ds in ds_names:
            f.write(','.join([str(fin_res[ds][m]) for m in cfg.TEST.EVAL_REPORT_METRICS]))
            f.write(',')
        f.write('\n')


if __name__ == "__main__":

    args = parse_aug()

    cfg = get_cfg_defaults()
    
    if args.exp_config != "":
        cfg.merge_from_file(args.exp_config)
    
    # merge from command line
    cfg.merge_from_list(args.opts)

    cfg_for_dump = copy.deepcopy(cfg)
    cfg = args
    for attr_str in cfg_for_dump:
        cfg.__setattr__(attr_str, cfg_for_dump.__getattr__(attr_str))
    
    cfg.savepath = f'test_res_{get_time_str()}'

    if not os.path.exists(cfg.savepath):
        os.makedirs(cfg.savepath)
    
    ## config file
    with open(os.path.join(cfg.savepath, 'test_config.yaml'), 'w') as f:
        f.write(cfg_for_dump.dump()) # use cfg for dump we can resotre the env 
    
    cfg.out_csv_res = os.path.join(cfg.savepath, 'eval.csv')

    setup_logger(cfg.savepath, 'test')

    model_wrapper = LDF_Wrapper(cfg)
    net = model_wrapper.model
    net.cuda()

    ## start evaluation
    if cfg.TEST.MODEL_ROOTS != '':
        model_lists = [e for e in os.listdir(cfg.TEST.MODEL_ROOTS) if e.endswith('.pth')]
        logging.info(model_lists)
        save_path = cfg.savepath
        for model_name in model_lists:
            ## every model have its own savepath
            cfg.savepath = os.path.join(save_path, model_name)
            cfg.save_mask_path = os.path.join(cfg.savepath, 'mask')
            model_path = os.path.join(cfg.TEST.MODEL_ROOTS, model_name)
            logging.info(f"evaluate model : {model_name} in {model_path}")
            cpt = torch.load(model_path) 
            if 'state_dict' in cpt:
                cpt = cpt['state_dict']
            net.load_state_dict(cpt)
            evaluate(model_wrapper,model_name, cfg)
    elif cfg.TEST.CHECKPOINT !=  "": # evaluate only one model, checkpoint file
        model_name = cfg.TEST.CHECKPOINT.split('/')[-1]
        cfg.save_mask_path = os.path.join(cfg.savepath, 'mask')
        state_dict = torch.load(cfg.TEST.CHECKPOINT)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        net.load_state_dict(state_dict)
        evaluate(model_wrapper, model_name, cfg)