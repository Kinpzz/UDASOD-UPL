from model_wrappers.base_wrapper import Base_Wrappepr
import torch
from lib.misc import eval_exp_pse

from lib.dataset import TestData, STDataTest
from torch.utils.data import DataLoader
from lib.pse_label_selection import it_pse_update_conf_lite
import os
import logging
from model_wrappers.ldf_wrapper import LDF_Wrapper

def setup_test_dataset_config(cfg):
    eval_config = {
        'duts_te': {'gt_path':f'{cfg.DATA.DATAROOT}/DUTS', 'list_file': 'test.txt'},
        'duts_val':{'gt_path':f'{cfg.DATA.DATAROOT}/DUTS', 'list_file': 'train_val.txt'},
        'duts_om': {'gt_path':f'{cfg.DATA.DATAROOT}/DUT-OMRON'},
        'thur': {'gt_path':f'{cfg.DATA.DATAROOT}/THUR15K'},
        'eccsd': {'gt_path':f'{cfg.DATA.DATAROOT}/ECSSD'},
        'hkuis': {'gt_path':f'{cfg.DATA.DATAROOT}/HKU-IS'},
        'sod': {'gt_path':f'{cfg.DATA.DATAROOT}/SOD'},
        'pascal_s': {'gt_path':f'{cfg.DATA.DATAROOT}/PASCAL-S'},
        'msra_b': {'gt_path':f'{cfg.DATA.DATAROOT}/MSRA-B'},
    }
    return eval_config


def setup_test_loader(cfg):
    ds_name = cfg.TEST.EVAL_DATASET
    return do_setup_loaders(cfg, ds_name)
    

def do_setup_loaders(cfg, ds_names):
    test_dataset_config = setup_test_dataset_config(cfg)
    for k in ds_names:
        gt_path = test_dataset_config[k]['gt_path']
        list_file = 'test.txt' if 'list_file' not in test_dataset_config[k] else test_dataset_config[k]['list_file']
        test_dataset = TestData(cfg=cfg, datapath=gt_path, list_file=os.path.join(gt_path, list_file))
        test_dataset_config[k]['loader'] = DataLoader(
            test_dataset, 
            collate_fn=TestData.collate, 
            batch_size=cfg.DATA.TEST_BATCH_SIZE, 
            shuffle=False, 
            pin_memory=True,
            num_workers=cfg.DATA.NUM_WORKER
        )

    return {k : v['loader'] for k, v in test_dataset_config.items() if k in ds_names} 


def setup_pse_test_loader(cfg):
    """
    args:
    - cfg: config object 
    ret:
    - loaders{dict}
    """
    ds_names = list(set(cfg.TEST.DS_EVAL_TRAIN + cfg.TEST.EVAL_DATASET))
    test_loaders = do_setup_loaders(cfg, ds_names)

    # used to update pseudo label
    pse_update_ds = STDataTest(cfg=cfg, 
        gt_path=cfg.DATA.TGT_DATAPATH, 
        list_file=os.path.join(cfg.DATA.TGT_DATAPATH, 'train.txt'),
        aug_type=cfg.SOLVER.AUG_TYPE,
    )
    
    pse_loader = DataLoader(
        pse_update_ds, 
        collate_fn=pse_update_ds.test_collate,
        batch_size=cfg.DATA.PSE_BATCH_SIZE,
        shuffle=False, 
        pin_memory=True,
        num_workers=cfg.DATA.NUM_WORKER
    )
    
    test_loaders[cfg.DATA.TGT_DATASET] = pse_loader
    return test_loaders


def update_dataset(train_loader, test_loaders, model_wrapper:Base_Wrappepr, cur_epoch, cur_round, cfg = None):

    """
    ret:
    - new_iterator: with data reloaded from the updated dataset
    """
    model = model_wrapper.model
    model.eval()
    round_pse_dir = os.path.join(cfg.savepath, f'pseudo_label_{cur_round}')
    # 如果不存在该 round 对应的伪标签
    if not os.path.exists(os.path.join(round_pse_dir, 'train.txt')):
        os.makedirs(round_pse_dir, exist_ok=True)
        infos = it_pse_update_conf_lite(
            test_loaders['duts_tr'],
            model_wrapper,
            {
                "cur_round_dir": round_pse_dir,
                "save_body_path": os.path.join(round_pse_dir,'body'),
                "save_detail_path": os.path.join(round_pse_dir,'detail'),
                "save_mask_path": os.path.join(round_pse_dir,'mask'),
                "save_file_list_path": os.path.join(round_pse_dir,'train.txt'),
                "filter_out_im_list_path": os.path.join(round_pse_dir,'filter_out.txt'),
                "save_var_path": os.path.join(round_pse_dir,'var'),
            },
            cur_round,
            cfg = cfg,
        )
        # current round target img number
        if cfg and 'tb_writer' in cfg and cfg.tb_writer and 'pse_train_list_len' in infos: 
            cfg.tb_writer.add_scalar(f'pse/tr_img_num', infos['pse_train_list_len'], global_step=cur_round)
        
        if cfg.SOLVER.PSE_POLICY == 'portion' and 'tb_writer' in cfg:
            cfg.tb_writer.add_scalar('train/tr_pse_portion', cfg.tgt_portion_list[cur_round], global_step=cur_round)
            cfg.tb_writer.add_scalar('train/src_portion', cfg.src_portion_list[cur_round], global_step=cur_round)
    
    ## for debug
    pse_mae = eval_exp_pse(round_pse_dir, cfg)
    if cfg and 'tb_writer' in cfg and cfg.tb_writer:
        cfg.tb_writer.add_scalar(f'pse/tr_mae', pse_mae, global_step=cur_round)

    src_portion = cfg.src_portion_list[cur_round]
    train_loader.dataset.update_file_list(round_pse_dir, os.path.join(round_pse_dir, 'train.txt'), src_portion)

    if cfg.SOLVER.REINIT_HEAD :
        logging.info(f"re init predict after round {cur_round}")
        model.init_head()
    
    return iter(train_loader)


def create_model(cfg):
    if "LDF" in cfg.MODEL.NAME:
        logging.info(f"setup LDF wrapper with model name : {cfg.MODEL.NAME}")
        return LDF_Wrapper(cfg)
    else:
        raise Exception(f"No Implementtaion model {cfg.MODEL.NAME}")