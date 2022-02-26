from model_wrappers.base_wrapper import Base_Wrappepr
import os
import torch
import numpy as np
import tqdm

from tqdm import tqdm

import lib.metrics as M
import torch.nn.functional as F
import cv2
import logging


def get_models_name(exp_root):
    """
    ret:
    - model_cpt{dict[name: state_dict]}
    """
    model_cpt = {}
    for fn in os.listdir(exp_root):
        if fn.startswith('round'):
            cpt = torch.load(os.path.join(exp_root, fn))
            model_cpt[fn] = cpt['state_dict']

        if fn.startswith('best_model'):
            cpt = torch.load(os.path.join(exp_root, fn))
            model_cpt[fn] = cpt
    return model_cpt

def get_report_metrics(mets):
    report_met = []
    b2d = {
        'fm': ['adp_fm', 'mean_fm', 'max_fm'],
        'em': ['adp_em', 'mean_em', 'max_em'],
    }
    for met in mets:
        if met in b2d:
            report_met += b2d[met]
        else:
            report_met += [met]
    return report_met
    

def test(test_loaders, model_wrapper:Base_Wrappepr,ds_names, metrics, save_res = True, cfg = None):
    """
    args:
    - test_loaders{Map}ds_name->ds_loader
    - mode_wapper
    - ds_names{list}
    - metrics{list}
    - cfg
    return:
    - dict{ds - > {met -> num}}
    """
    model_wrapper.model.eval()
    device = next(model_wrapper.model.parameters()).device
    with torch.no_grad():
        fin_res = {}
        for ds in ds_names:
            all_mets = {
                'fm': M.Fmeasure(),
                'wfm': M.WeightedFmeasure(),
                'sm': M.Smeasure(),
                'em': M.Emeasure(),
                'mae': M.MAE()
            }
            req_met = {k: all_mets[k] for k in metrics}
            loader = tqdm(test_loaders[ds])
            res = {}
            for image, mask_list, shape_list, name_list in loader: # mask->numpy array
                image = image.to(device).float()
                bn = image.size(0)
                out_dict = model_wrapper.pred_img(image) # tensor
                for idx in range(bn):
                    shape = shape_list[idx]
                    name = name_list[idx]
                    mask = mask_list[idx]
                    pred = model_wrapper.postprocess_np(out_dict['mask'][idx], shape).round()
                    if mask.shape != pred.shape:
                        print(f'img_name: {name}, shape gt {mask.shape}, shape pred {pred.shape}')
                        continue
                    if save_res: # for test only
                        save_mask_path = os.path.join(cfg.savepath, ds, 'mask')
                        # save_body_path = os.path.join(cfg.savepath, ds, 'body')
                        # save_detail_path = os.path.join(cfg.savepath, ds, 'detail')
                        if 'LDF' in cfg.MODEL.NAME: 
                            ## saving predition
                            for path, img in [(p, i) for p, i in [
                                # (save_body_path, model_wrapper.postprocess_np(out_dict['body'][idx], shape)),  
                                # (save_detail_path, model_wrapper.postprocess_np(out_dict['detail'][idx], shape)),
                                (save_mask_path, pred)] if p and p != '']:
                                if not os.path.exists(path):
                                    os.makedirs(path)
                                cv2.imwrite(os.path.join(path, f'{name}.png'), img)

                    for k in req_met:
                        req_met[k].step(pred=pred, gt=mask)
                    
            if 'sm' in req_met:
                sm = req_met['sm'].get_results()['sm']
                res['sm'] = sm.round(4)
            if 'wfm' in req_met:
                wfm = req_met['wfm'].get_results()['wfm']
                res['wfm'] = wfm.round(4)

            if 'fm' in req_met:
                fm = req_met['fm'].get_results()['fm']
                res['adp_fm'] = fm['adp'].round(4)
                res['mean_fm'] = fm['curve'].mean().round(4)
                res['max_fm'] = fm['curve'].max().round(4)
            
            if 'em' in req_met:
                em = req_met['em'].get_results()['em']
                res['adp_em'] = em['adp'].round(4)
                res['mean_em'] = -1 if em['curve'] is None else em['curve'].mean().round(4)
                res['max_em'] = -1 if em['curve'] is None else em['curve'].max().round(4)
            
            if 'mae' in req_met:
                mae = req_met['mae'].get_results()['mae']
                res['mae'] = mae.round(4)

            fin_res[ds] = res
            logging.info(f'{ds} : {res}')
        return fin_res


def eval(preds_dir, gts_dir, ds_names, metrics):
    """
    args:
    - preds_dir{String}: path of predicted maps
    - gts_dir{String}: ground truth dir
    - ds_names{List}: dataset to evaluate
    - metrics{List}: metrics
    return:
    - dict{ds - > {met -> num}}
    """
    fin_res = {}
    for ds in ds_names:
        all_mets = {
            'fm': M.Fmeasure(),
            'wfm': M.WeightedFmeasure(),
            'sm': M.Smeasure(),
            'em': M.Emeasure(),
            'mae': M.MAE()
        }
        req_met = {k: all_mets[k] for k in metrics}
        res = {}
        # for image, mask_list, shape_list, name_list in loader: # mask->numpy array
        pred_ds_dir = os.path.join(preds_dir, ds, 'mask')
        gt_ds_dir = os.path.join(gts_dir[ds]['gt_path'], 'mask')
        
        for img_name in tqdm(os.listdir(pred_ds_dir)):
            pred = cv2.imread(os.path.join(pred_ds_dir, img_name), 0)
            mask = cv2.imread(os.path.join(gt_ds_dir,img_name), 0)
            for k in req_met:
                req_met[k].step(pred=pred, gt=mask)
                
        if 'sm' in req_met:
            sm = req_met['sm'].get_results()['sm']
            res['sm'] = sm.round(4)
        if 'wfm' in req_met:
            wfm = req_met['wfm'].get_results()['wfm']
            res['wfm'] = wfm.round(4)

        if 'fm' in req_met:
            fm = req_met['fm'].get_results()['fm']
            res['adp_fm'] = fm['adp'].round(4)
            res['mean_fm'] = fm['curve'].mean().round(4)
            res['max_fm'] = fm['curve'].max().round(4)
        
        if 'em' in req_met:
            em = req_met['em'].get_results()['em']
            res['adp_em'] = em['adp'].round(4)
            res['mean_em'] = -1 if em['curve'] is None else em['curve'].mean().round(4)
            res['max_em'] = -1 if em['curve'] is None else em['curve'].max().round(4)
        
        if 'mae' in req_met:
            mae = req_met['mae'].get_results()['mae']
            res['mae'] = mae.round(4)

        fin_res[ds] = res
        logging.info(f'{ds} : {res}')
    return fin_res
