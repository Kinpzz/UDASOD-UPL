import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import numpy as np
import time
import cv2
from tqdm import tqdm
from model_wrappers.ldf_wrapper import LDF_Wrapper
import logging

def get_mae(img1, img2):
    # get mae from two gray scale images
    ims = []
    for pred in [img1, img2]:
        pred = pred / 255
        if pred.max() != pred.min(): # prediction normalization, why norm here ?
            pred = (pred - pred.min()) / (pred.max() - pred.min())
        ims.append(pred)
    # return pred, gt
    return np.mean(np.abs(ims[0] - ims[1]))


def trans_to_numpy_img(tsr, shape):
    """
    tsr[H, W]: model output result
    shape
    """
    pred = F.interpolate(tsr.unsqueeze(0), size=shape, mode='bilinear')
    pred = torch.sigmoid(pred[0][0]).cpu().numpy() * 255 # H, W
    return np.round(pred)


def mask2dt_bd(mask): 
    mask = mask.astype(np.uint8)
    body = cv2.blur(mask, ksize=(5,5))
    body = cv2.distanceTransform(body, distanceType=cv2.DIST_L2, maskSize=5)
    body = body**0.5
    tmp  = body[np.where(body>0)]
    if len(tmp)!=0:
        body[np.where(body>0)] = np.floor(tmp/np.max(tmp)*255)
    detail = mask-body
    return body, detail


def get_ent(pred_mask:torch.Tensor):
    """
    args:
    - pred_mask[1, H, W]: output of model
    returns:
    - ent map[H, W]
    """
    z = torch.zeros(()).to(pred_mask.device)
    pred_mask = torch.sigmoid(pred_mask[0])
    ent_map = torch.stack((1 - pred_mask, pred_mask), 0) # 2, H, W
    # avoid log(0)
    ent_map = torch.sum(-1 * torch.where(ent_map == 0., z, torch.log(ent_map)) * ent_map, 0) # H, W 
    return ent_map

def it_pse_update_conf_lite(
        test_loader,
        model_wrapper:LDF_Wrapper, 
        res_path,
        cur_round,
        cfg=None
    ):

    trans_back_list = test_loader.dataset.trans_back_funcs
    test_loader = tqdm(test_loader)
    test_im_shape = cfg.DATA.TEST_SHAPE
    # b_thres, w_thres = 0.001, 0.999
    b_thres, w_thres = cfg.SOLVER.B_THRES, cfg.SOLVER.W_THRES
    logging.info(res_path)
    model = model_wrapper.model
    model.eval() # model need to be evaluation mode
    with torch.no_grad():
        name2consis = {}
        name2ent = {}
        filter_out_file_list = []
        end = time.time()
        for image, shape_list, name_list in test_loader: # mask->numpy array
            data_time = time.time()
            pred_list = []
            bn = image[0].size(0)

            for i, im in enumerate(image):
                im = im.cuda().float()

                out_dict = model_wrapper.pred_img(im, test_im_shape)
                pred = model_wrapper.output2img(out_dict['mask']).cpu().numpy()
                pred = trans_back_list[i](pred) # bn, 1, h, w
                pred_list.append(pred.squeeze(1)) #b, h, w

                
                # from lib.exp_logging import get_unnorm_np_img
                # debug_save_dir = 'debug_err'
                # # if os.path.exists(debug_save_dir):
                # os.makedirs(debug_save_dir, exist_ok=True)
                # im_np = get_unnorm_np_img(im)
                # for j in range(bn):
                #     cv2.imwrite(os.path.join(debug_save_dir, f'{name_list[j]}_{i}.png'), im_np[j])
                #     # import pdb; pdb.set_trace();
                #     cv2.imwrite(os.path.join(debug_save_dir, f'{name_list[j]}_{i}_pred.png'), pred[j].squeeze() * 255)
                

            infer_time = time.time()

            for idx in range(bn):
                shape = shape_list[idx]
                name = name_list[idx]

                # original output
                pred_o = pred_list[0][idx] # np.ndarray h, w 
                assert pred_o.shape == tuple(test_im_shape), pred_o.shape
                
                # sailency region ratio
                posi_ratio = np.mean(np.round(pred_o * 255) > 128)
                if posi_ratio < b_thres or posi_ratio > w_thres:
                    # filter out image with extreme ratio
                    filter_out_file_list.append(f'{name}**{posi_ratio}')
                    continue

                pred_as = np.stack([pred_list[i][idx] for i in range(len(pred_list))], axis=0) # n, h, w
                pred_as_var = np.var(pred_as, axis = 0) # h, w 
                name2consis[name] = np.mean(pred_as_var) # 1

                pred_o = cv2.resize(pred_o, dsize=tuple(shape)[::-1], interpolation=cv2.INTER_LINEAR) # cv2 resize w, h, 
                pred_o = np.round(pred_o * 255)
                pred_as_var = cv2.resize(pred_as_var, dsize=tuple(shape)[::-1], interpolation=cv2.INTER_LINEAR) # cv2 resize w, h, 
                pred_as_var = np.round(pred_as_var * 255)

                # if cfg.SOLVER.ONE_HOT: # not one hot here
                #     pred_o = (pred_o > 128).astype(np.uint8) * 255
                
                if cfg.SOLVER.ONLY_MASK:
                    bd, dt = mask2dt_bd(pred_o)
                else:
                    raise NotImplementedError("Only support only_mask schema currently")

                for path, img in [(p, i) for p, i in [
                    (res_path["save_body_path"], bd),
                    (res_path["save_detail_path"], dt),
                    (res_path["save_mask_path"], pred_o),
                    (res_path["save_var_path"], pred_as_var)
                    ] if p and p != '']:
                    if not os.path.exists(path):
                        os.makedirs(path)
                    cv2.imwrite(os.path.join(path, f'{name}.png'), img)
            
            saving_time = time.time()
            test_loader.set_postfix(
                data_time = data_time - end, 
                infer_time=infer_time - data_time, 
                saving_time=saving_time - infer_time
            )
            end = time.time()
        

        ## metric to choose pseudo label
        sorted_name2metric = None
        name2metric = None
        if cfg.SOLVER.LABEL_SELECT_STRATEGY == 'ent':
            name2metric = name2ent
            sorted_name2metric = sorted(name2ent.items(), key=lambda tup: tup[1])
        elif cfg.SOLVER.LABEL_SELECT_STRATEGY == 'consis':
            name2metric = name2consis
            sorted_name2metric = sorted(name2consis.items(), key=lambda tup: tup[1])

        tot_len = len(sorted_name2metric)
        if cfg.SOLVER.PSE_POLICY == 'portion':
            ## filter image according to preset portion
            portion = cfg.tgt_portion_list[cur_round]
            filter_portion = cfg.SOLVER.PSE_FILTER_PORTION
            train_list_len = int(tot_len * portion)
            selected_win_start = int(filter_portion * train_list_len)
            selected_win_end = selected_win_start + train_list_len
        elif cfg.SOLVER.PSE_POLICY == 'threshold':
            ## filter image according to threshold
            var_thres = cfg.SOLVER.VAR_THRESHOLD
            new_train_lst = [tup for tup in sorted_name2metric if tup[1] <= var_thres]
            train_list_len = len(new_train_lst)
            selected_win_start = 0
            selected_win_end = len(new_train_lst)
            new_train_lst = sorted_name2metric[selected_win_start:selected_win_end]

        with open(res_path["save_file_list_path"], "w") as f:
            f.write('\n'.join([tup[0] for tup in sorted_name2metric[selected_win_start:selected_win_end]]))
        
        # maybe read from var file
        with open(os.path.join(res_path['cur_round_dir'], 'im_score.csv'), 'w') as f:
            f.write('\n'.join([f'{tup[0]}, {tup[1]}' for tup in sorted_name2metric]))

        if "filter_out_im_list_path" in res_path:
            with open(res_path["filter_out_im_list_path"], "w") as f:
                f.write('\n'.join(filter_out_file_list))
        
        # just for debug
        logging.info(f"write pseudo label into:")
        for k, v in res_path.items():
            logging.info(f"{k}: {v}")
            logging.info("==" * 10)
        logging.info(f"unfiltered dataset length: {tot_len}")
        logging.info(f"dataset length: {train_list_len} from idx {selected_win_start} to {selected_win_end}")

        return {
            'pse_train_list_len': train_list_len
        }

if __name__ == '__main__':
    # test
    # create a model
    pass