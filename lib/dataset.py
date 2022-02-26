#!/usr/bin/python3
#coding=utf-8

import sys
sys.path.append('.')

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import random
from lib.fda import FDA_source_to_target_np
import albumentations as A
import logging
from lib.custom_aug import Normalize, Resize, ToTensor, RandomCrop, RandomFlip

########################### Dataset Class ###########################
## testing dataset
class TestData(Dataset):
    def __init__(self, cfg, datapath, list_file):
        self.cfg        = cfg
        self.mean = np.array(cfg.DATA.PIXEL_MEAN).reshape((1, 1, 3))
        self.std = np.array(cfg.DATA.PIXEL_STD).reshape((1, 1, 3))
        self.normalize  = Normalize(mean=self.mean, std=self.std)
        self.resize     = Resize(352, 352)
        self.totensor   = ToTensor()

        # overwrite
        self.file_list_path =  list_file
        self.datapath = datapath 
        logging.info(f"reading data from {self.file_list_path}")
        
        self.mask_path = os.path.join(self.datapath, 'mask')
        self.img_path = os.path.join(self.datapath, 'image')
        self.img_extension = '.png' if 'HKU' in self.datapath else '.jpg'

        with open(self.file_list_path, 'r') as lines:
            self.img_paths = []
            self.mask_paths = []
            for line in lines:
                name = line.strip()
                self.img_paths.append(os.path.join(self.img_path, name + self.img_extension))
                # if not os.path.exists(self.img_paths[-1]):
                #     logging.info(f'{self.img_paths[-1]} not exists')
                self.mask_paths.append(os.path.join(self.mask_path, f'{name}.png'))

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        name  = img_path[img_path.rfind('/') + 1 : img_path.rfind('.')]
        image = cv2.imread(img_path)[:,:,::-1].astype(np.float32) # bgr -> rgb
        shape = image.shape[:2]
        mask  = cv2.imread(self.mask_paths[idx], 0).astype(np.float32) # gt masks
        image = self.normalize(image)
        image = self.resize(image) # no need to resize mask here
        image = self.totensor(image)
        return image, mask, shape, name

    def collate(batch): # class method
        image, mask_list, shape_list, name_list = [list(item) for item in zip(*batch)]
        image  = torch.from_numpy(np.stack(image, axis=0))
        return image, mask_list, shape_list, name_list

    def __len__(self):
        return len(self.img_paths)

## used to mixing source and target dataset
class MixSTData(Dataset):
    # maybe takes long time to write into disk
    PSE_MASK_FOLDER = 'mask'
    PSE_DETAIL_FOLDER = 'body'
    PSE_BODY_FOLDER = 'detail'
    PSE_VAR_FOLDER = 'var'

    SRC_MASK_FOLDER = 'mask'
    SRC_DETAIL_FOLDER = 'body-origin'
    SRC_BODY_FOLDER = 'detail-origin'
    
    def __init__(self, src_datapath, src_file_list_path, tgt_datapath, tgt_file_list_path = None, cfg = None):
        # augemtation code
        self.mean = np.array(cfg.DATA.PIXEL_MEAN).reshape((1, 1, 3))
        self.std = np.array(cfg.DATA.PIXEL_STD).reshape((1, 1, 3))
        self.normalize  = Normalize(mean=self.mean, std=self.std)
        self.cfg = cfg

        self.tgt_file_list_path = None
        self.pse_path = None # location to store pseudo label
        self.src_file_list_path = src_file_list_path
        # self.train_w, self.train_h = 352, 352

        # overwrite
        self.src_datapath = src_datapath # image reading
        self.tgt_datapath = tgt_datapath

        self.mode = 'train'
        self.img_extension = '.jpg' # assert true
        self.im_names = []

        ## target dataset, init will not get this
        self.tgt_im_paths = []
        self.tgt_body_paths = []
        self.tgt_detail_paths = []
        self.tgt_mask_paths = []
        self.tgt_var_paths = []

        fda_tar_im_paths = []
        with open(tgt_file_list_path, 'r') as lines:
            # update pse_path every round 
            for line in lines:
                name = line.strip()
                fda_tar_im_paths.append(os.path.join(self.tgt_datapath, 'image', name + self.img_extension))
        
        augs_list = None

        if cfg.DATA.STRONG_AUG:
            logging.info(f"augmentation with Elastic alpha={cfg.DATA.ET_ALPHA}, sigma={cfg.DATA.ET_SIGMA}, affine={cfg.DATA.ET_SIGMA},p={cfg.DATA.FDA_P}")
            augs_list = [
                A.HorizontalFlip(p=0.5),
                A.ElasticTransform(alpha=cfg.DATA.ET_ALPHA, sigma=cfg.DATA.ET_SIGMA, alpha_affine=cfg.DATA.ET_AFFINE, p=cfg.DATA.ET_P),
                A.OneOf([
                    A.MotionBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=5),
                    A.GaussNoise(var_limit=(5.0, 20.0)),
                ], p=0.5),
                A.RandomResizedCrop(height=352, width=352, scale=cfg.DATA.RC_SCALE),
            ]

            if cfg.DATA.FDA_P != 0:
                logging.info(f"augmentation with fda wiht beta={cfg.DATA.FDA_BETA}, p={cfg.DATA.FDA_P}")
                augs_list.append(A.FDA(fda_tar_im_paths, beta_limit=cfg.DATA.FDA_BETA, p = cfg.DATA.FDA_P))
            elif cfg.DATA.CJ_P != 0:
                logging.info(f"augmentaiton with color jitter wiht brihtness={cfg.DATA.CJ_B}, \
                    contrast={cfg.DATA.CJ_C},saturation={cfg.DATA.CJ_S}, hue={cfg.DATA.CJ_B}, p={cfg.DATA.CJ_P}")
                augs_list.append(
                    A.ColorJitter(
                        brightness=cfg.DATA.CJ_B, 
                        contrast=cfg.DATA.CJ_C,
                        saturation=cfg.DATA.CJ_S, 
                        hue=cfg.DATA.CJ_H,
                        p = cfg.DATA.CJ_P
                    ),
                )
        else:
            logging.info(f"augmentaiton with color jitter wiht brihtness={cfg.DATA.CJ_B}, \
                    contrast={cfg.DATA.CJ_C},saturation={cfg.DATA.CJ_S}, hue={cfg.DATA.CJ_B}, p={cfg.DATA.CJ_P}")
            augs_list = [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter( # add color jitter to waek augmentation
                        brightness=cfg.DATA.CJ_B, 
                        contrast=cfg.DATA.CJ_C,
                        saturation=cfg.DATA.CJ_S, 
                        hue=cfg.DATA.CJ_H,
                        p = cfg.DATA.CJ_P
                ),
                A.OneOf([ # guassin blur
                    A.MotionBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=5),
                    A.GaussNoise(var_limit=(5.0, 20.0)),
                ], p=0.5),
                A.RandomResizedCrop(height=352, width=352, scale=cfg.DATA.RC_SCALE),
            ]
        self.transforms = A.Compose(augs_list,additional_targets={'body': 'mask', 'detail':'mask', 'var':'mask'})

        ## source dataset
        self.src_im_paths = []
        self.src_body_paths = []
        self.src_detail_paths = []
        self.src_mask_paths = []
        self.src_im_names = []
        with open(self.src_file_list_path, 'r') as lines:
            # update pse_path every round 
            for line in lines:
                name = line.strip()
                self.src_im_names.append(name)
                # self.im_names.append(name)
                self.src_im_paths.append(os.path.join(self.src_datapath, 'image', name + self.img_extension))
                self.src_mask_paths.append(os.path.join(self.src_datapath, self.SRC_MASK_FOLDER, name + '.png'))
                self.src_detail_paths.append(os.path.join(self.src_datapath, self.SRC_DETAIL_FOLDER, name + '.png'))
                self.src_body_paths.append(os.path.join(self.src_datapath, self.SRC_BODY_FOLDER, name + '.png'))
        

        self.im_paths = self.src_im_paths[:]
        self.body_paths = self.src_body_paths[:]
        self.detail_paths = self.src_detail_paths[:]
        self.mask_paths = self.src_mask_paths[:]
        self.var_paths = [''] * len(self.im_paths)
        logging.info(f"setup source label dataset !! ")
        logging.info(f"datapath => {self.src_datapath} ")
        logging.info(f"length => {len(self)} ")

    def update_file_list(self, new_psepath, new_list_path, src_portion): # target target 
        self.pse_path = new_psepath
        self.tgt_file_list_path = new_list_path
        self.im_names = []

        self.im_paths = []
        self.body_paths = []
        self.detail_paths = []
        self.mask_paths = []

        self.tgt_im_paths = []
        self.tgt_body_paths = []
        self.tgt_detail_paths = []
        self.tgt_mask_paths = []
        self.tgt_var_paths = []

        # generating target dataset
        with open(self.tgt_file_list_path, 'r') as lines:
            # update pse_path every round 
            for line in lines:
                name = line.strip()
                self.im_names.append(name) # read all 
                self.tgt_im_paths.append(os.path.join(self.tgt_datapath, 'image', name + self.img_extension))
                self.tgt_mask_paths.append(os.path.join(new_psepath, self.PSE_MASK_FOLDER, name + '.png'))
                self.tgt_detail_paths.append(os.path.join(new_psepath, self.PSE_DETAIL_FOLDER, name + '.png'))
                self.tgt_body_paths.append(os.path.join(new_psepath, self.PSE_BODY_FOLDER, name + '.png'))
                self.tgt_var_paths.append(os.path.join(new_psepath, self.PSE_VAR_FOLDER, name + '.png'))


        src_num = int(src_portion * len(self.src_mask_paths))
        src_idxes = np.random.choice(len(self.src_mask_paths), src_num)
        self.im_paths = self.tgt_im_paths[:] + [self.src_im_paths[i] for i in src_idxes]
        self.mask_paths = self.tgt_mask_paths[:] + [self.src_mask_paths[i] for i in src_idxes]
        self.body_paths = self.tgt_body_paths[:] + [self.src_body_paths[i] for i in src_idxes]
        self.detail_paths = self.tgt_detail_paths[:] + [self.src_detail_paths[i] for i in src_idxes]
        self.var_paths = self.tgt_var_paths[:] + [''] * src_num
        src_train_lst_file = os.path.join(new_psepath, 'src_train_lst.txt')
        with open(src_train_lst_file, 'w') as f:
            f.write('\n'.join([self.src_im_names[i] for i in src_idxes]))
        logging.info(f"training with source domain image list : {src_train_lst_file}")
        
        logging.info(f"update pseudo label path to {new_psepath} with new list file {new_list_path}")
        logging.info(f"tot_len {len(self.im_paths)}, tgt_len: {len(self.tgt_im_paths)}, src_len: {src_num}")


    def __getitem__(self, idx):
        # name  = self.im_names[idx]
        img_path = self.im_paths[idx]
        mask_path = self.mask_paths[idx]
        body_path = self.body_paths[idx]
        detail_path = self.detail_paths[idx]
        var_path = self.var_paths[idx]
        image = cv2.imread(img_path) # bgr -> rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(mask_path, 0)

        if self.cfg.SOLVER.ONE_HOT: # training with onehot label
            mask[mask > 127] = 255
            mask[mask <= 127] = 0

        body  = cv2.imread(body_path, 0)
        detail= cv2.imread(detail_path,0)

        if var_path != '': # for warmup
            var = cv2.imread(var_path ,0)
        else:
            var = np.zeros_like(mask)

        aug_res = self.transforms(image = image, mask=mask, body=body, detail=detail, var=var)
        return self.normalize(aug_res['image'], aug_res['mask'], aug_res['body'], aug_res['detail'], aug_res['var'])


    def __len__(self):
        return len(self.im_paths)

    def collate(self, batch):
        size = self.cfg.DATA.TRAIN_MUTI_SCALE[np.random.randint(0, 5)] # ranodm sacle training important

        image, mask, body, detail, var = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            body[i]  = cv2.resize(body[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            detail[i]= cv2.resize(detail[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            var[i]= cv2.resize(var[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image  = torch.from_numpy(np.stack(image, axis=0)).permute(0,3,1,2).float()
        mask   = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1).float()
        body   = torch.from_numpy(np.stack(body, axis=0)).unsqueeze(1).float()
        detail = torch.from_numpy(np.stack(detail, axis=0)).unsqueeze(1).float()
        var = torch.from_numpy(np.stack(var, axis=0)).unsqueeze(1).float()
        return image, mask, body, detail, var

## used to generate pesudo label
class STDataTest(Dataset):
    def flip_trans_back(pred):
        """
        args:
        - pred{np.ndarray}[b, c, h, w]
        ret:
        - pred{np.ndarray}[b, c, h, w]
        """
        return pred[:, :, :, ::-1]
    
    def idt_trans_back(pred):
        return pred
    
    def __init__(self, cfg, gt_path, list_file = None, aug_type = 'flip'):
        """
        cfg: argments
            cfg.mode: mode for training
            cfg.mean, cfg.std: dataset statistic 
            cfg.datapath
        """
        self.cfg        = cfg
        # mean and std of duts
        # TODO:shape
        self.mean = np.array(cfg.DATA.PIXEL_MEAN).reshape((1, 1, 3))
        self.std = np.array(cfg.DATA.PIXEL_STD).reshape((1, 1, 3))
        self.normalize  = Normalize(mean=self.mean, std=self.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize     = Resize(352, 352)
        self.totensor   = ToTensor()
        self.aug_type = aug_type # only fuse

        # overwrite
        self.datapath = gt_path
        self.file_list_path =  list_file
        logging.info(f"reading data from {self.file_list_path}")

        self.fda_aug_num = cfg.SOLVER.FDA_NUM
        self.scale_num = cfg.SOLVER.RAND_SCALE_NUM
        self.flip_num = cfg.SOLVER.FLIP_NUM # should be 0 or 1
            
        self.img_extension = '.png' if 'HKU' in self.datapath else '.jpg'

        self.trans_back_funcs =  [STDataTest.idt_trans_back] + \
            [STDataTest.flip_trans_back] * self.flip_num + \
            [STDataTest.idt_trans_back] * (self.fda_aug_num + self.scale_num)


        if self.file_list_path:
            with open(self.file_list_path, 'r') as lines:
                self.img_paths = []
                for line in lines:
                    name = line.strip()
                    self.img_paths.append(self.datapath+'/image/'+name+self.img_extension)
        
        logging.info(f"iterative upadte dataset with {len(self.img_paths)} train images")

        
    def image_flip(self, img):
        # img{ndarray}[H, W, C]
        res = np.copy(img)
        return res[:, ::-1, :]
    
    def fda_aug(self, src_img, beta = 0.001):
        """
        args:
        - src_img[h, w, c] rgb
        ret:
        - aug_img[h, w, c]
        """ 
        # random style pair 
        # 同等类型的扰动程度
        tgt_idx = np.random.randint(0, len(self))
        tgt_im = cv2.imread(self.img_paths[tgt_idx])[:,:,::-1].astype(np.float32) # bgr -> rgb 
        tgt_im = self.resize(tgt_im)
        aug_im = FDA_source_to_target_np(src_img.transpose((2, 0, 1)), tgt_im.transpose((2, 0, 1)), L=beta)
        return aug_im.transpose((1, 2, 0))

    def get_fda_aug_list(self, src_img, num, beta=0.001): # same beta different version augmentaion
        tgt_idxes = np.random.choice(len(self), num, replace=False)
        ret_ims = []
        for idx in tgt_idxes:
            tgt_im = cv2.imread(self.img_paths[idx])[:,:,::-1].astype(np.float32) # bgr -> rgb
            tgt_im = self.resize(tgt_im)
            aug_im = FDA_source_to_target_np(src_img.transpose((2, 0, 1)), tgt_im.transpose((2, 0, 1)), L=beta).transpose((1, 2, 0))
            ret_ims.append(aug_im)
        return ret_ims

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        name  = img_path[img_path.rfind('/') + 1 : img_path.rfind('.')]
        image = cv2.imread(img_path)[:,:,::-1].astype(np.float32) # bgr -> rgb 
        shape = image.shape[:2]

        
        fda_aug_ims = self.get_fda_aug_list(self.resize(image), self.fda_aug_num)
        
        self.scale_list = [224, 256, 288, 320]
        self.aug_resize_funcs = [Resize(self.scale_list[i], self.scale_list[i]) for i in range(self.scale_num)]
        scale_ims = [aug(image) for aug in self.aug_resize_funcs]
        
        image = self.resize(image)
        flip_im = [self.image_flip(image) for _ in range(self.flip_num)] # ordres matter, this line should be done after image resize

        ims = [self.totensor(self.normalize(im)) for im in [image] + flip_im + fda_aug_ims + scale_ims] # order matters
        return ims, shape, name


    def test_collate(self, batch):
        ims_list, shape_list, name_list = [list(item) for item in zip(*batch)]
        ver_ims = [torch.from_numpy(np.stack(list(item), axis=0)) for item in zip(*ims_list)]
        return ver_ims, shape_list, name_list

    def __len__(self):
        # return len(self.img_paths[:100])
        return len(self.img_paths)


if __name__ == "__main__":
    datapath = 'data/DUTS'
    # class Cfg:
    #     def __init__(self) -> None:
    #         self.train_stronger_aug = False

    ## read config
    from config.defaults import get_cfg_defaults
    cfg = get_cfg_defaults()

    # num = 0``
    # mix_data = MixSTData(
    #     src_datapath=datapath,
    #     src_file_list_path=os.path.join(datapath,'train.txt'),
    #     tgt_datapath=datapath,
    #     tgt_file_list_path=os.path.join(datapath,'train.txt'),
    #     cfg=Cfg()
    # )
    # for i in range(10):
    #     mix_data[i]

    # src_path = 'data/CG4'
    # src_dataset = TrainData(
    #     datapath = src_path,
    #     list_file = os.path.join(src_path,'train.txt'),
    #     cfg = cfg
    # )

    from torch.utils.data import DataLoader
    from lib.exp_logging import make_train_img_grid
    source_loader = DataLoader(
        src_dataset,
        collate_fn=src_dataset.collate, 
        batch_size=2,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        num_workers=0,
    )

    d_iter = iter(source_loader)
    bat = next(d_iter)
    image, mask, body, detail, var = bat
    fig = make_train_img_grid(image, un_norm=True) # savfig
    print("done")    

