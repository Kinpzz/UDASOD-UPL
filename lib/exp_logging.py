# TODO:
# 1. log body image and detail image
# 
from os import name
from os.path import join
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision


input_mean = []
input_std = []

def log_exp_info(str, tb_writer:SummaryWriter = None):
    #TODO: add other info
    print(str)
    if tb_writer:
        tb_writer.add_text(
            'exp_log', str + '\n\n'
        )

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

def unnorm_imgs(imgs:torch.Tensor, mean, std):
    unnorm = NormalizeInverse(mean, std)
    img_tensor = torch.stack(list(map(unnorm, torch.unbind(imgs, 0))), 0)
    return img_tensor

def _get_sod_fig(imgs , gts , sods, bodys, details, im_name_list=None):
    """
    imgs[B, H, W, C]: source image range from 0 to 1
    gts[B, H, W]: groud truth mask
    sods[B, H, W]: predict
    bodys[B, H, W]: predict body 
    details[B, H, W]: predict detail
    """
    # TODO: additional information
    bn = len(imgs)
    # assert bn == 32
    fig = plt.figure(figsize=(5 * 2, bn * 2)) # TODO: figure size
    col_num, row_num = 5, bn
    counter = 0
    for r in range(row_num):
        ax0 = fig.add_subplot(row_num, col_num, r * col_num + 1)
        ax1 = fig.add_subplot(row_num, col_num, r * col_num + 2)
        ax2 = fig.add_subplot(row_num, col_num, r * col_num + 3) 
        ax3 = fig.add_subplot(row_num, col_num, r * col_num + 4) 
        ax4 = fig.add_subplot(row_num, col_num, r * col_num + 5)
        for ax in [ax0, ax1, ax2, ax3, ax4]:
            ax.set_axis_off()            

        ax0.imshow(imgs[r])
        ax1.imshow(gts[r], cmap='gray')
        ax2.imshow(sods[r], cmap='gray')
        ax3.imshow(bodys[r], cmap='gray') 
        ax4.imshow(details[r], cmap='gray')
        if im_name_list:
            ax2.set_title(im_name_list[r])
    plt.tight_layout()
    return fig


def _get_consis_fig(imgs_list):
    """
    imgs[B, H, W, C]: source image range from 0 to 1
    """
    # TODO: additional information
    bn, c, h, w = imgs_list[0].shape
    fig = plt.figure(figsize=(len(imgs_list) * 2, bn * 2)) # TODO: figure size
    col_num, row_num = len(imgs_list), bn
    for r in range(row_num):
        ax1 = fig.add_subplot(row_num, col_num, r * col_num + 1)
        ax1.set_axis_off()
        ax1.imshow(imgs_list[0][r])
        for ci, ims in enumerate(imgs_list[1:]):
            ax = fig.add_subplot(row_num, col_num, r * col_num + ci + 2, sharey=ax1, sharex=ax1)
            ax.set_axis_off()
            ax.imshow(ims[r])
    plt.tight_layout()
    return fig

def get_consis_fig(imgs, aug_imgs, un_norm = True):
    args = []
    mean = [124.55, 118.90, 102.94]
    std = [ 56.77,  55.97,  57.50]
    imgs , aug_imgs = [e.cpu() for e in [imgs , aug_imgs]]

    if un_norm:
        imgs = unnorm_imgs(imgs, mean, std).permute((0, 2, 3, 1))
        aug_imgs = unnorm_imgs(aug_imgs, mean, std).permute((0, 2, 3, 1))

    args.append(torch.clamp(imgs / 255, 0, 1).numpy())
    args.append(torch.clamp(aug_imgs / 255, 0, 1).numpy())
    return _get_consis_fig(args)

def get_unnorm_np_img(imgs):
    # rgb 2 bgr
    mean = [124.55, 118.90, 102.94]
    std = [ 56.77,  55.97,  57.50]
    imgs = imgs.cpu()
    imgs = unnorm_imgs(imgs, mean, std).permute((0, 2, 3, 1))
    return imgs.numpy()[:, :, :, ::-1]


def get_train_ims(imgs, gray_imgs, un_norm = True):
    args = []
    mean = [124.55, 118.90, 102.94]
    std = [ 56.77,  55.97,  57.50]
    imgs = imgs.cpu()
    gray_imgs = [e.squeeze().cpu().numpy() for e in gray_imgs]
    if un_norm:
        imgs = unnorm_imgs(imgs, mean, std).permute((0, 2, 3, 1))

    args.append(torch.clamp(imgs / 255, 0, 1).numpy())
    return _get_consis_fig(args + gray_imgs)

def get_consis_fig_list(ims_list, un_norm = True):
    args = []
    mean = [124.55, 118.90, 102.94]
    std = [ 56.77,  55.97,  57.50]
    ims_list = [e.cpu() for e in ims_list]

    for imgs in ims_list:
        if un_norm:
            imgs = unnorm_imgs(imgs, mean, std).permute((0, 2, 3, 1))

        args.append(torch.clamp(imgs / 255, 0, 1).numpy())
    return _get_consis_fig(args)


def get_sod_fig_np(imgs , gts , sods, bodys, details, im_name_list = None):
    """
    args:
    - imgs[B, H, W,C]:
    - gts[B, H, W]:
    - sods[B, H, W]:
    - bodys[B, H, W]:
    - details[B, H, W]:
    return:
    - matplot.Figure
    """
    # order matters
    args = []
    for ims in [imgs, gts, sods, bodys, details]:
        preds = [np.clip(im / 255, 0, 1) for im in ims]
        args.append(preds)
    return _get_sod_fig(*args, im_name_list)


def get_sod_fig(imgs , gts , sods, bodys, details, un_norm = True):
    """
    args:
    - imgs[B, C , H, W]:
    - gts[B, H, W]:
    - sods[B, H, W]:
    - bodys[B, H, W]:
    - details[B, H, W]:
    return:
    - matplot.Figure
    """
    # order matters
    args = []
    mean = [124.55, 118.90, 102.94]
    std = [ 56.77,  55.97,  57.50]
    imgs , gts , sods, bodys, details = [e.cpu() for e in [imgs , gts , sods, bodys, details]]
    if un_norm:
        imgs = unnorm_imgs(imgs, mean, std).permute((0, 2, 3, 1))
    args.append(torch.clamp(imgs / 255, 0, 1).numpy())
    args.append(np.round(gts.cpu().squeeze().numpy()))
    for ims in [sods, bodys, details]:
        pred = torch.sigmoid(ims).cpu().squeeze().numpy() # H, W
        args.append(np.round(pred))
    return _get_sod_fig(*args)


def make_train_img_grid(imgs, un_norm = False, num_colum = 4):
    """
    input: bn, c, h, w
    """
    mean = [124.55, 118.90, 102.94]
    std = [ 56.77,  55.97,  57.50]
    if un_norm:
        imgs = unnorm_imgs(imgs, mean, std) / 255
    imgs_grid = torchvision.utils.make_grid(imgs, nrow=num_colum, padding=5)
    return imgs_grid