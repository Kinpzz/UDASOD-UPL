## adapt from LDF 
import sys
sys.path.append('.')

import cv2
import torch
import numpy as np

########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std): # keep dtype consistence
        self.mean = mean.astype(np.float32)
        self.std  = std.astype(np.float32)

    def __call__(self, image, mask=None, body=None, detail=None, var=None):
        image = (image - self.mean)/self.std
        if mask is None:
            return image
        if var is None:
            return image, mask/255, body/255, detail/255
        else: 
            return image, mask/255, body/255, detail/255, var / 255

class RandomCrop(object):
    def __call__(self, image, mask=None, body=None, detail=None, var=None):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        if mask is None:
            return image[p0:p1,p2:p3, :]
        if var is None:
            return image[p0:p1,p2:p3, :], mask[p0:p1,p2:p3], body[p0:p1,p2:p3], detail[p0:p1,p2:p3]
        else:
            return image[p0:p1,p2:p3, :], mask[p0:p1,p2:p3], body[p0:p1,p2:p3], detail[p0:p1,p2:p3], var[p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask=None, body=None, detail=None, var=None):
        if np.random.randint(2)==0:
            if mask is None:
                return image[:,::-1,:].copy()
            if var is None:
                return image[:,::-1,:].copy(), mask[:, ::-1].copy(), body[:, ::-1].copy(), detail[:, ::-1].copy()
            else:
                return image[:,::-1,:].copy(), mask[:, ::-1].copy(), body[:, ::-1].copy(), detail[:, ::-1].copy(), var[:, ::-1].copy()
        else:
            if mask is None:
                return image
            if var is None:
                return image, mask, body, detail
            else:
                return image, mask, body, detail, var

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None, body=None, detail=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        body  = cv2.resize( body, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        detail= cv2.resize( detail, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, body, detail

class ToTensor(object):
    def __call__(self, image, mask=None, body=None, detail=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image
        mask  = torch.from_numpy(mask)
        body  = torch.from_numpy(body)
        detail= torch.from_numpy(detail)
        return image, mask, body, detail
