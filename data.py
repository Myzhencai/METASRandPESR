from __future__ import division
import torch
import os
import glob
import math
import numpy as np
from torch.utils.data import Dataset
import random
import utils
import time
import imageio

class SRDataset(Dataset):
    def __init__(self, dataset, dset_type, patch_size, num_repeats, is_aug=False, crop_type=None, fixed_length=None):
        origin_path = os.path.join('data/origin/', dset_type, dataset)
        bin_path    = os.path.join('data/bin/', dset_type, dataset)
        self.scale = 4
        self.is_aug = is_aug
        self.crop_type = crop_type
        self.patch_size = patch_size
        self.num_repeats = num_repeats
        self.fixed_length = fixed_length
        random.seed(1)

        np_inputs = os.path.join(bin_path, 'inputs.npy')
        np_labels = os.path.join(bin_path, 'labels.npy')

        if os.path.exists(np_inputs) and os.path.exists(np_labels):
            self.inputs = np.load(np_inputs)
            self.labels = np.load(np_labels)
        else:
            print('Numpy binary for {} dataset is not created. Reading image...'.format(dset_type))
            since = time.time()
            hr_path = os.path.join(origin_path, 'HR')
            lr_path = os.path.join(origin_path, 'LR')
            hr_globs = glob.glob(os.path.join(hr_path, '*.bmp'))
            lr_globs = glob.glob(os.path.join(lr_path, '*.bmp'))
            if len(hr_globs) == 0:
                raise Exception('No images found')
            hr_globs.sort()
            lr_globs.sort()
            self.inputs = [imageio.imread(inp) for inp in lr_globs]
            self.labels = [imageio.imread(lbl) for lbl in hr_globs]
            print('Complete reading images in %f seconds' %(time.time() - since))

            print('Writing data to npy...')
            since = time.time()
            if not os.path.exists(bin_path):
                os.makedirs(bin_path)

            np.save(os.path.join(bin_path, 'inputs.npy'), self.inputs)
            np.save(os.path.join(bin_path, 'labels.npy'), self.labels)
            print('Complete writing in %f seconds' %(time.time() - since))
       
        # simple test
        #self.inputs = self.inputs[0:1024]
        #self.labels = self.labels[0:1024]

    def __len__(self):
        if self.fixed_length is not None:
            return self.fixed_length
        return len(self.inputs)*self.num_repeats
    
    def __getitem__(self, idx):
        idx = idx % len(self.inputs)

        inp = self.inputs[idx]
        lbl = self.labels[idx]

        if self.crop_type is not None:
            inp, lbl = self._crop(inp, lbl, self.crop_type)
            
        if self.is_aug:
            inp, lbl = self._aug_data(inp, lbl)

        inp, lbl = self._to_tensor(inp, lbl)
        return inp, lbl

    def _aug_data(self, inp, lbl):
        
        aug_idx = random.randint(0,7)
        assert aug_idx >= 0
        assert aug_idx <= 7

        if (aug_idx>>2)&1 == 1:
            # transpose
            inp = inp.transpose((1, 0, 2)).copy()
            lbl = lbl.transpose((1, 0, 2)).copy()
        if (aug_idx>>1)&1 == 1:
            # vertical flip
            inp = inp[::-1, :, :].copy()
            lbl = lbl[::-1, :, :].copy()
        if aug_idx&1 == 1:
            # horizontal flip
            inp = inp[:, ::-1, :].copy()
            lbl = lbl[:, ::-1, :].copy()

        return inp, lbl
    
    def _crop(self, inp, lbl, crop_type):
        ih, iw, ic = inp.shape #shape of original image

        inp_patch_size = self.patch_size
        lbl_patch_size = inp_patch_size*self.scale

        if crop_type is 'random':
            # indexing inp patch
            h = random.randint(0, ih - inp_patch_size)
            w = random.randint(0, iw - inp_patch_size)
            # indexing lbl patch
            H = h*self.scale
            W = w*self.scale
        elif crop_type is 'fixed':
            h, w, H, W = 0, 0, 0, 0
        else:
            raise Exception('Unknown crop type: {}'.format(crop_type))

        inp = inp[h:h+inp_patch_size, w:w+inp_patch_size, :]
        lbl = lbl[H:H+lbl_patch_size, W:W+lbl_patch_size, :]

        return inp, lbl

    def _to_tensor(self, inp, lbl):
        inp = inp.transpose(2, 0, 1)
        lbl = lbl.transpose(2, 0, 1)
        return torch.FloatTensor(inp), torch.FloatTensor(lbl)

    def _normalize(self, inp, lbl):
        # transpose to channel-last image
        inp = inp.transpose(1, 2, 0)
        lbl = lbl.transpose(1, 2, 0)
        inp = (inp - self.channel_means)/255
        lbl = (lbl - self.channel_means)/255
        inp = inp.transpose(2, 0, 1)
        lbl = lbl.transpose(2, 0, 1)
        return inp, lbl
        
        
        
        
