# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/23 23:00
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""


import os
import numpy as np
from PIL import Image
from scipy import io
from torch.utils.data import Dataset
iheight, iwidth = 2272,1704  # raw image size
gt_height,gt_width = 55,305
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path, rgb=True):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    if rgb:
        with open(path, 'rb') as f:
            img = Image.open(path)
    else:
        img = io.loadmat(path)['Position3DGrid']
        img = img[:,:,3]
    return img


def readPathFiles(file_path, root_dir):
    im_gt_paths = []
    path = os.path.join(root_dir,file_path,'rgb')
    img_files = os.listdir(path)
    for img_file in img_files:
        if img_file.endswith('.jpg'):
            name = os.path.splitext(img_file)[0][4:]
            img_file = os.path.join(root_dir,file_path,'rgb',img_file)
            depth_file = 'depth_sph_corr-{}.mat'.format(name)
            depth_file = os.path.join(root_dir,file_path,'depth',depth_file)
            im_gt_paths.append((img_file,depth_file))
    return im_gt_paths


# array to tensor
from dataloaders import transforms as my_transforms
to_tensor = my_transforms.ToTensor()


class Make3DFolder(Dataset):
    def __init__(self, root_dir='/data/make3d/train',
                 mode='train', loader=pil_loader, size=(460, 345)):
        super(Make3DFolder, self).__init__()
        self.root_dir = root_dir

        self.mode = mode
        self.im_gt_paths = None
        self.loader = loader
        self.size = size

        if self.mode == 'train':
            self.im_gt_paths = readPathFiles('train', root_dir)

        elif self.mode == 'test':
            self.im_gt_paths = readPathFiles('val', root_dir)

        else:
            print('no mode named as ', mode)
            exit(-1)

    def __len__(self):
        return len(self.im_gt_paths)

    def train_transform(self, im, gt):
        im = np.array(im)
        #im = np.transpose(im, (2,0,1))
        gt = np.array(gt).astype(np.float32)

        s = np.random.uniform(1.0, 1.5)  # random scaling
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
        color_jitter = my_transforms.ColorJitter(0.4, 0.4, 0.4)


        transform = my_transforms.Compose([
            my_transforms.Resize((736,512), interpolation='bilinear'),
            my_transforms.Rotate(angle),
            my_transforms.Resize(s),
            my_transforms.CenterCrop(self.size),
            my_transforms.HorizontalFlip(do_flip)
        ])



        im_ = transform(im)
        im_ = color_jitter(im_)

        gt_ = transform(gt)

        im_ = np.array(im_).astype(np.float32)
        gt_ = np.array(gt_).astype(np.float32)

        im_ /= 255.0
        gt_ /= 100.0 * s
        im_ = to_tensor(im_)
        gt_ = to_tensor(gt_)

        gt_ = gt_.unsqueeze(0)

        return im_, gt_

    def val_transform(self, im, gt):
        im = np.array(im)
        im = im.astype('uint8')

        gt = np.array(gt).astype(np.float32)

        transform = my_transforms.Compose([
            #my_transforms.Crop(130, 10, 240, 1200),
            my_transforms.Resize((736,512), interpolation='bilinear'),
            my_transforms.CenterCrop(self.size)
        ])

        im_ = transform(im)
        gt_ = transform(gt)

        im_ = np.array(im_).astype(np.float32)
        gt_ = np.array(gt_).astype(np.float32)

        im_ /= 255.0
        #gt_ /= 100.0
        im_ = to_tensor(im_)
        gt_ = to_tensor(gt_)

        gt_ = gt_.unsqueeze(0)
        return im_, gt_

    def __getitem__(self, idx):
        im_path, gt_path = self.im_gt_paths[idx]

        # if self.mode == 'train':
        #     im_path = os.path.join(self.root_dir, 'kitti_raw_data', im_path)

        im = self.loader(im_path)
        gt = self.loader(gt_path, rgb=False)

        if self.mode == 'train':
            im, gt = self.train_transform(im, gt)

        else:
            im, gt = self.val_transform(im, gt)
        return im, gt


import torch
from tqdm import tqdm

if __name__ == '__main__':
    root_dir = '../data/make3d'

    # im_gt_paths = readPathFiles('./eigen_val_pairs.txt', root_dir)

    data_set = Make3DFolder(root_dir, mode='test', size=(460, 345))
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=4, shuffle=False, num_workers=0)

    print('dataset num is ', len(data_loader))
    max_depth,min_depth = -1,10000
    for im, gt in tqdm(data_loader):

        # print(im)

        valid = (gt > 0.0)
        max_depth = max(max_depth,torch.max(gt[valid]))
        min_depth = min(min_depth,torch.min(gt[valid]))
    print(min_depth,'-',max_depth)