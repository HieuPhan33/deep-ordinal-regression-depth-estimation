import torch

import numpy as np
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader

iheight, iwidth = 480, 640  # raw image size
alpha, beta = 0.02, 10.0  # NYU Depth, min depth is 0.02m, max depth is 10.0m
K = 68  # NYU is 68, but in paper, 80 is good

'''
In this paper, all the images are reduced to 288 x 384 from 480 x 640,
And the model are trained on random crops of size 257x353.
'''


class NYUDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb'):
        super(NYUDataset, self).__init__(root, type, sparsifier, modality)
        self.output_size = (257, 353)

    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5)  # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize((iheight,iwidth)),
            transforms.Resize(288.0 / iheight),  # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np)  # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize((iheight, iwidth)),
            transforms.Resize(288.0 / iheight),
            transforms.CenterCrop(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)
        return rgb_np, depth_np

import os
import utils
import seaborn as sns
from collections import Counter
from tqdm import tqdm
args = utils.parse_command()
import matplotlib.pyplot as plt
print(args)
if __name__ == '__main__':
    data_dir = r'Z:\10-Share\depth estimation'
    valdir = os.path.join(data_dir, 'data', 'uow_dataset_full', 'val')
    traindir = os.path.join(data_dir, 'data', 'uow_dataset_full', 'train')
    train_set = NYUDataset(traindir, type='train')
    val_set = NYUDataset(valdir, type='val')

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=8, shuffle=False, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=8, shuffle=False, pin_memory=True)
    c = Counter()
    max_depth, min_depth = -1, 10000
    for im, gt in tqdm(train_loader):
        # print(im)
        valid = (gt > 0.0)
        max_depth = max(max_depth, torch.max(gt[valid]))
        min_depth = min(min_depth, torch.min(gt[valid]))
        labels,_ = utils.get_labels_sid(args,gt[valid])
        labels,counts = np.unique(labels.cpu().detach().numpy(),return_counts=True)
        d = {k:v for k,v in zip(labels,counts)}
        c.update(d)
    print(min_depth, '-', max_depth)
    #c = Counter({62: 23994577, 61: 23183834, 63: 22670103, 60: 22313719, 64: 20980929, 59: 19317232, 65: 18881079, -2147483648: 17137674, 66: 16685405, 67: 14848254, 58: 13287308, 68: 13199124, 69: 11732497, 70: 10237682, 71: 9000830, 72: 7654900, 57: 7321212, 73: 6868126, 74: 5969406, 75: 4998240, 56: 4646872, 76: 4263847, 55: 3186416, 77: 3167009, 54: 2454062, 53: 1568245, 78: 1546472, 52: 1231918, 51: 902041, 48: 665191, 50: 624923, 49: 499470, 79: 400990, 45: 204760, 44: 131008, 43: 111338, 46: 3829})
    print(c)
    sns.barplot(x=list(c.keys()),y=list(c.values()))
    plt.savefig('depth_distribution.png',dpi=200)
    plt.show()
    #print(min_depth,'-',max_depth)