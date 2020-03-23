import utils
args = utils.parse_command()
import matplotlib.pyplot as plt
print(args)
import torch
import numpy as np
import h5py
from dataloaders import transforms
import os

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    rgb,depth = val_transform(rgb,depth)
    rgb = np.transpose(rgb, (2, 0, 1))
    return rgb, depth

def val_transform(rgb, depth):
    iheight, iwidth = 480, 640  # raw image size
    output_size = (257, 353)
    depth_np = depth
    transform = transforms.Compose([
        transforms.Resize((iheight, iwidth)),
        transforms.Resize(288.0 / iheight),
        transforms.CenterCrop(output_size),
    ])
    rgb_np = transform(rgb)
    rgb_np = np.asfarray(rgb_np, dtype='float') / 255
    depth_np = transform(depth_np)

    return rgb_np, depth_np
def evaluate(input_files, model):
    preds, targets, inputs = [0]*len(input_files),[0]*len(input_files),[0]*len(input_files)
    for i,input_file in enumerate(input_files):
        input,target = h5_loader(input_file)
        input,target = torch.tensor(input,dtype=torch.float).unsqueeze(0), torch.tensor(target).unsqueeze(0)
        rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
        depth_target_cpu = np.squeeze(target.cpu().numpy())
        if args.gpu:
            input, target = input.cuda(), target.cuda()
            torch.cuda.synchronize()
        with torch.no_grad():
            pred, _ = model(input)  # @wx 注意输出
        pred = utils.get_depth_sid(args, pred)
        if args.gpu:
            torch.cuda.synchronize()
        depth_pred_cpu = np.squeeze(pred.data.cpu().numpy())

        d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
        d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
        depth_target_col = utils.colored_depthmap(depth_target_cpu, d_min, d_max)
        depth_pred_col = utils.colored_depthmap(depth_pred_cpu, d_min, d_max)

        inputs[i] = rgb
        preds[i] = depth_pred_col
        targets[i] = depth_target_col
        # pred = utils.get_depth_sid(args, pred)
        # pred_depth_map = transforms.ToPILImage()(pred[0].cpu())
        # target_depth_map = transforms.ToPILImage()(target[0].cpu())
        # input_img = transforms.ToPILImage('RGB')(input[0].cpu())
        # if i == 0:
        #     img_merge = utils.merge_into_row(input, target, pred)
        # elif i == len(input_files) - 1:
        #     filename = 'result/evaluation.png'
        #     utils.save_image(img_merge, filename)
        # else:
        #     row = utils.merge_into_row(input, target, pred)
        #     img_merge = utils.add_row(img_merge, row)

        #pred = utils.get_depth_sid(args, pred)
        # pred_depth_map = transforms.ToPILImage()(pred[0].cpu())
        # target_depth_map = transforms.ToPILImage()(target[0].cpu())
        # input_img = transforms.ToPILImage('RGB')(input[0].cpu())

    rgb_merge = np.vstack(inputs)
    preds_merge = np.vstack(preds)
    targets_merge = np.vstack(targets)
    utils.save_image(rgb_merge,'result/rgb.png')
    utils.save_image(preds_merge,'result/preds.png')
    utils.save_image(targets_merge,'result/targets.png')

def main():
    data_dir = r'/media/vasp/Data1/Users/vmhp806/data'
    dataset = 'uow_dataset_full'
    input_files = ['1.h5','2.h5','3.h5']
    input_files = [os.path.join(data_dir,dataset,'train/general',f) for f in input_files]
    rsa = r'result/uow_dataset_full_old/sSE/model_best.pth.tar'
    checkpoint = torch.load(rsa)
    model = checkpoint['model']
    del checkpoint
    evaluate(input_files,model)

if __name__ == '__main__':
    main()