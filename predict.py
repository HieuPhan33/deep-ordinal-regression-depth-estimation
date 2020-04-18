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

def get_file(name):
    data_dir = r"Z:/10-Share/depth estimation/data"
    path = os.path.join(data_dir,'uow_dataset_full','train','general',name)
    return h5_loader(path)

def display_rgb_gt(name):
    input,target = get_file(name)
    rgb = 255*input
    d_min = min(np.min(target), np.min(target))
    d_max = max(np.max(target), np.max(target))
    depth_target_col = utils.colored_depthmap(target, d_min, d_max)
    #depth_target_col = np.transpose(depth_target_col,(2,0,1))
    #result = np.hstack((rgb,depth_target_col))
    rgb = np.transpose(rgb,(1,2,0))
    rgb = np.array(rgb,dtype='uint8')
    name = os.path.splitext(name)[0]
    utils.save_image(rgb,'result/{}_rgb_diagram.png'.format(name))
    utils.save_image(depth_target_col,'result/{}_depth_diagram.png'.format(name))
    #plt.imshow(rgb)
    #plt.show()
    # fig, axs = plt.subplots(1,2)
    # axs[0].imshow(rgb)
    # axs[1].imshow(depth_target_col)
    # plt.show()

def evaluate(input_files, model,result_dir):
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
    utils.save_image(rgb_merge,os.path.join(result_dir,'rgb.png'))
    utils.save_image(preds_merge,os.path.join(result_dir,'pred.png'))
    utils.save_image(targets_merge,os.path.join(result_dir,'target.png'))

def main():
    data_dir = r'Z:\10-Share\depth estimation\data'
    dataset = 'uow_dataset_full'
    # input_files = ['49.h5','60.h5','67.h5']
    # input_files = [os.path.join(data_dir,dataset,'val\general',f) for f in input_files]
    # result_dir = r'result\uow_dataset_full\run_3'
    # rsa = os.path.join(result_dir,'model_best.pth.tar')
    # checkpoint = torch.load(rsa)
    # model = checkpoint['model']
    # model.src_device_obj = torch.device('cuda:0')
    # del checkpoint
    # evaluate(input_files,model,result_dir)
    display_rgb_gt('3.h5')
    # for name in os.listdir(os.path.join(data_dir,dataset,'train','general')):
    #     display_rgb_gt(name)
    #     input("Press Enter to continue...")
        #print(name)

if __name__ == '__main__':
    main()