import torch
import torch.nn.functional as F

import numpy as np
import os, argparse
import imageio

from network.SLMNet import SLMNet
from data import test_dataset
from tqdm import tqdm
import time
import cv2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str, 
        default='../Dataset/')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument("--model", type=str, default='SLMNet',
        help='model name:[SLMNet]')
parser.add_argument("--smap_save", type=str, default='../SalPreds/',
        help='model name')
parser.add_argument("--load", type=str,
            default='',
              help="restore from checkpoint")
opt = parser.parse_args()


def create_folder(save_path):
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Create Folder [“{save_path}”].")
    return save_path


model = eval(opt.model)()

if opt.load is not None and os.path.isfile(opt.load):
    checkpoint = torch.load(opt.load, map_location=torch.device('cpu'))
    try:
        model.load_state_dict(checkpoint)
        print('try: load pth from:', opt.load)
    except:
        model_dict      = model.state_dict()
        pretrained_dict = checkpoint
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            # if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            newK=k.split('module.')[-1]
            print(newK)
            temp_dict[newK] = v
            load_key.append(newK)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        print('except: load pth from:', opt.load)
    del checkpoint  # free memory

    
model.cuda()
model.eval()


test_datasets = ['ORSSD', 'EORSSD', 'ORSI4199']
for dataset in test_datasets:
    # load data
    image_root = opt.test_path + dataset + '/Test/Images/'
    gt_root = opt.test_path + dataset + '/Test/Masks/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    method=opt.load.split('/')[-1].split('.')[0]
    save_path = create_folder(opt.smap_save + dataset + '/'+method+'/')
    print('{} preds for {}'.format(method, dataset))
   
    cost_time = list()
    for i in tqdm(range(test_loader.size), desc=dataset):
        with torch.no_grad():
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            name = name.split('/')[-1]
            image = image.cuda()
       
            start_time = time.perf_counter()
            s1,s2,s3,s4, s1_sig,s2_sig,s3_sig,s4_sig=model(image)

            res=s1
            cost_time.append(time.perf_counter() - start_time)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            imageio.imsave(save_path+name, res)
      
    cost_time.pop(0)
    print('Mean running time is: ', np.mean(cost_time))
    print("FPS is: ", test_loader.size / np.sum(cost_time))
print("Predict Done!")


