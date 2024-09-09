from tqdm import tqdm
import utils
import os
import random
import argparse
import numpy as np
import sys

from torch.utils import data
from datasets.EORSSD_Dataset import EORSSDDataset
from metrics.SOD_metrics import SODMetrics
from network.SLMNet import SLMNet
import torch.nn.functional as F
import torch
import torch.nn as nn
from utils11 import clip_gradient, adjust_lr
from torch.utils.tensorboard import SummaryWriter
import pytorch_iou


def get_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--trainset_path", type=str, 
            default='../Dataset/EORSSD/Train/')
    parser.add_argument("--testset_path", type=str, 
            default='../Dataset/EORSSD/Test/')
    parser.add_argument("--dataset", type=str, default='EORSSD', 
            help='Name of dataset:[EORSSD, ORSSD, ORSI4199]')

    parser.add_argument("--num_classes", type=int, default=2,
                        help='num_classes')
  
    parser.add_argument("--model", type=str, default='SLMNet',
        help='model name:[SLMNet]')
    parser.add_argument("--epochs", type=int, default=45,
                        help="epoch number (default: 40)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate (default: 1e-4)")

    parser.add_argument("--batch_size", type=int, default=8,
                        help='batch size ')
    parser.add_argument("--trainsize", type=int, default=352)
    parser.add_argument("--n_cpu", type=int, default=8,
                        help="download datasets")
    parser.add_argument("--pretrained", type=str,
            default=None, 
            help="restore from checkpoint")
    parser.add_argument("--ckpt", type=str,
            default=None, help="restore from checkpoint")
 
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--save_ep", type=int, default=5,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--save_path", type=str, default='./CHKP_FT/',
                        help="epoch interval for eval (default: 100)")
    parser.add_argument('--log_dir', type=str, default='./logs/', help='')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
    return parser


def get_dataset(opts):

    train_dst = EORSSDDataset(is_train=True,voc_dir=opts.trainset_path, trainsize=opts.trainsize)
    val_dst = EORSSDDataset(is_train=False,voc_dir=opts.testset_path, trainsize=opts.trainsize)
    return train_dst, val_dst


def validate(opts, model, loader, device,  metrics):
    metrics.reset()
    with torch.no_grad():
        for step, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            s1,s2,s3,s4, s1_sig,s2_sig,s3_sig,s4_sig=model(images)
            outputs=s1_sig
            preds=outputs.squeeze()
            labels=labels.squeeze()
    
            metrics.update(preds, labels)

        score = metrics.get_results()
    return score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CE = torch.nn.BCEWithLogitsLoss()
MSE = torch.nn.MSELoss()
IOU = pytorch_iou.IOU(size_average = True)


def main():
    opts = get_argparser().parse_args()
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)
        
    opts.log_dir=opts.log_dir+'{}_{}_ep{}'.format(opts.model, opts.dataset, opts.epochs)
    tb_writer = SummaryWriter(opts.log_dir)
    
    
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    
    print("Device: %s" % device)

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst = get_dataset(opts)
   
    print('opts:',opts)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu,
        drop_last=True)  
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))


    model = eval(opts.model)(img_size=opts.trainsize)
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.999), 
                                 eps=1e-08, weight_decay=opts.weight_decay)
    metrics=SODMetrics(cuda=True)


    if opts.pretrained is not None and os.path.isfile(opts.pretrained):
        # Load SODNet(SLMNet) from SSMAE
        checkpoint = torch.load(opts.pretrained, map_location=torch.device('cpu'))
     
        try:
            model.load_state_dict(checkpoint)
            print('try: load pth from:', opts.pretrained)
        except:
            model_dict      = model.state_dict()
            pretrained_dict = checkpoint
            load_key, no_load_key, temp_dict = [], [], {}
            for k, v in pretrained_dict.items():
                # if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                if k.split('.')[1]=='SODNet':
                    newK=k.split('module.SODNet.')[-1]
                    print(newK)
                    temp_dict[newK] = v
                    load_key.append(newK)
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict)

            print('except: load pth from:', opts.pretrained)

    cur_epoch=0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
    

    model=model.to(device)
    tags = ["train_loss", "learning_rate","MAE","Sm"]
    for epoch in range(cur_epoch,opts.epochs):
        model.train()
        cur_itrs=0
        data_loader = tqdm(train_loader, file=sys.stdout)
        running_loss = 0.0
        adjust_lr(optimizer, opts.lr, epoch, opts.decay_rate, opts.decay_epoch)

        for (images, gts) in data_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            gts = gts.to(device, dtype=torch.float32)

            optimizer.zero_grad()
           
            s1,s2,s3,s4, s1_sig,s2_sig,s3_sig,s4_sig= model(images)
            
            loss1 = CE(s1, gts) + IOU(s1_sig, gts)
            loss2 = CE(s2, gts) + IOU(s2_sig, gts)
            loss3 = CE(s3, gts) + IOU(s3_sig, gts)
            loss4 = CE(s4, gts) + IOU(s4_sig, gts)
    
            total_loss = loss1 + loss2/2 + loss3/4 +loss4/8 
            
            running_loss += total_loss.data.item()

            total_loss.backward()
            optimizer.step()

            data_loader.desc = "Epoch {}/{}, loss={:.4f}".format(epoch, opts.epochs, running_loss/cur_itrs)


        print("validation...")
        model.eval()
        val_score = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)

        print('val_score:',val_score)
        tb_writer.add_scalar(tags[0], (running_loss/cur_itrs), epoch)
        tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[2], val_score["MAE"], epoch)
        tb_writer.add_scalar(tags[3], val_score["Sm"], epoch)
        
        if (epoch+1) % opts.save_ep == 0:
            torch.save(model.state_dict(), opts.save_path+'lastest_{}_{}.pth'.format(opts.model, opts.dataset))
          

if __name__ == '__main__':
    main()
