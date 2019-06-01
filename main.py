#PyTorch 0.3.1, TorchVision 0.2.0, Python 2.7
import torch
import argparse
import numpy as np
import scipy.misc as misc
from torch.utils import data
from loader import get_loader, get_data_path
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from model import *
import os
# torch.backends.cudnn.benchmark=True

num_class=41

def get_iou(pred, gt):
    if pred.shape != gt.shape:
        print ('pred shape', pred.shape, 'gt shape', gt.shape)
    assert (pred.shape == gt.shape)
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    max_label = int(num_class) - 1  # labels from 0,1, ... C
    count = np.zeros((max_label + 1,))
    for j in range(1,max_label + 1):
        x = np.where(pred == j)
        p_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        x = np.where(gt == j)
        GT_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        n_jj = set.intersection(p_idx_j, GT_idx_j)
        u_jj = set.union(p_idx_j, GT_idx_j)

        if len(GT_idx_j) != 0:
            count[j] = float(len(n_jj)) / float(len(u_jj))

    result_class = count
    Aiou = np.sum(result_class[:]) / float(len(np.unique(gt))-1)
    count_class=(result_class!=0)

    return Aiou, result_class,count_class

def accuracy(model,loader):
    num_correct_lbl, num_samples_lbl = 0, 0; Aiou=0;Am_acc=0
    num_samples=len(loader)

    class_Aiou = np.zeros((num_class,))
    count_class = np.zeros((num_class,))

    out_prdlbl='outimgs/'
    for i, (img, lbl) in enumerate(loader):
        img_var = Variable(img.cuda(), volatile=True)
        _,_,_,_,_,pred_labels = model(img_var)

        _, preds_lbl = pred_labels.data.cpu().max(1)
        lbl=np.squeeze(lbl).numpy(); preds_lbl=np.squeeze(preds_lbl).numpy()
        preds_lbl=misc.imresize(preds_lbl,lbl.shape,mode='F')

        img=img.cpu().numpy()
        img=img[0].transpose(1,2,0)
        iou,class_iou,class_ct = get_iou(preds_lbl, lbl)
        Aiou += iou
        class_Aiou += class_iou
        count_class[class_ct] += 1
        print("processing: %d/%d"%(i,len(loader)))

        lbl_0=(lbl==0)
        preds_lbl[lbl_0]=0
        ## save images
        #plt.imsave(os.path.join(out_prdlbl,"%d.png"%(i+1)),preds_lbl*4,cmap='nipy_spectral',vmin=0,vmax=(num_class-1)*4)

        t_cls=np.unique(lbl)
        if t_cls[0]==0:
            t_cls=t_cls[1:]

        mask_lbl=(lbl!=0)
        lbl=torch.from_numpy(lbl[mask_lbl]).long()
        preds_lbl = torch.from_numpy(preds_lbl[mask_lbl]).long()
        num_correct_lbl += (preds_lbl.long()==lbl.long()).sum()
        num_samples_lbl += preds_lbl.numel()

        m_acc=0
        for cls in t_cls:
            m_acc += float((preds_lbl[lbl.long()==cls].long()==cls).sum())/float((lbl.long()==cls).sum())
        Am_acc += m_acc/len(t_cls)


    acc_lbl = float(num_correct_lbl)/num_samples_lbl
    Aiou/=num_samples
    Am_acc/=num_samples
    class_Aiou[1:]/=count_class[1:]
    class_Aiou=np.where(np.isnan(class_Aiou),0,class_Aiou)

    return Aiou,Am_acc,acc_lbl,class_Aiou

def test(args):

    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True, split="test",img_size=(240,320))
    testloader = data.DataLoader(loader)

    # Setup Model
    model = Model_2b_depgd_GAP_MS(ResidualBlock, UpProj_Block, 1, num_class)
    from collections import OrderedDict
    state_dict = torch.load(args.model_path,map_location=lambda storage, location: storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
       name = k[7:]  # remove `module.`
       new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    Aiou,Am_acc,acc_lbl,class_Aiou = accuracy(model,testloader)
    print("PixAcc, mAcc, and mIoU are: %f, %f, %f"%(acc_lbl,Am_acc,np.sum(class_Aiou[1:])/float(num_class-1)))
    print("class Aiou:",class_Aiou[1:])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='models/sem_2b_FeatProp_GAP_MS_NYU40.pth',
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='NYU40',
                        help='Dataset to use')
    args = parser.parse_args()
    test(args)
