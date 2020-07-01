#-*- coding:utf-8 -*-
import sys
import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import requests
import json
import cv2
from time import time
import os
import cv2
import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from resnet import resnet50


def parms():
    parser = argparse.ArgumentParser(description='CSRnet demo')
    parser.add_argument('--save_dir', type=str, default='tmp/',
                        help='Directory for detect result')
    parser.add_argument('--breath_modelpath', type=str,
                        default='rbcar_mafa_best_cs.pth', help='trained model')
    parser.add_argument('--threshold', default=0.65, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--use_cuda', default=False, type=bool,
                        help='gpu run')
    parser.add_argument('--img_dir', type=str, default='tmp/',
                        help='Directory for images')
    parser.add_argument('--file_in', type=str, default='tmp.txt',
                        help='image namesf')
    return parser.parse_args()


class BreathMask(object):
    def __init__(self,args):
        if args.use_cuda and torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False
        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        self.loadmodel(args.breath_modelpath)
        self.real_num = 0

    def loadmodel(self,modelpath):
        if self.use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        # self.net = shufflenet_v2_x1_0(pretrained=False,num_classes=6).to(device)
        self.net = resnet50(pretrained=False, num_classes=5).to(device)
        # self.net = mobilenet_v2(pretrained=False,num_classes=6).to(device)
        state_dict = torch.load(modelpath,map_location=device)
        state_dict = self.rename_dict(state_dict)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        if self.use_cuda:
            cudnn.benckmark = True
        # torch.save(self.net.state_dict(),'rbcar_best.pth')

    def rename_dict(self,state_dict):
        state_dict_new = dict()
        for key,value in list(state_dict.items()):
            state_dict_new[key[7:]] = value
        return state_dict_new
        
    def propress(self,imgs):
        # rgb_mean = np.array([123.,117.,104.])[np.newaxis, np.newaxis,:].astype('float32')
        # rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
        # rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_mean = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_std = np.array([0.225, 0.225, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        img_out = []
        for img in imgs:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            w,h = img.shape[:2]
            if w != cfg.InputSize_w or h != cfg.InputSize_h:
                img = cv2.resize(img,(cfg.InputSize_w,cfg.InputSize_h))
            img = img.astype('float32')
            img /= 255.0
            img -= rgb_mean
            img /= rgb_std
            img = np.transpose(img,(2,0,1))
            img_out.append(img)
        return np.array(img_out)

    def inference(self,imglist):
        t1 = time.time()
        imgs = self.propress(imglist)
        bt_img = torch.from_numpy(imgs)
        if self.use_cuda:
            bt_img = bt_img.cuda()
        output = self.net(bt_img)
        pred_cls = torch.argmax(output,dim=1)
        scores = [output[idx,pred_cls[idx]].data.cpu().numpy() for idx in range(output.size(0))]
        t2 = time.time()
        # print('consuming:',t2-t1)
        return scores, pred_cls.data.cpu().numpy()


def crop_body_by_file(src):
    r = requests.post('http://192.168.1.105:20630/v1/algo/body/detect', files={'src': open(src, 'rb'),
                                                                               'classes': (None, 'truck')})
    if r.status_code == 200:
        return json.loads(r.content)
    else:
        raise Exception('failed ' + str(r.status_code))


if __name__ == '__main__':
    args = parms()
    model = BreathMask(args)

    # root = '/media/changshuang/My Book/StreeData'
    # dir_names = os.listdir(root)
    # num = 0
    # for dir_name in dir_names:
    #     cur_dir = os.path.join(root, dir_name)
    #     sub_dirs = os.listdir(cur_dir)
    #     for sub_dir in sub_dirs:
    #         img_dirs = os.path.join(cur_dir, sub_dir)
    #         img_names = os.listdir(img_dirs)
    #
    #         for img_name in img_names:
    #             img_path = os.path.join(img_dirs, img_name)
    #             print(img_path)
    #             data_dict = crop_body_by_file(img_path)
    #             img = cv2.imread(img_path)
    #             datas = data_dict['data']
    #
    #             if datas:
    #                 for data in datas:
    #                     h = data['height']
    #                     l = data['left']
    #                     t = data['top']
    #                     w = data['width']
    #
    #                     if min(h, w) >= 30:
    #                         dec_img = img[t: t + h + 1, l: l + w + 1]
    #                         score, pre_cls = model.inference(np.array([dec_img]))
    #                         print(pre_cls)
    #                         # cv2.imshow('img', dec_img)
    #                         # cv2.waitKey()
    #
    #                         if pre_cls[0] == 1:
    #                             # dec_img = cv2.resize(dec_img, (224, 224))
    #                             cv2.imwrite('/media/changshuang/My Book/ExtractZhatuData/{:0>8}.jpg'.format(num), img)
    #                             num += 1
    #                             break

img_path = '/media/changshuang/My Book/DetData/00000006.jpg'
data_dict = crop_body_by_file(img_path)
img = cv2.imread(img_path)
datas = data_dict['data']

if datas:
    for data in datas:
        h = data['height']
        l = data['left']
        t = data['top']
        w = data['width']
        dec_img = img[t: t + h + 1, l: l + w + 1]
# img = cv2.imread('/media/changshuang/My Book/DetData/00000007.jpg')
        score, pre_cls = model.inference(np.array([dec_img]))
        cv2.imshow('img', dec_img)
        print(pre_cls)
        cv2.waitKey()