# -*- coding:utf-8 -*-
import sys
import os
import cv2
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../networks'))
from breathmask import BreathMask
sys.path.append(os.path.join(os.path.dirname(__file__), '../detection'))
from detect import detection_by_image
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from logger import setup_logger


def parms():
    parser = argparse.ArgumentParser(description='CSRnet demo')
    parser.add_argument('--save_dir', type=str, default='tmp/',
                        help='Directory for detect result')
    parser.add_argument('--breath_modelpath', type=str,
                        default='../networks/rbcar_mafa_best_cs.pth', help='trained model')
    parser.add_argument('--threshold', default=0.65, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--use_cuda', default=False, type=bool,
                        help='gpu run')
    parser.add_argument('--img_dir', type=str, default='tmp/',
                        help='Directory for images')
    parser.add_argument('--file_in', type=str, default='tmp.txt',
                        help='image namesf')
    return parser.parse_args()


args = parms()
model = BreathMask(args)
log = setup_logger('logs/log1.txt')
frame = cv2.imread('/home/changshuang/Desktop/1.jpg')

data_dict = detection_by_image(frame)
datas = data_dict['data']
num = 0
if datas:
    for data in datas:
        h = data['height']
        l = data['left']
        t = data['top']
        w = data['width']

        if min(h, w) >= 30:
            dec_img = frame[t: t + h + 1, l: l + w + 1]
            score, pre_cls = model.inference(np.array([dec_img]))
            cv2.imshow('frame', dec_img)
            cv2.waitKey()
            if pre_cls[0] == 1:
                cv2.imwrite('/home/changshuang/Desktop' + '/{:0>8}.jpg'.format(num), frame)
                log.info('************save file [{:0>8}.jpg]************'.format(num))
                num += 1
                break