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
                        default='../networks/rbcar_mafa_best_xiaoyu.pth', help='trained model')
    parser.add_argument('--threshold', default=0.65, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--use_cuda', default=False, type=bool,
                        help='gpu run')
    parser.add_argument('--img_dir', type=str, default='tmp/',
                        help='Directory for images')
    parser.add_argument('--file_in', type=str, default='tmp.txt',
                        help='image namesf')
    return parser.parse_args()


if __name__ == '__main__':
    args = parms()
    model = BreathMask(args)

    log = setup_logger('../../logs/log.txt')

    root = '/media/changshuang/My Book/ShangYu_videos/short video'
    path = '/media/changshuang/My Book/ShangYu_Extract_Save'

    video_anno = os.path.join(os.path.dirname(__file__), '../../video/ShangYu_short_video_path.txt')
    with open(video_anno, 'r') as f:
        video_annos = f.readlines()
        for video_index in range(len(video_annos)):
            video_path = os.path.join(root, video_annos[video_index].strip())
            if '#' in video_path:
                print(video_path)
                continue
            log.info(video_path)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                log.info("===================don't playing================")
            else:
                total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                print('-------------------total_frame------------------', total_frame)
                frame_index = 0
                while frame_index < total_frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()
                    if ret:
                        if frame_index % 100 == 0:
                            print('##############cur frame index value ({})##############'.format(frame_index))
                        if frame_index % 1 == 0:
                            data_dict = detection_by_image(frame)
                            datas = data_dict['data']
                            if datas:
                                for box_index, data in enumerate(datas):
                                    h = data['height']
                                    l = data['left']
                                    t = data['top']
                                    w = data['width']

                                    dec_img = frame[t: t + h + 1, l: l + w + 1]
                                    score, pre_cls = model.inference(np.array([dec_img]))
                                    # cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 0, 255), 2)

                                    if pre_cls[0] == 1:
                                        save_path = os.path.join(path, 'video_pos' + str(video_index))
                                        if not os.path.exists(save_path):
                                            os.makedirs(save_path)

                                        dec_img = cv2.resize(dec_img, (224, 224))
                                        cv2.imwrite(save_path + '/v{}_f{}_b{}.jpg'.format(video_index, frame_index, box_index), dec_img)
                                        # log.info('************save file [{:0>8}.jpg]************'.format(frame_index))
                                    else:
                                        save_path = os.path.join(path, 'video_neg' + str(video_index))
                                        if not os.path.exists(save_path):
                                            os.makedirs(save_path)

                                        dec_img = cv2.resize(dec_img, (224, 224))
                                        cv2.imwrite(save_path + '/v{}_f{}_b{}.jpg'.format(video_index, frame_index, box_index), dec_img)
                                        # break
                        # cv2.imshow('frame', frame)
                        if cv2.waitKey(1) & 0xff == 27:
                            break
                        frame_index += 1
                    else:
                        log.info("don't playing " + str(frame_index))

        cap.release()
