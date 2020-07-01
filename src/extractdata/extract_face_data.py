# -*-coding:utf-8-*-
import requests
import json
import cv2
from time import time
import os


def crop_body_by_file(src):
    r = requests.post('http://192.168.1.105:20630/v1/algo/body/detect', files={'src': open(src, 'rb'),
                                                                               'classes': (None, 'person,bicycle,car,motorbike,bus,truck')})
    if r.status_code == 200:
        return json.loads(r.content)
    else:
        raise Exception('failed ' + str(r.status_code))


if __name__ == '__main__':
    root = '/media/changshuang/My Book/FaceData'
    dir_names = os.listdir(root)
    num = 0
    for dir_name in dir_names:
        cur_dir = os.path.join(root, dir_name)
        sub_dirs = os.listdir(cur_dir)
        for sub_dir in sub_dirs:
            img_dirs = os.path.join(cur_dir, sub_dir)
            img_names = os.listdir(img_dirs)
            print(img_dirs)
            # pre_count = 0
            for img_name in img_names:
                img_path = os.path.join(img_dirs, img_name)
                # print(img_path)
                data_dict = crop_body_by_file(img_path)
                img = cv2.imread(img_path)
                datas = data_dict['data']

                # pos_conut = len(datas)
                # if pre_conut == pos_conut:
                #     continue
                # else:
                #     pre_conut = pos_conut

                if datas:
                    for data in datas:
                        h = data['height']
                        l = data['left']
                        t = data['top']
                        w = data['width']
                        if min(h, w) >= 30:
                            cv2.imwrite('/media/changshuang/My Book/ExtractFaceData/{:0>8}.jpg'.format(num), img)
                            num += 1
                            break
            #               cv2.rectangle(img, (l, t), (l + w, t + h), (0, 0, 255), 2)
                # cv2.imshow('img', img)
                # if cv2.waitKey() == 27:
                #     exit()