# -*-coding:utf-8-*-
import requests
import json
import cv2


def detection_by_path(src):
    r = requests.post('http://192.168.1.105:20630/v1/algo/body/detect',
                      files={'src': open(src, 'rb'),
                             'classes': (None, 'person,bicycle,car,motorbike,bus,truck')})
    if r.status_code == 200:
        return json.loads(r.content)
    else:
        raise Exception('failed ' + str(r.status_code))


def detection_by_image(src):
    r = requests.post('http://192.168.1.105:20630/v1/algo/body/detect',
                      files={'src': (None, cv2.imencode('.jpg', src)[1]),
                             'classes': (None, 'truck')})
    if r.status_code == 200:
        return json.loads(r.content)
    else:
        raise Exception('failed ' + str(r.status_code))