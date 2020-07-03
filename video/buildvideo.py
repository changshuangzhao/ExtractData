# -*-coding:utf-8-*-
import os
# root = '/media/changshuang/My Book/ShangYu_videos/short video'
# root = '/Users/yanyan/Desktop/ShangYu_videos/long video'
root = '/Users/yanyan/Desktop/ShangYu_videos/short video 1'

video_names = sorted(os.listdir(root))
with open('ShangYu_short_video_1_path.txt', 'w') as f:
    for i in range(len(video_names)):
        if '.DS_Store' in video_names[i]:
            continue
        f.writelines(video_names[i] + '\n')
