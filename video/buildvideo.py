# -*-coding:utf-8-*-
import os
# root = '/media/changshuang/My Book/ShangYu_videos/short video'
# root = '/Users/yanyan/Desktop/ShangYu_videos/long video'
root = '/Users/yanyan/Desktop/zhatuche_test_video'

video_names = sorted(os.listdir(root))
print(len(video_names))
with open('zhatuche_test_video_path.txt', 'w') as f:
    for i in range(len(video_names)):
        f.writelines(str(video_names[i]) + '\n')
