import os
from tqdm import tqdm
import argparse
import numpy as np
import imageio
from imageio import imread
import random
## input path video of concatenated pngs format
## output all video frames of png format
parser = argparse.ArgumentParser()
parser.add_argument('--video_path',help='path to the mp4 video folder', type=str)
parser.add_argument('--output_path', help='path to the output png frames', type=str)
parser.add_argument('--sample_portion', help='portion of sampling frames', type=float, default=1.0)
args = parser.parse_args()

video_path = args.video_path
output_path = args.output_path
if not os.path.exists(output_path):
    os.makedirs(output_path)

video_list = os.listdir(video_path)
processed_video_list = os.listdir(output_path)
# random.shuffle(video_list)
# video_list_sub = video_list[0:1000]
for video in tqdm(video_list):
    if video in processed_video_list:
        continue
    # video_name = video.split('.')[0]
    # print('processing video {}'.format(video_name))
    video_file = os.path.join(video_path, video)
    try:
        reader = imageio.get_reader(video_file)
    except OSError:
        print('a video can not be loaded: %s' % video_file)
        continue

    if not os.path.exists(os.path.join(output_path, video)):
        os.makedirs(os.path.join(output_path, video))

    # video_array = []
    i = 0
    try:
        for im in reader:
            # img_name = video_name + '_' + str(i).zfill(5) + '.png'
            img_name = str(i).zfill(5) + '.png'
            i+=1
            assert im.shape == (256,256,3)
            imageio.imsave(os.path.join(output_path, video, img_name),im=im)
            # video_array.append(im)
            # print(len(video_array))
    except RuntimeError:
        pass
    # random.shuffle(video_array)
    # # video_array = np.array(video_array)
    # video_len = len(video_array)
    # # print(len(video_array))
    # i = 0
    # for im in video_array:
        # img_name = video_name + '_' + str(i).zfill(5) + '.png'
        # i+=1
        # if i < video_len * args.sample_portion:
            # imageio.imsave(os.path.join(output_path, img_name),im=im)
        # else:
    #         break
  #   total+=1
    # if total == 1000:
  #       break



