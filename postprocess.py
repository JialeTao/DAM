import os
from tqdm import tqdm
import argparse
import numpy as np
import imageio
from imageio import imread

## input path video of concatenated pngs format
## output all video frames of png format
parser = argparse.ArgumentParser()
parser.add_argument('--png_video_path',help='path to the png video folder', type=str)
parser.add_argument('--output_path', help='path to the output png video frames', type=str)
args = parser.parse_args()

png_video_path = args.png_video_path
output_path = args.output_path
if not os.path.exists(output_path):
    os.makedirs(output_path)

png_video_list = os.listdir(png_video_path)
for png_video in tqdm(png_video_list):
    video_name = png_video.split('.')[0]
    # print('processing video {}'.format(video_name))
    png_video_file = os.path.join(png_video_path, png_video)
    png_video_array = imread(png_video_file)
    # png_video_array = png_video_arra0'meta']
    # print(png_video_array.shape)
    _, n, c = png_video_array.shape
    n_frames = int(n / 256)
    # print('total {} frames'.format(n_frames))
    for i in range(n_frames):
        frame_start = i * 256
        frame_end = (i+1) * 256
        img_i = png_video_array[:,frame_start:frame_end,:]
        assert img_i.shape == (256,256,3)
        img_name = video_name + '_' + str(i).zfill(5) + '.png'
        imageio.imsave(os.path.join(output_path, img_name),im=img_i)



