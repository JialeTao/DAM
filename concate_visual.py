import imageio
import os
from tqdm import tqdm
from skimage.transform import resize
import glob
import numpy as np
# custom_path  = 'log_fashion_att0.8/reconstruction'
# fomm_path = 'checkpoints/reconstruction-fashion'
# out_path = custom_path + '-cmp-fomm'
path1 = 'CVPR2022_HDAM/taichi/log_taichi-256-HDAM-weight1-weight10-subkp3-attention-value0_2-bg-motion-skips-perceptual-aug10-clip-grad1/visualization'
path2= 'taichi/20220227_DAM_bs64_root_motion_rot10_single_jacob_clip_grad1_bg_skips/visualization'
path3='../articulated-animation/checkpoints/taichi256/visualization'
out_path=path1+'_cmp_root_motion_pca'
if not os.path.exists(out_path):
    os.makedirs(out_path)
# custom_path = glob.glob(os.path.join(custom_path, '*.mp4'))
# fomm_path = glob.glob(os.path.join(fomm_path, '*.mp4'))
path1 = glob.glob(os.path.join(path1, '*.mp4'))
path2 = glob.glob(os.path.join(path2, '*.mp4'))
path3 = glob.glob(os.path.join(path3, '*.mp4'))
i = 0
# for img_custom, img_fomm in tqdm(zip(custom_path, fomm_path)):
for img1, img2, img3 in tqdm(zip(path1,path2, path3)):
    reader1 = imageio.get_reader(img1)
    reader2 = imageio.get_reader(img2)
    reader3 = imageio.get_reader(img3)
    frames = []
    try:
        for frame1, frame2, frame3 in zip(reader1, reader2,reader3):
            frame = np.concatenate((frame1,frame2,frame3), axis=1)
            frames.append(frame)
    except:
        pass
    video = np.array(frames)
    imageio.mimsave(os.path.join(out_path, os.path.basename(img1)),video)
    # reader_custom = imageio.get_reader(img_custom)
    # reader_fomm = imageio.get_reader(img_fomm)
    # frames = []
    # try:
       # for frame_custom, frame_fomm in zip(reader_custom, reader_fomm):
           # frame = np.concatenate((frame_custom[:,0:512,:],frame_fomm[:,768:1024,:],frame_custom[:,768:1024,:],frame_fomm[:,1024:1280,:],frame_custom[:,1024:1280,:]), axis=1)
           # # print(frame.shape)
           # frames.append(frame)
    # except:
       # pass
    # # print(len(frames))
    # reader_custom.close()
    # reader_fomm.close()
    # video = np.array(frames)
    # # print(video.shape)
    # imageio.mimsave(os.path.join(out_path, os.path.basename(img_custom)), video)
    # # print(frames[0].shape)
    # # os.makedirs(os.path.join(d_out, image))
    # # [imageio.imsave(os.path.join(d_out, image, str(i).zfill(7) + '.png'), frame) for i, frame in enumerate(frames)]
