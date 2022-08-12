import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from logger import Logger, Visualizer
import numpy as np
import imageio
import cv2
# from sync_batchnorm import DataParallelWithCallback


def draw_affine(img, shift, affine):
    img = img * 255
    draw_ratio = 15
    for i,(s,a) in enumerate(zip(shift, affine)):
        s = (s+1) / 2 * 255
        # print(a)
        scale = torch.norm(a,p=2,dim=1,keepdim=True)
        a_unit = a / scale
        scale_ratio = scale[1,0] / scale[0,0]

        start = s
        end1 = s+a_unit[0,:] * draw_ratio
        end2 = s+a_unit[1,:] * draw_ratio * scale_ratio
        # start = (start+1) / 2 * 255
        # end1 = (end1+1) / 2 * 255
        # end2 = (end2+1) / 2 * 255
        img = cv2.arrowedLine(img,pt1=tuple(start), pt2=tuple(end1), color=(0, 0, 255), thickness=2)
        img = cv2.arrowedLine(img,pt1=tuple(start), pt2=tuple(end2), color=(0, 0, 255), thickness=2)

        if i > 9:
            p1 = (end1-s) + (end2-s) + s
            p2 = (end1-p1)*2 + p1
            p4 = (end2-p1)*2 + p1
            p3 = (s-p1)*2 + p1
            p1,p2,p3,p4 = tuple(p1),tuple(p2),tuple(p3),tuple(p4)
            img=cv2.line(img, pt1=p1, pt2=p2, color=(255, 0, 0), thickness=2)
            img=cv2.line(img, pt1=p1, pt2=p4, color=(255, 0, 0), thickness=2)
            img=cv2.line(img, pt1=p2, pt2=p3, color=(255, 0, 0), thickness=2)
            img=cv2.line(img, pt1=p3, pt2=p4, color=(255, 0, 0), thickness=2)
    return img / 255


def visualization(config, generator, kp_detector, checkpoint, log_dir, bg_predictor=None):
    log_dir = os.path.join(log_dir, 'visualization')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector, bg_predictor=bg_predictor)
    else:
        print('warining: reconstruction without checkpoiont, make sure you are using the trained models...')
        # raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    # if distributed:
        # sampler = DistributedSampler(dataset)
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False, sampler=sampler)
    # else:
    #     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    generator.eval()
    kp_detector.eval()
    if bg_predictor is not None:
        bg_predictor.eval()

    from frames_dataset import read_video
    file_name = 'data/taichi-mp4/test/A3ZmT97hAWU#007482#007618.mp4'
    video_name = 'A3ZmT97hAWU#007482#007618.mp4'
    video = read_video(file_name, frame_shape=(256,256),read_first_frame=False)
    # print(video.shape)
    result_video = []
    for img in video:
        # print(img.shape)
        frame = torch.from_numpy(img).cuda().permute(2,0,1).unsqueeze(0)
        kp = kp_detector(frame)
        shift = kp['value'][0,:,:]
        affine = kp['jacobian'][0,:,:,:]
        result_img = draw_affine(img, shift, affine)
        result_video.append(result_img)
    # print(result_img.shape, type(result_img))
    if len(result_video) == 1:
        imageio.imsave(os.path.join(log_dir, video_name+'.png'), result_video[0])
    else:
        imageio.mimsave(os.path.join(log_dir, video_name), result_video)



