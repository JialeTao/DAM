import matplotlib

matplotlib.use('Agg')

import os, sys
import random
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
import torch
import torch.distributed as dist
import numpy as np
import pickle
import pandas as pd

from frames_dataset import FramesDataset

from modules.generator import OcclusionAwareGenerator
from modules.discriminator import MultiScaleDiscriminator
from modules.keypoint_detector import KPDetector
from modules.bg_motion_predictor import BGMotionPredictor


from train import train
from reconstruction import reconstruction
from animate import animate
from visualization import visualization

import datetime
import time

if __name__ == "__main__":


    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate", "visualization"])
    parser.add_argument("--log_dir", default='./log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.add_argument("--bg_predictor", default=None, help="if use bg_predictor")
    parser.add_argument("--local_rank", default=-1, type=int, help="distributed machine")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    distributed = opt.local_rank >= 0
    if distributed:
        # os.environ["NCCL_BLOCKING_WAIT"] = "1"
        # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(0, 18000))

        rank = torch.distributed.get_rank()
        local_rank = opt.local_rank if opt.local_rank != -1 else torch.cuda.current_device()
        device = torch.device('cuda:{}'.format(local_rank))
        torch.cuda.set_device(device)
        print(local_rank, rank, device, torch.cuda.device_count(), torch.cuda.current_device())
        world_size = torch.distributed.get_world_size()
        print("world_size : ", world_size)

        config['train_params']['batch_size'] = int(config['train_params']['batch_size'] / world_size)
    else:
        device = torch.device('cuda:{}'.format(opt.device_ids[0]))
        world_size=1

    if opt.checkpoint is not None and opt.mode != "train":
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = opt.log_dir + '_' + os.path.basename(opt.config).split('.')[0]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    if torch.cuda.is_available():
        generator.to(device)
    if opt.verbose:
        print(generator)

    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])
    if torch.cuda.is_available():
        discriminator.to(device)
    if opt.verbose:
        print(discriminator)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if torch.cuda.is_available():
        kp_detector.to(device)
    if opt.verbose:
        print(kp_detector)

    if opt.bg_predictor is not None:
        bg_predictor = BGMotionPredictor(num_channels=config['model_params']['common_params']['num_channels'],
                                        **config['model_params']['bg_predictor_params'])
        if torch.cuda.is_available():
            bg_predictor.to(device)
    else:
        bg_predictor=None

    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])

    if opt.mode == 'visualization':
        visualization(config, generator, kp_detector, opt.checkpoint, log_dir, bg_predictor=bg_predictor)

    if opt.mode == 'train':
        print("Training...")
        train(config, generator, discriminator, kp_detector, opt.checkpoint, log_dir, dataset, opt.device_ids, bg_predictor=bg_predictor, local_rank=rank, world_size=world_size)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(config, generator, kp_detector, opt.checkpoint, log_dir, dataset, bg_predictor=bg_predictor)
    elif opt.mode == 'animate':
        print("Animate...")
        animate(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)
