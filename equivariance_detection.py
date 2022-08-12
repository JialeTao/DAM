import os, sys
import random
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
import torch
import numpy as np
from tqdm import tqdm


from torch.utils.data import DataLoader
from frames_dataset import FramesDataset
from frames_dataset import DatasetRepeater
from logger import Visualizer

from modules.keypoint_detector import KPDetector
from modules.model import Transform

if __name__ == "__main__":

    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to dam config")
    parser.add_argument("--config_hdam", default=None, help="path to hdam config")
    parser.add_argument("--equi_threshold", default=0.25, type=float, help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.eval()
    if torch.cuda.is_available():
        # kp_detector.to(opt.device_ids[0])
        kp_detector.cuda()

    checkpoint = torch.load(opt.checkpoint, map_location='cuda:{}'.format(0))
    state_dict = checkpoint['kp_detector']
    kp_detector.load_state_dict(state_dict)

    dataset = FramesDataset(is_train=True, **config['dataset_params'])
    dataset = DatasetRepeater(dataset, 150)
    dataloader = DataLoader(dataset, batch_size=config['train_params']['batch_size'], shuffle=False, num_workers=6, drop_last=True)
    # print(dataloader.__len__())

    equivariance_list = []
    equivariance_jacobian_list = []
    # print(kp_detector.scale_factor)
    i = 0
    for x in tqdm(dataloader):
        if i >=100:
            break
        else:
            i = i+1
        x['source'] = x['source'].cuda()
        kp_source = kp_detector(x['source'])

        transform = Transform(x['source'].shape[0],**config['train_params']['transform_params'])
        transformed_frame = transform.transform_frame(x['source'][:,0:3,:,:])
        transformed_kp = kp_detector(transformed_frame)

        if config['train_params']['loss_weights']['root_motion_kp_distance'] != 0 or config['train_params']['loss_weights']['root_motion_sub_root_distance'] != 0:
            num_kp = config['model_params']['common_params']['num_kp']
            num_root_kp = config['model_params']['generator_params']['num_root_kp']
            kp_source['value'] = kp_source['value'][:,0:num_kp-num_root_kp,:]
            kp_source['jacobian'] = kp_source['jacobian'][:,0:num_kp-num_root_kp,:,:]

            transformed_kp['value'] = transformed_kp['value'][:,0:num_kp-num_root_kp,:]
            transformed_kp['jacobian'] = transformed_kp['jacobian'][:,0:num_kp-num_root_kp,:,:]


        ## Value loss part
        value = torch.abs(kp_source['value'] - transform.warp_coordinates(transformed_kp['value']))
        value = value.mean(0).mean(-1)
        value = value.unsqueeze(0)
        equivariance_value = config['train_params']['loss_weights']['equivariance_value'] * value

        ## jacobian loss part
        jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']),
                                            transformed_kp['jacobian'])
        normed_source = torch.inverse(kp_source['jacobian'])
        normed_transformed = jacobian_transformed
        value = torch.matmul(normed_source, normed_transformed)
        eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())
        value = torch.abs(eye - value).mean(0).mean(-1).mean(-1)
        value = value.unsqueeze(0)
        equivariance_jacobian = config['train_params']['loss_weights']['equivariance_jacobian'] * value

        equivariance_list.append(equivariance_value.data.cpu().numpy())
        equivariance_jacobian_list.append(equivariance_jacobian.data.cpu().numpy())

    equivariance_list = np.concatenate(equivariance_list, axis=0)
    equivariance_jacobian_list = np.concatenate(equivariance_jacobian_list, axis=0)
    equivariance = np.mean(equivariance_list, axis=0)
    equivariance_jacobian = np.mean(equivariance_jacobian_list, axis=0)
    print("equivariance loss: %s" % equivariance)
    print("equivariance_jacobian loss: %s" % equivariance_jacobian)
    # print(equivariance_list.shape)
    # print(equivariance_jacobian_list.shape)

    with open(opt.config_hdam) as f:
        config_hdam = yaml.load(f)
    ignore_kp_list = []
    for i in range(len(equivariance)):
        if equivariance[i] > opt.equi_threshold:
            ignore_kp_list.append(i)
    print( config_hdam['model_params']['kp_detector_params']['ignore_kp_list'], config_hdam['visualizer_params']['ignore_kp_list'])
    config_hdam['model_params']['kp_detector_params']['ignore_kp_list'] = ignore_kp_list
    config_hdam['visualizer_params']['ignore_kp_list'] = ignore_kp_list.copy()
    print("writing the ingnore_kp_list to the hdam config: %s" % ignore_kp_list)
    with open(opt.config_hdam, "w") as f:
        yaml.dump(config_hdam, f)
    
