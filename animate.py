import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from frames_dataset import PairedDataset
from logger import Logger, Visualizer
import imageio
from scipy.spatial import ConvexHull
import numpy as np

# from sync_batchnorm import DataParallelWithCallback


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new


def animate(config, generator, kp_detector, checkpoint, log_dir, dataset):
    log_dir = os.path.join(log_dir, 'animation')
    # png_dir = os.path.join(log_dir, 'png')
    animate_params = config['animate_params']

    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=animate_params['num_pairs'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='animate'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # if not os.path.exists(png_dir):
        # os.makedirs(png_dir)

    # if torch.cuda.is_available():
        # generator = DataParallelWithCallback(generator)
        # kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            predictions = []
            visualizations = []

            if torch.cuda.is_available():
                x['driving_video'] = x['driving_video'].cuda()
                x['source_video'] = x['source_video'].cuda()
            driving_video = x['driving_video']
            source_frame = x['source_video'][:, :, 0, :, :]

            kp_source = kp_detector(source_frame)
            kp_driving_initial = kp_detector(driving_video[:, :, 0])

            for frame_idx in range(driving_video.shape[2]):
                driving_frame = driving_video[:, :, frame_idx]
                kp_driving = kp_detector(driving_frame)
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                       kp_driving_initial=kp_driving_initial, **animate_params['normalization_params'])
                # out = generator(source_frame, kp_source=kp_source, kp_driving=kp_norm)

                if kp_detector.subroot_leaf_attention:
                    num_kp = config['model_params']['common_params']['num_kp']
                    num_root_kp = config['model_params']['generator_params']['num_root_kp']
                    prior_kp_list = []
                    subroot_leaf_attention = kp_detector.attention_block(sub_root_kp=kp_driving['value'][:,num_kp-num_root_kp:num_kp-1,:], leaf_kp=kp_driving['value'][:,0:num_kp-num_root_kp,:], feature_map=kp_driving['feature_map'])
                    subroot_leaf_attention = subroot_leaf_attention[0]
                    for i in range(subroot_leaf_attention.size(0)):
                        # index n * 1
                        subroot_leaf_index = torch.nonzero(subroot_leaf_attention[i] > 0.1)
                        subroot_leaf_index = subroot_leaf_index[:, 0].data.cpu().numpy().tolist()
                        # print(subroot_leaf_index)
                        prior_kp_list.append(subroot_leaf_index)

                    # print(subroot_leaf_attention)
                    # continue
                else:
                    prior_kp_list = config['train_params']['prior_kp_list']
                config['visualizer_params']['prior_kp_list'] = prior_kp_list

                if config['train_params']['loss_weights']['root_motion_kp_distance'] != 0 or config['train_params']['loss_weights']['root_motion_sub_root_distance'] != 0:
                    num_kp = config['model_params']['common_params']['num_kp']
                    num_root_kp = config['model_params']['generator_params']['num_root_kp']
                    kp_source_for_motion = {}
                    kp_source_for_motion['value'] = kp_source['value'][:,0:num_kp-num_root_kp,:]
                    kp_source_for_motion['jacobian'] = kp_source['jacobian'][:,0:num_kp-num_root_kp,:,:]
                    kp_driving_for_motion = {}
                    kp_driving_for_motion['value'] = kp_norm['value'][:,0:num_kp-num_root_kp,:]
                    kp_driving_for_motion['jacobian'] = kp_norm['jacobian'][:,0:num_kp-num_root_kp,:,:]
                    out = generator(source_frame, kp_source=kp_source_for_motion, kp_driving=kp_driving_for_motion)
                else:
                    out = generator(source_frame, kp_source=kp_source, kp_driving=kp_norm)

                out['kp_driving'] = kp_driving
                out['kp_source'] = kp_source
                out['kp_norm'] = kp_norm

                del out['sparse_deformed']

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize(source=source_frame,
                                                                                    driving=driving_frame, out=out)
                visualization = visualization
                visualizations.append(visualization)

            # predictions = np.concatenate(predictions, axis=1)
            result_name = "-".join([x['driving_name'][0], x['source_name'][0]])
            # imageio.imsave(os.path.join(png_dir, result_name + '.png'), (255 * predictions).astype(np.uint8))

            image_name = result_name + animate_params['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)
