import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from logger import Logger, Visualizer
import numpy as np
import imageio
# from sync_batchnorm import DataParallelWithCallback


def reconstruction(config, generator, kp_detector, checkpoint, log_dir, dataset, bg_predictor=None, distributed=False):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector, bg_predictor=bg_predictor)
    else:
        print('warining: reconstruction without checkpoiont, make sure you are using the trained models...')
        # raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    if distributed:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    generator.eval()
    kp_detector.eval()
    if bg_predictor is not None:
        bg_predictor.eval()
    loss_list = []

    from frames_dataset import read_video
    loss_list_bg = []
    loss_list_fg = []
    for it, x in tqdm(enumerate(dataloader)):

        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with torch.no_grad():
            predictions = []
            visualizations = []
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()
            kp_source = kp_detector(x['video'][:, :, 0])
            for frame_idx in range(x['video'].shape[2]):
                source = x['video'][:, :, 0]
                driving = x['video'][:, :, frame_idx]
                kp_driving = kp_detector(driving)

                if bg_predictor is not None:
                    bg_params = bg_predictor(source,driving)
                else:
                    bg_params = None

                if kp_detector.subroot_leaf_attention:
                    num_kp = config['model_params']['common_params']['num_kp']
                    num_root_kp = config['model_params']['generator_params']['num_root_kp']
                    prior_kp_list = []
                    subroot_leaf_attention = kp_detector.attention_block(sub_root_kp=kp_driving['value'][:,num_kp-num_root_kp:num_kp-1,:], leaf_kp=kp_driving['value'][:,0:num_kp-num_root_kp,:], feature_map=kp_driving['feature_map'])
                    subroot_leaf_attention = subroot_leaf_attention[0]
                    for i in range(subroot_leaf_attention.size(0)):
                        prior_kp_list.append([])
                    for i in range(subroot_leaf_attention.size(1)):
                        if kp_detector.ignore_kp_list is None or i not in kp_detector.ignore_kp_list:
                            prior_kp_list[torch.argmax(subroot_leaf_attention[:,i])].append(i)
                else:
                    prior_kp_list = config['train_params']['prior_kp_list']
                config['visualizer_params']['prior_kp_list'] = prior_kp_list

                if config['train_params']['loss_weights']['root_motion_kp_distance'] != 0 or config['train_params']['loss_weights']['root_motion_sub_root_distance'] != 0:
                    num_kp = config['model_params']['common_params']['num_kp']
                    num_root_kp = config['model_params']['generator_params']['num_root_kp']

                    if not config['model_params']['generator_params']['root_motion']:
                        kp_source_for_motion = {}
                        kp_source_for_motion['value'] = kp_source['value'][:,0:num_kp-num_root_kp,:]
                        kp_source_for_motion['jacobian'] = kp_source['jacobian'][:,0:num_kp-num_root_kp,:,:]
                        kp_driving_for_motion = {}
                        kp_driving_for_motion['value'] = kp_driving['value'][:,0:num_kp-num_root_kp,:]
                        kp_driving_for_motion['jacobian'] = kp_driving['jacobian'][:,0:num_kp-num_root_kp,:,:]
                        out = generator(source, kp_source=kp_source_for_motion, kp_driving=kp_driving_for_motion, bg_params=bg_params)
                    else:
                        out = generator(source, kp_source=kp_source, kp_driving=kp_driving, bg_params=bg_params)
                else:
                    out = generator(source, kp_source=kp_source, kp_driving=kp_driving, bg_params=bg_params)
                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving
                out['kp_norm'] = kp_driving
                del out['sparse_deformed']
                del out['deformed']
                # del out['occlusion_map']
                # del out['prediction']
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize(source=source,
                                                                                    driving=driving, out=out)
                visualizations.append(visualization)
                loss_list.append(torch.abs(out['prediction'] - driving).mean().cpu().numpy())

            predictions = np.concatenate(predictions, axis=1)
            imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * predictions).astype(np.uint8))

            image_name = x['name'][0] + config['reconstruction_params']['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

    # print(len(loss_list))
    print("Reconstruction loss: %s" % np.mean(loss_list))
    return loss_list

