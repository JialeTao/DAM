import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
from skimage.draw import circle, line

import matplotlib.pyplot as plt
import collections
import cv2


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            try:
                os.makedirs(self.visualizations_dir)
            except FileExistsError:
                pass
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        print(loss_string)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    def save_cpk(self, emergent=False, models=None):
        if models is not None:
            cpk = {k: v.state_dict() for k, v in models.items()}
        else:
            cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num))
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, discriminator=None, kp_detector=None,bg_predictor=None,
                 optimizer_generator=None, optimizer_discriminator=None, optimizer_kp_detector=None):
        checkpoint = torch.load(checkpoint_path)
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if bg_predictor is not None:
            bg_predictor.load_state_dict(checkpoint['bg_predictor'])
            print('load bg_predictor params success')
        if discriminator is not None:
            try:
               discriminator.load_state_dict(checkpoint['discriminator'])
            except:
               print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except RuntimeError as e:
                print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])

        return checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)


class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow', draw_line=False, sub_kp_num=[2,2,3,3],
                 num_kp=14, num_root_kp=4, prior_kp_list=None, ignore_kp_list=[]):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)
        self.draw_line = draw_line
        self.sub_kp_num = sub_kp_num
        self.num_root_kp = num_root_kp
        self.num_kp = num_kp
        self.prior_kp_list = prior_kp_list if prior_kp_list is not None else [[0, 1], [2, 3], [4, 5, 6], [7, 8, 9]]
        self.ignore_kp_list = ignore_kp_list

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        image = image.copy()
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        num_root_kp = self.num_root_kp
        # print(kp_array)
        leaf_start = 0
        kp_ind = 0

        num_kp = self.num_kp
        for kp_ind, kp in enumerate(kp_array):
            if kp_ind == num_kp-1:
                rr, cc = circle(kp[1], kp[0], self.kp_size*2, shape=image.shape[:2])
                image[rr, cc] = np.array(self.colormap(kp_ind / num_kp ))[:3]
                # draw line between root kp and sub_root kp
                if self.draw_line:
                    for i in range(len(self.prior_kp_list)):
                        kp_leaf = kp_array[kp_ind-i-1,  :]
                        rr_line, cc_line = line(int(kp[1]), int(kp[0]), int(kp_leaf[1]), int(kp_leaf[0]))
                        rr_line[rr_line>255] = 255
                        cc_line[cc_line>255] = 255
                        image[rr_line, cc_line] = np.array(self.colormap(kp_ind / num_kp))[:3]
                # draw line between root kp and sub_root kp
            elif kp_ind >= num_kp-self.num_root_kp and kp_ind < num_kp-1:
                # continue
                rr, cc = circle(kp[1], kp[0], self.kp_size*1.5, shape=image.shape[:2])
                image[rr, cc] = np.array(self.colormap(kp_ind / num_kp ))[:3]
                # draw line between sub_root kp and motion kp
                if self.draw_line:
                    leaf_index = kp_ind - (self.num_kp - self.num_root_kp)
                    for i in self.prior_kp_list[leaf_index]:
                        kp_leaf = kp_array[i, :]
                        rr_line, cc_line = line(int(kp[1]), int(kp[0]), int(kp_leaf[1]), int(kp_leaf[0]))
                        rr_line[rr_line>255] = 255
                        cc_line[cc_line>255] = 255
                        image[rr_line, cc_line] = np.array(self.colormap(kp_ind / num_kp))[:3]
                # draw line between sub_root kp and motion kp
            elif kp_ind < num_kp-self.num_root_kp:
                if kp_ind in self.ignore_kp_list:
                    continue
                rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
                image[rr, cc] = np.array(self.colormap(kp_ind / num_kp ))[:3]
                # # draw number around kp
                # # print(image.shape)
                # # image = image.transpose(1, 0, 2,).copy()
                # image = (255 * image).copy()
                # image = cv2.putText(image.astype(np.uint8), str(kp_ind), (int(kp[0]), int(kp[1])), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255))
                # image = image / 255
            # elif kp_ind > num_kp-1:
                # # print(self.colormap((kp_ind-11) / num_kp))
                # if kp_ind - num_kp in self.ignore_kp_list:
                    # continue
                # color = self.colormap((kp_ind-num_kp) / num_kp)
                # min_x = int(kp[0] - self.kp_size)
                # min_y = int(kp[1] - self.kp_size)
                # max_x = min_x + self.kp_size * 2
                # max_y = min_y + self.kp_size * 2
                # image = (255 * image).copy()
                # image = cv2.rectangle(image.astype(np.uint8), (min_x, min_y), (max_x, max_y), (color[0]*255, color[1]*255,color[2]*255), thickness=-1)
                # image = image / 255.0
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, driving, source, out):
        images = []

        # Source image with keypoints
        source = source.data.cpu()
        kp_source = out['kp_source']['value'].data.cpu().numpy()
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))
        images.append(source)

        # Driving image with keypoints
        kp_driving = out['kp_driving']['value'].data.cpu().numpy()
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))
        images.append(driving)

        # Deformed image
        if 'deformed' in out:
            deformed = out['deformed'].data.cpu().numpy()
            deformed = np.transpose(deformed, [0, 2, 3, 1])
            images.append(deformed)

        # Result with and without keypoints
        if 'prediction' in out:
            prediction = out['prediction'].data.cpu().numpy()
            prediction = np.transpose(prediction, [0, 2, 3, 1])
            if 'kp_norm' in out:
                kp_norm = out['kp_norm']['value'].data.cpu().numpy()
                images.append((prediction, kp_norm))
            images.append(prediction)


        ## Occlusion map
        if 'occlusion_map' in out:
            occlusion_map = out['occlusion_map'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)

        ## local mask
        if 'mask' in out:
            mask_map = out['mask'].data.cpu()
            for i in range(mask_map.shape[1]):
                mask_i = mask_map[:,i:i+1,:,:].repeat(1,3,1,1)
                mask_i= F.interpolate(mask_i, size=source.shape[1:3]).numpy()
                mask_i= np.transpose(mask_i, [0, 2, 3, 1])
                images.append(mask_i)

        # Deformed images according to each individual transform
        if 'sparse_deformed' in out:
            full_mask = []
            for i in range(out['sparse_deformed'].shape[1]):
                image = out['sparse_deformed'][:, i].data.cpu()
                image = F.interpolate(image, size=source.shape[1:3])
                mask = out['mask'][:, i:(i+1)].data.cpu().repeat(1, 3, 1, 1)
                mask = F.interpolate(mask, size=source.shape[1:3])
                image = np.transpose(image.numpy(), (0, 2, 3, 1))
                mask = np.transpose(mask.numpy(), (0, 2, 3, 1))

                if i != 0:
                    color = np.array(self.colormap((i - 1) / (out['sparse_deformed'].shape[1] - 1)))[:3]
                else:
                    color = np.array((0, 0, 0))

                color = color.reshape((1, 1, 1, 3))

                images.append(image)
                if i != 0:
                    images.append(mask * color)
                else:
                    images.append(mask)

                full_mask.append(mask * color)

            images.append(sum(full_mask))

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
