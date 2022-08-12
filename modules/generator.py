import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.dense_motion import DenseMotionNetwork


class OcclusionAwareGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False,
                 root_motion_kp_distance=False, num_root_kp=1, sub_kp_num=[2,2,3,3], num_root_kp_orders=1, skips=False, root_motion=False):
        super(OcclusionAwareGenerator, self).__init__()

        self.num_root_kp = num_root_kp
        self.sub_kp_num = sub_kp_num
        self.num_root_kp_orders = num_root_kp_orders
        self.root_motion = root_motion
        if dense_motion_params is not None:
            if root_motion_kp_distance and not root_motion:
                dense_motion_kp_num = num_kp - num_root_kp
            else:
                dense_motion_kp_num = num_kp
            self.dense_motion_network = DenseMotionNetwork(num_kp=dense_motion_kp_num, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels
        self.skips = skips

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)

    def apply_optical(self, input_previous=None, input_skip=None, motion_params=None):
        if motion_params is not None:
            if 'occlusion_map' in motion_params:
                occlusion_map = motion_params['occlusion_map']
            else:
                occlusion_map = None
            deformation = motion_params['deformation']
            input_skip = self.deform_input(input_skip, deformation)

            if occlusion_map is not None:
                if input_skip.shape[2] != occlusion_map.shape[2] or input_skip.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=input_skip.shape[2:], mode='bilinear')
                if input_previous is not None:
                    input_skip = input_skip * occlusion_map + input_previous * (1 - occlusion_map)
                else:
                    input_skip = input_skip * occlusion_map
            out = input_skip
        else:
            out = input_previous if input_previous is not None else input_skip
        return out

    def forward(self, source_image, kp_driving, kp_source, driving_image=None,kp_driving_next=None, bg_params=None):
        # Encoding (downsampling) part
        out = self.first(source_image)
        skips = [out]
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            skips.append(out)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                    kp_source=kp_source, bg_params=bg_params)
            deformation = dense_motion['deformation']
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            out_backflow = self.apply_optical(input_previous=None, input_skip=out, motion_params=dense_motion)

            # if occlusion_map is not None:
                # if out_backflow.shape[2] != occlusion_map.shape[2] or out_backflow.shape[3] != occlusion_map.shape[3]:
                    # occlusion_map = F.interpolate(occlusion_map, size=out_backflow.shape[2:], mode='bilinear')
                # out_backflow = out_backflow * occlusion_map
            output_dict["deformed"] = self.deform_input(source_image, deformation)

        # Decoding part
        out_backflow = self.bottleneck(out_backflow)
        for i in range(len(self.up_blocks)):
            if self.skips:
                out_backflow = self.apply_optical(input_skip=skips[-(i + 1)], input_previous=out_backflow, motion_params=dense_motion)
            out_backflow = self.up_blocks[i](out_backflow)
        if self.skips:
            out_backflow = self.apply_optical(input_skip=skips[0], input_previous=out_backflow, motion_params=dense_motion)
        out_backflow = self.final(out_backflow)
        out_backflow = F.sigmoid(out_backflow)
        if self.skips:
            out_backflow = self.apply_optical(input_skip=source_image, input_previous=out_backflow, motion_params=dense_motion)

        output_dict["prediction"] = out_backflow

        return output_dict
