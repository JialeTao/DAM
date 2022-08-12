from torch import nn
import torch
import torch.nn.functional as F
from modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d
from torch.nn import BatchNorm2d


class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1, single_jacobian_map=False, pad=0,
                 subroot_leaf_attention=False, attention_scale=1, attention_channel=64, detach_feature=False,reduce_dim='row', ignore_kp_list=None):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)

        self.subroot_leaf_attention = subroot_leaf_attention
        if subroot_leaf_attention:
            self.subroot_attention_block = nn.Sequential(nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=attention_channel, kernel_size=(1, 1), padding=0),
                                                         BatchNorm2d(64),
                                                         nn.ReLU()
                                                         )
            self.leaf_attention_block = nn.Sequential(nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=attention_channel, kernel_size=(1, 1), padding=0),
                                                         BatchNorm2d(64),
                                                         nn.ReLU()
                                                         )
            self.attention_scale=attention_scale
            self.detach_feature = detach_feature
            self.reduce_dim = reduce_dim
            self.ignore_kp_list = ignore_kp_list

        self.num_kp = num_kp

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3)) # N * 10 * 2
        kp = {'value': value}

        return kp

    def attention_block(self, sub_root_kp, leaf_kp, feature_map):
        # input sub_root_kp: N * 4 * 2
        # input leaf_kp: N * 10 * 2
        # input feature_map: N * C * H * W
        # return attention weight of sub_root feature and leaf feature

        if self.detach_feature:
            feature_map = feature_map.detach()
        # N * C * 4 * 1
        sub_root_feature = F.grid_sample(feature_map, grid=sub_root_kp.unsqueeze(2))
        sub_root_feature = self.subroot_attention_block(sub_root_feature)
        sub_root_feature = sub_root_feature.squeeze(-1).permute(0,2,1)
        # N * C * 10 * 1
        leaf_feature = F.grid_sample(feature_map, grid=leaf_kp.unsqueeze(2))
        leaf_feature = self.leaf_attention_block(leaf_feature)
        leaf_feature = leaf_feature.squeeze(-1)

        channel = sub_root_feature.shape[1]

        # N * 4 * 10
        sub_root_leaf_sim = torch.matmul(sub_root_feature, leaf_feature)
        sub_root_leaf_sim = (channel**-0.5) * sub_root_leaf_sim
        if self.reduce_dim == 'row':
            if self.ignore_kp_list is not None:
                sub_root_leaf_sim[:,:,self.ignore_kp_list] = sub_root_leaf_sim[:,:,self.ignore_kp_list] * 0 - float('inf')
            sub_root_leaf_sim = F.softmax(self.attention_scale * sub_root_leaf_sim, dim=2)
        else:
            sub_root_leaf_sim = F.softmax(self.attention_scale * sub_root_leaf_sim, dim=1)

        return sub_root_leaf_sim

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)
        out['heatmap'] = heatmap
        out['feature_map'] = feature_map

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])

            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            # N * 10 *2 *2


            out['jacobian'] = jacobian

        return out
