from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid
from torchvision import models
import numpy as np
import imageio
from torch.autograd import grad


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        # N * (HW) * 2 * 1
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            # 1 * (HW) * 1 * 2 - 1 * 1 * 25 *2
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params, bg_predictor=None):
        super(GeneratorFullModel, self).__init__()
        self.bg_predictor = bg_predictor
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        # specify the relation of anchors, instead of learning the attention
        self.use_prior_kp = train_params['use_prior_kp']
        self.prior_kp_list = train_params['prior_kp_list']

        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.subroot_leaf_attention = kp_extractor.subroot_leaf_attention
        self.num_kp = kp_extractor.num_kp
        self.num_root_kp = generator.num_root_kp
        self.sub_kp_num = generator.sub_kp_num
        self.num_root_kp_orders = generator.num_root_kp_orders

        self.l1 = nn.L1Loss()


        self.kp_distance_metric = 'L2'
        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

    def kp_distance(self, kp1, kp2):
        # input N * num_kp * 2
        if self.kp_distance_metric == 'L2':
            return torch.norm(kp1-kp2, dim=2)
        elif self.kp_distance_metric == 'L1':
            return torch.abs(kp1-kp2)
        else:
            dkp = kp1 - kp2
            d2kp = dkp.pow(2)
            dkp_tuple = torch.cat([dkp,d2kp], dim=-1)
            kp_dist = self.generator.dist_param * dkp_tuple
            kp_dist = torch.sum(kp_dist, dim=-1)
            return kp_dist

    # Batched index_select
    def batched_index_select(self, t, dim, index):
        # selcet to dim 1 of indx
        dummy = index.unsqueeze(2).expand(index.size(0), index.size(1), t.size(2))
        out = t.gather(dim, dummy) # b x e x f
        return out




    def forward(self, x, aug_driving=False, warm_kp=False):
        kp_source = self.kp_extractor(x['source'])
        # if self.train_params['stage2']['is_stage2']:
        #     kp_source = detach_kp(kp_source)
        bs, channel, h, w =  x['driving'].shape

        kp_driving = self.kp_extractor(x['driving'][:,0:3,:,:])

        if self.bg_predictor is not None:
            bg_params = self.bg_predictor(x['source'], x['driving'][:,0:3,:,:])
        else:
            bg_params = None

        num_kp = self.num_kp
        if (self.loss_weights['root_motion_kp_distance'] != 0 or self.loss_weights['root_motion_sub_root_distance'] != 0) and not self.generator.root_motion:
            num_root_kp = self.num_root_kp
            kp_source_for_motion = {}
            kp_source_for_motion['value'] = kp_source['value'][:,0:num_kp-num_root_kp,:]
            kp_source_for_motion['jacobian'] = kp_source['jacobian'][:,0:num_kp-num_root_kp,:,:]
            kp_driving_for_motion = {}
            kp_driving_for_motion['value'] = kp_driving['value'][:,0:num_kp-num_root_kp,:]
            kp_driving_for_motion['jacobian'] = kp_driving['jacobian'][:,0:num_kp-num_root_kp,:,:]
            generated = self.generator(x['source'], kp_source=kp_source_for_motion, kp_driving=kp_driving_for_motion, driving_image=x['driving'][:,0:3,:,:], bg_params=bg_params)
        else:
            generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving, driving_image=x['driving'][:,0:3,:,:], bg_params=bg_params)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        loss_values = {}

        pyramide_real = self.pyramid(x['driving'][:,0:3,:,:])
        pyramide_generated = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if sum(self.loss_weights['perceptual_aug']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params_aug'])
            driving_aug = transform.transform_frame(x['driving'][:,0:3,:,:])
            kp_driving_aug = self.kp_extractor(driving_aug)

            if self.bg_predictor is not None and self.loss_weights['use_bg_params']:
                bg_params = self.bg_predictor(x['source'], driving_aug)
            else:
                bg_params = None

            if self.loss_weights['root_motion_kp_distance'] != 0 or self.loss_weights['root_motion_sub_root_distance'] != 0:
                kp_driving_aug_for_motion = {}
                kp_driving_aug_for_motion['value'] = kp_driving_aug['value'][:,0:num_kp-num_root_kp,:]
                kp_driving_aug_for_motion['jacobian'] = kp_driving_aug['jacobian'][:,0:num_kp-num_root_kp,:,:]
                generated_aug = self.generator(x['source'], kp_source=kp_source_for_motion, kp_driving=kp_driving_aug_for_motion, driving_image=driving_aug, bg_params=bg_params)
            else:
                generated_aug = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving_aug, driving_image=driving_aug, bg_params=bg_params)

            if sum(self.loss_weights['perceptual_aug']) != 0:
                # generated_aug.update({'kp_source': kp_source, 'kp_driving': kp_driving})
                pyramide_real_aug = self.pyramid(driving_aug)
                pyramide_generated_aug = self.pyramid(generated_aug['prediction'])

                value_total = 0
                for scale in self.scales:
                    x_vgg = self.vgg(pyramide_generated_aug['prediction_' + str(scale)])
                    y_vgg = self.vgg(pyramide_real_aug['prediction_' + str(scale)])

                    for i, weight in enumerate(self.loss_weights['perceptual_aug']):
                        value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                        value_total += self.loss_weights['perceptual_aug'][i] * value
                    loss_values['perceptual_aug'] = value_total

        if self.loss_weights['reconstruction'] != 0:
            value_total = 0
            for scale in self.rec_scales:
                x_rec = pyramide_generated['prediction_' + str(scale)]
                y_rec = pyramide_real['prediction_' + str(scale)]
                value = torch.abs(x_rec-y_rec)
                value_total += value.mean()
                loss_values['recnstruction'] = value_total * self.loss_weights['reconstruction']

        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['source'].shape[0], **self.train_params['transform_params'])
            # transformed_frame = transform.transform_frame(x['driving'][:,start_center:end_center,:,:])
            transformed_frame = transform.transform_frame(x['source'][:,0:3,:,:])
            transformed_kp = self.kp_extractor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = torch.abs(kp_source['value'] - transform.warp_coordinates(transformed_kp['value']))
                value = value.mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0 or self.loss_weights['equivariance_jacobian_focal'] != 0 or self.loss_weights['equivariance_jacobian_shrinkage'] != 0:
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']),
                                                    transformed_kp['jacobian'])
                normed_source = torch.inverse(kp_source['jacobian'])
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_source, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())
                value = torch.abs(eye - value)

                value = value.mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        if self.loss_weights['root_motion_kp_distance'] != 0 or self.loss_weights['root_motion_sub_root_distance'] != 0:
            value_total = 0
            value_total2 = 0
            jacobian_root_base = torch.matmul(kp_source['jacobian'][:,-1,:,:], torch.inverse(kp_driving['jacobian'][:,-1,:,:]))

            if self.num_root_kp > 1 :
                # if self.num_root_kp_orders == 2:
                if 2 in self.num_root_kp_orders:
                    jacobian_root = jacobian_root_base.unsqueeze(1).repeat(1,num_root_kp-1,1,1)
                    ## z_sub_root - z_root
                    relative_coordinate_sub_root = kp_driving['value'][:,num_kp-num_root_kp:num_kp-1,:] - kp_driving['value'][:,num_kp-1:num_kp,:]
                    relative_coordinate_sub_root = torch.matmul(jacobian_root, relative_coordinate_sub_root.unsqueeze(-1))
                    ## T_{s<-d}(z_root)
                    root_motion = kp_source['value'][:,num_kp-1:num_kp,:]
                    root_motion = root_motion.repeat(1,num_root_kp-1,1)
                    ## T_{s-d}(z_sub_root)
                    sub_root_motion_controlled = root_motion + relative_coordinate_sub_root.squeeze(-1)
                    value = torch.norm(sub_root_motion_controlled-kp_source['value'][:,num_kp-num_root_kp:num_kp-1,:], dim=2)
                    value_total += self.loss_weights['root_motion_sub_root_distance'] * value.mean()
                # elif self.num_root_kp_orders == 1:
                if 1 in self.num_root_kp_orders:
                    jacobian_root = jacobian_root_base.unsqueeze(1).repeat(1,num_kp-num_root_kp,1,1)
                    relative_coordinate = kp_driving['value'][:,0:num_kp-num_root_kp,:] - kp_driving['value'][:,num_kp-1:num_kp,:]
                    relative_coordinate = torch.matmul(jacobian_root, relative_coordinate.unsqueeze(-1))
                    root_motion = kp_source['value'][:,num_kp-1:num_kp,:]
                    root_motion = root_motion.repeat(1,num_kp-num_root_kp,1)
                    kp_motion = root_motion + relative_coordinate.squeeze(-1)
                    value = torch.norm(kp_motion-kp_source['value'][:,0:num_kp-num_root_kp,:], dim=2)
                    value_total += self.loss_weights['root_motion_kp_distance'] * value.mean()


                # if self.loss_weights['root_kp_distance'] > 0:
                    # value_kp_distance = root_motion - kp_source['value'][:,num_kp-num_root_kp:num_kp-1,:]
                    # value_kp_distance = torch.norm(value_kp_distance,dim=2)
                    # value_total2 += -1 * self.loss_weights['root_kp_distance'] * value_kp_distance.mean()

                kp_total = 0
                sub_root = 0
                jacobian_sub_root = torch.matmul(kp_source['jacobian'][:,num_kp-num_root_kp:num_kp-1,:,:], torch.inverse(kp_driving['jacobian'][:,num_kp-num_root_kp:num_kp-1,:,:]))
                if  not self.subroot_leaf_attention:
                    if  not self.use_prior_kp:
                        for kp_num in self.sub_kp_num:
                            ## z_sub_leaf - z_sub_root
                            relative_coordinate_sub_leaf = kp_driving['value'][:,kp_total:kp_total+kp_num,:] - kp_driving['value'][:,sub_root+num_kp-num_root_kp:sub_root+1+num_kp-num_root_kp,:]
                            relative_coordinate_sub_leaf = torch.matmul(jacobian_sub_root[:,sub_root:sub_root+1,:,:].repeat(1,kp_num,1,1,), relative_coordinate_sub_leaf.unsqueeze(-1))
                            ## T_{s<-d}(z_sub_root)
                            sub_root_motion = kp_source['value'][:,sub_root+num_kp-num_root_kp:sub_root+1+num_kp-num_root_kp,:]
                            sub_root_motion = sub_root_motion.repeat(1,kp_num,1)
                            ## T_{s-d}(z_sub_leaf)
                            sub_leaf_motion_controlled = sub_root_motion + relative_coordinate_sub_leaf.squeeze(-1)
                            value = torch.norm(sub_leaf_motion_controlled-kp_source['value'][:,kp_total:kp_total+kp_num,:], dim=2)
                            value_total += self.loss_weights['sub_root_motion_kp_distance'] * value.mean()

                            if self.loss_weights['root_kp_distance'] > 0:
                                value_kp_distance = sub_root_motion - kp_source['value'][:,kp_total:kp_total+kp_num,:]
                                value_kp_distance = torch.norm(value_kp_distance,dim=2)
                                value_total2 += -1 * self.loss_weights['root_kp_distance'] * value_kp_distance.mean()

                            kp_total += kp_num
                            sub_root += 1

                    else:
                        for kp in self.prior_kp_list:
                            for i in range(len(kp)):
                                if i == 0:
                                    kp_driving_sub = kp_driving['value'][:,kp[i]:kp[i]+1,:]
                                    kp_source_sub = kp_source['value'][:,kp[i]:kp[i]+1,:]
                                else:
                                    kp_driving_sub = torch.cat([kp_driving_sub,kp_driving['value'][:,kp[i]:kp[i]+1,:]], dim=1)
                                    kp_source_sub = torch.cat([kp_source_sub,kp_source['value'][:,kp[i]:kp[i]+1,:]], dim=1)
                            ## z_sub_leaf - z_sub_root
                            relative_coordinate_sub_leaf = kp_driving_sub - kp_driving['value'][:,sub_root+num_kp-num_root_kp:sub_root+1+num_kp-num_root_kp,:]
                            relative_coordinate_sub_leaf = torch.matmul(jacobian_sub_root[:,sub_root:sub_root+1,:,:].repeat(1,len(kp),1,1,), relative_coordinate_sub_leaf.unsqueeze(-1))
                            ## T_{s<-d}(z_sub_root)
                            sub_root_motion = kp_source['value'][:,sub_root+num_kp-num_root_kp:sub_root+1+num_kp-num_root_kp,:]
                            sub_root_motion = sub_root_motion.repeat(1,len(kp),1)
                            ## T_{s-d}(z_sub_leaf)
                            sub_leaf_motion_controlled = sub_root_motion + relative_coordinate_sub_leaf.squeeze(-1)
                            value = torch.norm(sub_leaf_motion_controlled-kp_source_sub, dim=2)
                            value_total += self.loss_weights['sub_root_motion_kp_distance'] * value.mean()
                            sub_root += 1
                else:
                    # N * 4 * 10
                    subroot_leaf_attention = self.kp_extractor.attention_block(sub_root_kp=kp_driving['value'][:,num_kp-num_root_kp:num_kp-1,:], leaf_kp=kp_driving['value'][:,0:num_kp-num_root_kp,:], feature_map=kp_driving['feature_map'])
                    ## z_leaf - z_sub_root
                    z_leaf = kp_driving['value'][:,0:num_kp-num_root_kp,:]
                    z_leaf = z_leaf.unsqueeze(1)
                    z_sub_root = kp_driving['value'][:,num_kp-num_root_kp:num_kp-1,:]
                    z_sub_root = z_sub_root.unsqueeze(2).repeat(1,1,num_kp-num_root_kp,1)
                    relative_coordinate_sub_leaf = z_leaf - z_sub_root
                    relative_coordinate_sub_leaf = torch.matmul(jacobian_sub_root.unsqueeze(2).repeat(1,1,num_kp-num_root_kp,1,1), relative_coordinate_sub_leaf.unsqueeze(-1))
                    ## T_{s<-d}(z_sub_root)
                    sub_root_motion = kp_source['value'][:,num_kp-num_root_kp:num_kp-1,:]
                    sub_root_motion = sub_root_motion.unsqueeze(2).repeat(1,1,num_kp-num_root_kp,1)
                    ## T_{s-d}(z_sub_leaf)
                    sub_leaf_motion_controlled = sub_root_motion + relative_coordinate_sub_leaf.squeeze(-1)
                    # N * 4 * 10
                    value = torch.norm(sub_leaf_motion_controlled-kp_source['value'][:,0:num_kp-num_root_kp,:].unsqueeze(1), dim=3)
                    value = subroot_leaf_attention * value
                    value = value.sum(2).sum(1)
                    value_total += self.loss_weights['sub_root_motion_kp_distance'] * value.mean()

                    subroot_leaf_attention = subroot_leaf_attention[:,:,subroot_leaf_attention.sum(1).sum(0)!=0]
                    if self.loss_weights['orthognal_sub_leaf'] != 0:
                        if self.loss_weights['orthognal_type'] == 'L1':
                            # L1 sparse
                            value_ortho = subroot_leaf_attention.unsqueeze(2) - subroot_leaf_attention.unsqueeze(1)
                            value_ortho = torch.abs(value_ortho)
                            value_ortho = value_ortho.mean(3)
                        elif self.loss_weights['orthognal_type'] == 'inner_product':
                            ## inner product L1
                            value_ortho = subroot_leaf_attention.unsqueeze(2) * subroot_leaf_attention.unsqueeze(1)
                            value_ortho = value_ortho.sum(3)

                        for i in range(value_ortho.shape[1]):
                            value_ortho[:,i,i] = 0
                        if self.loss_weights['orthognal_type'] == 'L1':
                            value_ortho = -1 * value_ortho[value_ortho!=0]
                        else:
                            value_ortho = value_ortho[value_ortho!=0]

                        loss_values['sub_attetion_distance'] = self.loss_weights['orthognal_sub_leaf'] * value_ortho.mean()
                    if self.loss_weights['complete_sub_leaf_attention'] != 0:
                        value_att = torch.sum(subroot_leaf_attention, dim=1)
                        value_att = self.loss_weights['sub_leaf_attention_value'] - value_att
                        value_att[value_att<0] = 0
                        loss_values['complete_sub_leaf_attention'] = self.loss_weights['complete_sub_leaf_attention'] * value_att.mean()
                    # print(subroot_leaf_attention[0,:,:])

                loss_values['root_motion_kp_distance'] = value_total
                # if self.loss_weights['root_kp_distance'] > 0:
                #     loss_values['root_kp_distance'] = value_total2

            else:
                jacobian_root = jacobian_root_base.unsqueeze(1).repeat(1,num_kp-1,1,1)
                relative_coordinate = kp_driving['value'][:,0:num_kp-1,:] - kp_driving['value'][:,num_kp-1:num_kp,:]
                relative_coordinate = torch.matmul(jacobian_root, relative_coordinate.unsqueeze(-1))
                root_motion = kp_source['value'][:,num_kp-1:num_kp,:]
                root_motion = root_motion.repeat(1,num_kp-1,1)
                kp_motion = root_motion + relative_coordinate.squeeze(-1)
                value = self.kp_distance(kp_motion, kp_source['value'][:,0:num_kp-1,:])
                value = value.mean()
                loss_values['root_motion_kp_distance'] = self.loss_weights['root_motion_kp_distance'] * value

        return loss_values, generated


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())

        kp_driving = generated['kp_driving']
        discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
        discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        return loss_values
