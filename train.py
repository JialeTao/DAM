from tqdm import trange
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel

from torch.optim.lr_scheduler import MultiStepLR
from frames_dataset import DatasetRepeater


def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, device_ids, bg_predictor=None, local_rank=-1, world_size=1):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    if bg_predictor is not None:
        optimizer_bg_predictor = torch.optim.Adam(bg_predictor.parameters(), lr=train_params['lr_bg_predictor'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch= 0
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
        checkpoint = torch.load(checkpoint, map_location='cuda:{}'.format(torch.cuda.current_device()))
        if bg_predictor is not None:
            bg_predictor.load_state_dict(checkpoint['bg_predictor'])
            print('load bg_predictor success... ')
        if kp_detector is not None:
            state_dict = checkpoint['kp_detector']
            temp_dict = kp_detector.state_dict()

            # load first-order-root model for second-order-root model
            num_kp_saved = state_dict['kp.weight'].data.shape[0]
            num_kp_temp = temp_dict['kp.weight'].data.shape[0]
            if num_kp_saved != num_kp_temp:
                print("warning :the loaded kp_detector has a diffrent kp_num of %d with that of temp model %d" % (num_kp_saved, num_kp_temp) )
                kp_weight_old = state_dict['kp.weight'].clone()
                kp_new = nn.Conv2d(in_channels=temp_dict['kp.weight'].data.shape[1], out_channels=temp_dict['kp.weight'].data.shape[0],kernel_size=(7, 7),padding=0)
                kp_weight_new = kp_new.weight.cuda(device)
                kp_weight_new[0:num_kp_saved-1,:,:,:] = kp_weight_old[0:-1,:,:,:]
                kp_weight_new[-1,:,:,:] = kp_weight_old[-1,:,:,:]
                state_dict['kp.weight'] = kp_weight_new

                kp_bias_old = state_dict['kp.bias'].clone()
                kp_bias_new = kp_new.bias.cuda(device)
                kp_bias_new[0:num_kp_saved-1] = kp_bias_old[0:-1]
                kp_bias_new[-1] = kp_bias_old[-1]
                state_dict['kp.bias'] = kp_bias_new

                num_jacobian_maps = temp_dict['jacobian.weight'].shape[0]
                num_jacobian = int(num_jacobian_maps / 4)

                jacobian_weight_old = state_dict['jacobian.weight'].clone()
                jacobian_weight_new = torch.zeros(temp_dict['jacobian.weight'].shape).cuda(device)
                jacobian_weight_new[0:(num_kp_saved-1)*4,:,:,:] = jacobian_weight_old[0:(num_kp_saved-1)*4,:,:,:]
                jacobian_weight_new[(num_kp_temp-1)*4:,:,:,:] = jacobian_weight_old[(num_kp_saved-1)*4:,:,:,:]
                state_dict['jacobian.weight'] = jacobian_weight_new

                jacobian_bias_old = state_dict['jacobian.bias'].clone()
                jacobian_bias_new = torch.tensor([1, 0, 0, 1] * num_jacobian, dtype=torch.float).cuda(device)
                jacobian_bias_new[0:(num_kp_saved-1)*4] = jacobian_bias_old[0:(num_kp_saved-1)*4]
                jacobian_bias_new[(num_kp_temp-1)*4:] = jacobian_bias_old[(num_kp_saved-1)*4:]
                state_dict['jacobian.bias'] = jacobian_bias_new
            # load first-order-root model for second-order-root model
            kp_detector.load_state_dict(state_dict, strict=False)
        generator.load_state_dict(checkpoint['generator'])
    else:
        start_epoch = 0  

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))
    if bg_predictor is not None:
        scheduler_bg_predictor = MultiStepLR(optimizer_bg_predictor, train_params['epoch_milestones'], gamma=0.1,
                                            last_epoch=-1 + start_epoch * (train_params['lr_bg_predictor'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    # dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)

    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params, bg_predictor=bg_predictor)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    distributed = local_rank >= 0
    if torch.cuda.is_available():
        if distributed:
            train_sampler = DistributedSampler(dataset)
            dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=False, num_workers=6, drop_last=True, sampler=train_sampler)

            generator_full = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator_full)
            discriminator_full = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator_full)
            device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
            generator_full.to(device)
            discriminator_full.to(device)
            generator_full = torch.nn.parallel.DistributedDataParallel(generator_full,
                                                                       find_unused_parameters=True,
                                                                       device_ids=[torch.cuda.current_device()],
                                                                       output_device=torch.cuda.current_device())
            discriminator_full = torch.nn.parallel.DistributedDataParallel(discriminator_full,
                                                                           find_unused_parameters=True,
                                                                           device_ids=[torch.cuda.current_device()],
                                                                           output_device=torch.cuda.current_device())
        else:
            dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)
            generator_full = nn.DataParallelWithCallback(generator_full, device_ids=device_ids)
            discriminator_full = nn.DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            if distributed:
                dataloader.sampler.set_epoch(epoch)
            for x in dataloader:
                # print(x['driving'].shape)
                losses_generator, generated = generator_full(x)
                x['driving'] = x['driving'][:,0:3,:,:]
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                if  train_params['clip_generator_grad']:
                    nn.utils.clip_grad_norm_(generator.parameters(), train_params['clip'])
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                if  train_params['clip_kp_detector_grad']:
                    nn.utils.clip_grad_norm_(kp_detector.parameters(), train_params['clip'])
                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()
                if bg_predictor is not None:
                    optimizer_bg_predictor.step()
                    optimizer_bg_predictor.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                if distributed:
                    for key in losses_generator:
                        torch.distributed.reduce(losses_generator[key], dst=0)
                        losses_generator[key] = losses_generator[key] / world_size
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                if distributed:
                    if local_rank == 0:
                        logger.log_iter(losses=losses)
                else:
                    logger.log_iter(losses=losses)

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            if bg_predictor is not None:
                scheduler_bg_predictor.step()

            if bg_predictor is not None:
                if not distributed or (distributed and local_rank == 0):
                    logger.log_epoch(epoch, {'generator': generator,
                                            'discriminator': discriminator,
                                            'kp_detector': kp_detector,
                                            'bg_predictor': bg_predictor,
                                            'optimizer_generator': optimizer_generator,
                                            'optimizer_discriminator': optimizer_discriminator,
                                            'optimizer_kp_detector': optimizer_kp_detector,
                                             'optimizer_bg_predictor': optimizer_bg_predictor}, inp=x, out=generated)
            else:
                if not distributed or (distributed and local_rank == 0):
                    logger.log_epoch(epoch, {'generator': generator,
                                            'discriminator': discriminator,
                                            'kp_detector': kp_detector,
                                            'optimizer_generator': optimizer_generator,
                                            'optimizer_discriminator': optimizer_discriminator,
                                             'optimizer_kp_detector': optimizer_kp_detector}, inp=x, out=generated)
