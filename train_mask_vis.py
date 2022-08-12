from tqdm import trange
from tqdm import tqdm
import torch
import imageio
import yaml


from argparse import ArgumentParser
from torch.utils.data import DataLoader

from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel
from modules.mask_generator import MaskGenerator

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import FramesDataset
from frames_dataset import DatasetRepeater


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    train_params = config['train_params']
    dataset = FramesDataset( **config['dataset_params'],is_train=True)

    mask_generator = MaskGenerator(**config['model_params']['mask_generator_params'],
                            **config['model_params']['common_params'])
    if torch.cuda.is_available():
        mask_generator.cuda()

    mask_ckp = torch.load(config['model_params']['mask_generator_params']['checkpoint'])
    mask_generator.load_state_dict(mask_ckp['mask_generator'])
    print('load mask generator success... ')
    mask_generator.eval()
    for param in mask_generator.parameters():
        param.requires_grad = False

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)

    for x in dataloader:
        mask = mask_generator(x['source'][0:1,:,:,:].cuda())
        mask = mask[0,:,:,:].repeat(3,1,1)
        mask = mask.permute(1,2,0)
        imageio.imsave('./mask_vis/' + x['name'][0] + '.png', 255*mask.data.cpu().numpy())
        mask[mask < 0.2] = 0
        imageio.imsave('./mask_vis/' + x['name'][0] + '_clamp0.15.png', 255*mask.data.cpu().numpy())

