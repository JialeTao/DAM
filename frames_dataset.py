import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
import imageio
from imageio import mimread
import pandas as pd

import numpy as np
from torch.utils.data import Dataset
from augmentation import AllAugmentationTransform
import glob

import time

def read_video(name, frame_shape, read_first_frame=False):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        if not read_first_frame:
            video_array = np.array(
                [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
            # print(video_array.shape)
        else:
            video_array = np.array(
                [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(1)])
            # print(video_array.shape)
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        #video = np.array(mimread(name,memtest=False))
        reader = imageio.get_reader(name)
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
                if read_first_frame:
                    break
        except RuntimeError:
            pass
        reader.close()
        video = np.array(driving_video)
        # print(video_array.shape)
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, data_list=None, data_list_test=None, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None, read_first_frame=False):
        self.root_dir = root_dir
        # self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.read_first_frame=read_first_frame

        f = open(data_list)
        file_list = f.readlines()
        if id_sampling:
            train_video_ids = []
            train_videos = {}
            for file_name in file_list:
                img_name = file_name.strip().split('/')[1]
                instance_id = file_name.strip().split('/')[0]
                video_id = instance_id.split('#')[0]
                if video_id not in train_video_ids:
                    train_video_ids.append(video_id)
                if video_id not in train_videos.keys():
                    train_videos[video_id] = {}
                if instance_id not in train_videos[video_id].keys():
                    train_videos[video_id][instance_id] = []
                train_videos[video_id][instance_id].append(img_name)
            f.close()
        else:
            train_video_ids = []
            train_videos = {}
            for file_name in file_list:
                img_name = file_name.strip().split('/')[1]
                instance_id = file_name.strip().split('/')[0]
                if instance_id not in train_video_ids:
                    train_video_ids.append(instance_id)
                if instance_id not in train_videos.keys():
                    train_videos[instance_id] = []
                train_videos[instance_id].append(img_name)
            f.close()

        f_test = open(data_list_test)
        file_list = f_test.readlines()
        test_video_ids = []
        test_videos = {}
        for file_name in file_list:
            img_name = file_name.strip().split('/')[1]
            instance_id = file_name.strip().split('/')[0]
            if instance_id not in test_video_ids:
                test_video_ids.append(instance_id)
            if instance_id not in test_videos.keys():
                test_videos[instance_id] = []
            test_videos[instance_id].append(img_name)
        f_test.close()

        data_dir_name = self.root_dir.split('/')[-2]
        if is_train:
            local_dir_name = os.path.join('data', data_dir_name, 'train')
            self.root_dir = local_dir_name
        else:
            local_dir_name = os.path.join('data', data_dir_name, 'test')
            # print(local_dir_name)
            # print(os.path.join(self.root_dir+'test/', test_video_ids[0], test_videos[test_video_ids[0]][0]))
            self.root_dir = local_dir_name
        self.local_dir = local_dir_name
        print(self.root_dir)

        # if os.path.exists(os.path.join(root_dir, 'train')):
            # assert os.path.exists(os.path.join(root_dir, 'test'))
            # print("Use predefined train-test split.")
            # if id_sampling:
                # train_videos = {os.path.basename(video).split('#')[0] for video in
                                # os.listdir(os.path.join(root_dir, 'train'))}
                # train_videos = list(train_videos)
            # else:
                # train_videos = os.listdir(os.path.join(root_dir, 'train'))
            # test_videos = os.listdir(os.path.join(root_dir, 'test'))
            # self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
            # # test_videos = os.listdir(os.path.join(root_dir, 'train'))
            # # self.root_dir = os.path.join(self.root_dir, 'train')
        # else:
            # print("Use random train-test split.")
        #     train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        # if is_train:
            # self.videos = train_videos
        # else:
            # self.videos = test_videos

        if is_train:
            self.videos = train_video_ids
            self.video_dicts = train_videos
        else:
            self.videos = test_video_ids
            self.video_dicts = test_videos
            # print(len(test_video_ids), len(list(test_videos.keys())))

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            # name = self.videos[idx]
            # path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))

            name = self.videos[idx]
            name_list = sorted(list(self.video_dicts[name].keys()))
            video_name = np.random.choice(name_list)
            path = os.path.join(self.root_dir, video_name)
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)
            video_name = name

        # video_name = os.path.basename(path)

        if self.is_train:
            # frames = sorted(os.listdir(path))
            if self.id_sampling:
                frames = self.video_dicts[name][video_name]
            else:
                frames = self.video_dicts[name]
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))

            video_array = []
            for idx in frame_idx:
                try:
                    img = img_as_float32(io.imread(os.path.join(path, frames[idx])))
                except TypeError:
                    img = img_as_float32(io.imread(os.path.join(path, frames[idx].decode())))
                if len(img.shape) == 2:
                    img = gray2rgb(img)
                if img.shape[-1] == 4:
                    img = img[..., :3]

                video_array.append(img)
        else:
            video_array = read_video(path, frame_shape=self.frame_shape, read_first_frame=self.read_first_frame)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(num_frames)
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array_aug = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array_aug[0], dtype='float32')
            driving = np.array(video_array_aug[1], dtype='float32')
            # out['driving'] = np.concatenate(driving.transpose((0, 3, 1, 2)))
            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        self.initial_dataset.read_first_frame = False
        first = self.initial_dataset[pair[0]]
        self.initial_dataset.read_first_frame = True
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}
