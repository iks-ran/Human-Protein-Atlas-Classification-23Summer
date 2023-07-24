import numpy as np
import cv2
import os
import pandas as pd
import json
from torch.utils.data import Dataset
from .image_utils import IMAGE_TRANSFORM
from einops import rearrange


class HPA(Dataset):
    def __init__(self, 
                data_path: str, 
                split: str, 
                color_info: str, 
                num_labels=28, 
                transformations={}, 
                input_size=224, 
                **kwargs) -> None:
        super(HPA).__init__()

        self.path = os.path.join(data_path, 'HPA')
        self.img_path = os.path.join(self.path, 'train')
        self.csv_path = os.path.join(self.path, f'{split}.csv')
        self.channels = ['red', 'blue', 'green', 'yellow']
        self.num_labels = num_labels
        self.color_info = color_info
        self.input_size = tuple(input_size)

        self._init_samples()
        self._init_color_info()
        self._init_transformations(transformations)

    def _init_samples(self):

        samples = pd.read_csv(self.csv_path).to_numpy()
        self.sample_ids = samples[:, 0]
        self.sample_targets = samples[:, 1]
        self.num_samples = len(samples)
        print(f'Loaded Human Protein Atlas {self.num_samples} samples!')

    def _init_color_info(self):

        with open(self.color_info, 'r') as f:
            data = f.read()
        color_info = json.loads(data)
        self.std = dict()
        self.mean = dict()
        for channel in self.channels:
            self.std.update({channel: color_info[f'{channel}_std']})
            self.mean.update({channel: color_info[f'{channel}_mean']})

    def _init_transformations(self, transformations):
        
        self.trasnform = dict()
        for transform, params in transformations.items():
            self.trasnform.update({transform: [IMAGE_TRANSFORM.get(transform), params]})

    def __getitem__(self, index):

        channels = []
        file_path = os.path.join(self.img_path, self.sample_ids[index])
        for channel in self.channels:
            image = cv2.imread(f'{file_path}_{channel}.png', 0) / 255.0
            channels.append((image - self.mean[channel]) / self.std[channel])
        img = np.stack(channels, axis=-1)

        inp = img.astype(dtype=np.float32)
        for transform, params in self.trasnform.values():
            inp = transform(inp, **params)
        if inp.shape[:2] != self.input_size:
            inp = cv2.resize(inp, self.input_size)
        inp = rearrange(inp, 'h w c -> c h w')

        target = np.asarray(self.sample_targets[index].split(' '), dtype=np.int32)
        label = np.zeros(shape=(self.num_labels), dtype=np.int32)
        label[target] = 1

        ret = {'inp': inp, 'label': label}

        return ret

    def __len__(self):
        return self.num_samples