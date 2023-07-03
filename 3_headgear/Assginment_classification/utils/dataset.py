# ./utils/dataset.py
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from utils.config import load_config

import warnings
warnings.filterwarnings("ignore")

class HeadGearDataset(Dataset):
    def __init__(self, annotations_file, dataset_path, mode, transform=None, target_transform=None):
        self.config = load_config('configs/configs.yaml')  # moved here
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.img_labels[self.img_labels['data set'] == mode]
        
        # TODO: Define the attributes of this dataset
        # self.transform = # fill this in
        # self.target_transform = # fill this in
        # self.dataset_path = # fill this in
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_path = dataset_path

    def __len__(self):
        # TODO: Return the length of the dataset
        return len(self.dataset_path)
        
    def __getitem__(self, idx):
        # TODO: Return the idx-th item of the dataset
        # img_path = # fill this in  # 'filepaths' column
        img_path = self.dataset_path['filepaths']

        # TODO: path join
        # img_path = # fill this in
        img_path = os.path.join(self.dataset_path, img_path)
        image = self.load_image(img_path)
        
        # TODO: Return the idx-th item of the dataset
        # label = # fill this in # 'class id' column
        label = self.dataset_path[self.dataset_path['class id']]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image[idx], label[idx]

    def load_image(self, path):
        with Image.open(path) as img:
            img.load()  # This forces the image file to be read into memory
            return img  # Return the PIL image directly
