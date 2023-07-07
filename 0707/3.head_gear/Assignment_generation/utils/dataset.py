# ./utils/dataset.py
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from utils.config import load_config
# from utils.transforms import *

import warnings
warnings.filterwarnings("ignore")

# TODO: Create the HeadGearDataset class
class HeadGearDataset(Dataset):
    def __init__(self, annotations_file, dataset_path, mode, transform=None, target_transform=None):
        self.config = load_config('configs/configs.yaml')  # moved here
        self.image = pd.read_csv(annotations_file)
        self.image_mode = self.image(self.image['data set'] == mode)
        self.dataset_path = dataset_path
        self.transform = transform
        self.target_transform = target_transform
                

    def __len__(self):
        return self.image_mode

    def __getitem__(self, idx):
        # image_path = self.image_mode['filepaths'][idx]
        image_path = self.image_mode.iloc[idx, 1]
       
        image_path = os.path.join(self.dataset_path, image_path)
        image = self.load_image(image_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            image = self.target_transform(image)

        label = self.image_mode.iloc[idx, 0]
        # label = self.image_mode['class id'][idx]       
                
        return image, label

    def load_image(self, path):
        with Image.open(path) as img:
            img.load()  # This forces the image file to be read into memory
            return img  # Return the PIL image directly
