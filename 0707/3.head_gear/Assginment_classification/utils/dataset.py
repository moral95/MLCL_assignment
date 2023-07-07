# ./utils/dataset.py
import os
import pandas as pd
import cv2
from torch.utils.data import Dataset
from PIL import Image
from utils.config import load_config
from torchvision.io import read_image

import warnings
warnings.filterwarnings("ignore")

class HeadGearDataset(Dataset):
    def __init__(self, annotations_file, dataset_path, mode, transform=None, target_transform=None):
        self.config = load_config('configs/configs.yaml')  # moved here
        self.img = pd.read_csv(annotations_file)
        self.img_mod = self.img[self.img['data set'] == mode]
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_path = dataset_path
        # print("self.img:",self.img)
        # print("self.img_mod :",self.img_mod)
        # print("self.dataset_path :",self.dataset_path)
        # print("self.imag_mod :",self.img_mod)
        print(f"{mode} length : {len(self.img_mod['filepaths'])}")
    
    def __len__(self):
        # return len(self.img_mod['filepaths'])
        return len(self.img_mod)
        
    def __getitem__(self, idx):
        # img_path = self.img_mod['filepaths'][idx]
        img_path = self.img_mod.iloc[idx, 1]
        img_path = os.path.join(self.dataset_path, img_path)
        image = self.load_image(img_path)
        # image = cv2.imread(img_path)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            image = self.target_transform(image)
        # label = self.img_mod['class id'][idx]
        label = self.img_mod.iloc[idx, 0]
        return image, label

    def load_image(self, path):
        with Image.open(path) as img:
            img.load()  # This forces the image file to be read into memory
            return img  # Return the PIL image directly
