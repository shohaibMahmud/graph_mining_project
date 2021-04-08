import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms

class videoDataSet(Dataset):
    """Custom dataset for video data"""
    def __init__(self, root_dir, numOfFramesInSample, transform=None):
        self.labelList = pd.read_csv(root_dir+'\\label\\labels.csv')
        self.root_dir = root_dir+'\\img\\'
        self.transform = transform
        self.numOfFramesInSample = numOfFramesInSample
    def __len__(self):
        return len(self.labelList)

    def __getitem__(self, idx):
        list_of_img_name = [self.root_dir+str(idx)+'_'+str(n)+'.png'\
                           for n in range(self.numOfFramesInSample)]
        img_samples = torch.empty(self.numOfFramesInSample, 3, 320, 240)
        for n,img_name in enumerate(list_of_img_name):
            img = Image.open(img_name)
            img = img.convert('RGB')
            img = transforms.ToTensor()(img)
            img_samples[n] = img
        labels = torch.tensor(self.labelList.iloc[idx, 1])
        if self.transform:
            img_samples = self.transform(img_samples)
        return img_samples, labels