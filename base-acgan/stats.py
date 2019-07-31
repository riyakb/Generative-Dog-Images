# stats

from __future__ import print_function, division
import os
import csv
import torch
import random
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from PIL import ImageFilter
from hparams import hparams
from data import DogData

# create_csv_data()
def quick_data():

    train_dataset = DogData(csv_file=hparams.train_csv,
                        root_dir=hparams.images_dir,
                        file_format='.jpg',
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ]))

    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size,
                            num_workers=2, shuffle=True)

    for ii, (img, breed, img_names) in enumerate(train_loader):
        print(ii, img.shape, breed, img_names)
        break

# quick_data()
