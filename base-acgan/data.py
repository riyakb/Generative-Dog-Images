from __future__ import print_function, division
import os
import json
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
import xml.etree.ElementTree as ET
from ast import literal_eval


class DogData(Dataset):

  def __init__(self, csv_file, root_dir, transform=None, header='infer', image_shape=hparams.image_shape, file_format='.jpg'):
        'Initialization'
        create_csv_data()
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.data_frame = pd.read_csv(csv_file, header=header)
        self.transform = transform
        self.image_shape = hparams.image_shape
        self.file_format = file_format


        temp = [x[0] for x in list(map(literal_eval, list(self.data_frame.iloc[:,2])))]
        temp = pd.DataFrame(temp)
        temp = temp.groupby([0]).size().reset_index(name='counts')
        self.breed_data = np.array(list(temp))
        self.breeds = np.array(list(temp.iloc[:,0]))
        self.breed_count = np.array(list(temp.iloc[:,1]))

        self.breed_to_idx = {breed: idx for idx, breed in enumerate(self.breeds)}
        print(self.breed_to_idx)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_frame)

  def __getitem__(self, index):
        'Generates one sample of data'
        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[index, 0]+self.file_format)
        image = Image.open(img_name)

        image_size = self.data_frame.iloc[index, 1]

        object = literal_eval(self.data_frame.iloc[index, 2])
        dog_breed = object[0]
        bbox = object[1]

        image = image.crop(bbox)
        image = image.resize(hparams.image_shape, Image.ANTIALIAS)

        if self.transform:
            image = self.transform(image)

        return (image, self.breed_to_idx[dog_breed], self.data_frame.iloc[index, 0])


def read_content(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    file_name = xml_file.split('/')[-1]

    for sz in root.iter('size'):
        width = int(sz.find("width").text)
        height = int(sz.find("height").text)
        depth = int(sz.find("depth").text)

    image_size = (width, height, depth)

    objects = []
    object_list = []

    for obj in root.iter('object'):

        dog_name = obj.find('name').text
        dog_pose = obj.find('pose').text
        truncated = obj.find('truncated').text
        difficult = obj.find('difficult').text

        for bx in obj.iter('bndbox'):
            xmin = int(bx.find("xmin").text)
            ymin = int(bx.find("ymin").text)
            xmax = int(bx.find("xmax").text)
            ymax = int(bx.find("ymax").text)
        bbox = (xmin, ymin, xmax, ymax)

        object = (dog_name, bbox, dog_pose, truncated, difficult)

        objects.append(object)
        object_list.append((file_name, image_size, object))

    return file_name, image_size, objects, object_list


def create_csv_data():
    data = []
    object_data = []

    breed_list = os.listdir(hparams.annotation_dir)
    file_names = [list(map(lambda x: hparams.annotation_dir+breed+'/'+x, os.listdir(hparams.annotation_dir+breed))) for breed in breed_list]


    for breed in file_names:
        for fname in breed:
            file_name, image_size, objects, object_list = read_content(fname)
            content = [file_name, image_size, objects]
            data.append(content)
            object_data += object_list

    data_frame = pd.DataFrame(data)
    print(data_frame.head())
    data_frame.to_csv(hparams.images_csv, index=False, header=['img_name', 'img_size', 'objects'])
    print('images.csv written in input')

    data_frame = pd.DataFrame(object_data)
    print(data_frame.head())
    data_frame.to_csv(hparams.train_csv, index=False, header=['img_name', 'img_size', 'object'])
    print('train.csv written in input')
