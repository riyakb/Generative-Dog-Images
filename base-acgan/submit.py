from __future__ import print_function, division
import os
import json
import csv
import torch
import random
import code
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.utils import save_image
from torch.autograd import Variable
from torch import optim
from data import APTOSData, OptimizedRounder
from model import Rater
from metric import quadratic_weighted_kappa, accuracy_metrics
from hparams import hparams

def submit(model_path=hparams.model, send_stats=False):

    test_dataset = APTOSData(csv_file=hparams.submission_csv,
                        root_dir=hparams.test_dir,
                        split=1.0,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ]),
                        ds_type='submit', file_format='.png')

    test_loader = DataLoader(test_dataset, batch_size=64,
                            shuffle=False, num_workers=2)

    print('loaded test data of length :'+str(len(test_loader)))

    if hparams.cuda:
        model = Rater(hparams.image_shape, hparams.num_classes, pretrained=False).cuda(hparams.gpu_device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = Rater(hparams.image_shape, hparams.num_classes, pretrained=False)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    print('Model loaded')

    Tensor = torch.cuda.FloatTensor if hparams.cuda else torch.FloatTensor

    rounder = OptimizedRounder()

    print('Submitting model results on {0} examples. '.format(len(test_loader)))
    with open('submission.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id_code', 'diagnosis'])
        with torch.no_grad():
            pred_labels_list = []
            labels_list = []
            img_names_list = []
            for (img, labels, img_names) in tqdm(test_loader):
                img = Variable(img.type(Tensor), requires_grad=False)
                labels = Variable(labels.float(), requires_grad=False)
                if hparams.cuda:
                    img = img.cuda(hparams.gpu_device)
                    labels = labels.cuda(hparams.gpu_device)
                pred_logits, _ = model(img)
                pred_labels = torch.tensor(rounder.predict(pred_logits.view(-1), hparams.coefficients))
                img_names_list += list(img_names)
                pred_labels_list.append(pred_labels.view(-1))
                labels_list.append(labels.view(-1))

            pred_labels = torch.cat(pred_labels_list, dim=0)

            for i in range(len(img_names_list)):
                writer.writerow([img_names_list[i], pred_labels[i].item()])

    print('Submission file saved as submission.csv')
