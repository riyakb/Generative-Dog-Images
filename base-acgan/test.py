import time
import code
import os, torch
import torch
import csv
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.utils import save_image
from torch.autograd import Variable
from torch import optim
from data import APTOSData
from model import Rater, OptimizedRounder
from metric import quadratic_weighted_kappa, accuracy_metrics
from hparams import hparams


def test(model_path=hparams.model, send_stats=False):

    test_dataset = APTOSData(csv_file=hparams.train_csv,
                        root_dir=hparams.train_dir,
                        split=0.95,
                        test_split=0.95,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ]),
                        ds_type='test', file_format='.png')

    extra_test_dataset = APTOSData(csv_file=hparams.dg_test_csv,
                        root_dir=hparams.dg_test_dir,
                        split=0,
                        test_split=0.0,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ]),
                        ds_type='test', file_format='.jpg')

    test_loader = DataLoader(test_dataset, batch_size=64,
                            shuffle=False, num_workers=2)

    extra_test_loader = DataLoader(extra_test_dataset, batch_size=64,
                            shuffle=False, num_workers=2)

    test_loaders = [test_loader, extra_test_loader]

    # print('loaded test data of length :'+str(len(test_loader)))

    if hparams.cuda:
        model = Rater(hparams.image_shape, hparams.num_classes, pretrained=False).cuda(hparams.gpu_device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = Rater(hparams.image_shape, hparams.num_classes, pretrained=False)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    # print('Model loaded')

    Tensor = torch.cuda.FloatTensor if hparams.cuda else torch.FloatTensor

    rounder = OptimizedRounder()

    print('Testing model on {0} examples. '.format(len(test_loader)))
    with open('submission.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id_code', 'diagnosis'])
        with torch.no_grad():
            pred_labels_list = []
            labels_list = []
            img_names_list = []
            for loader in tqdm(test_loaders):
                for i, (img, labels, img_names) in enumerate(loader):
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
            labels = torch.cat(labels_list, dim=0)

            for i in range(len(img_names_list)):
                writer.writerow([img_names_list[i], pred_labels[i].item()])

    if send_stats:
        kappa, precision, recall, f1, accuracy, precision_list, recall_list, f1_list, accuracy_list = accuracy_metrics(pred_labels, labels, True)
    kappa, precision, recall, f1, accuracy = accuracy_metrics(pred_labels, labels, False)

    print('== Test on -- '+model_path+' == kappa - {0:.4f}, precision - {1:.4f}, recall - {2:.4f}, f1 - {3:.4f}, accuracy - {4:.4f} =='.format(kappa, precision, recall, f1, accuracy))

    return kappa

def test_on_big(model_path=hparams.model, send_stats=False):

    big_test_dataset = APTOSData(csv_file=hparams.big_train_csv,
                        root_dir=hparams.big_train_dir,
                        split=0.80,
                        test_split=0.80,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ]),
                        ds_type='test', file_format='.jpeg')

    big_test_loader = DataLoader(big_test_dataset, batch_size=64,
                            shuffle=False, num_workers=2)

    test_loaders = [big_test_loader]

    # print('loaded test data of length :'+str(len(test_loader)))

    if hparams.cuda:
        model = Rater(hparams.image_shape, hparams.num_classes, pretrained=False).cuda(hparams.gpu_device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = Rater(hparams.image_shape, hparams.num_classes, pretrained=False)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    # print('Model loaded')

    Tensor = torch.cuda.FloatTensor if hparams.cuda else torch.FloatTensor

    rounder = OptimizedRounder()

    print('Testing model on {0} examples. '.format(len(big_test_loader)))
    with open('big_submission.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id_code', 'diagnosis'])
        with torch.no_grad():
            pred_labels_list = []
            labels_list = []
            img_names_list = []
            for loader in tqdm(test_loaders):
                for i, (img, labels, img_names) in enumerate(loader):
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
            labels = torch.cat(labels_list, dim=0)

            for i in range(len(img_names_list)):
                writer.writerow([img_names_list[i], pred_labels[i].item()])

    if send_stats:
        kappa, precision, recall, f1, accuracy, precision_list, recall_list, f1_list, accuracy_list = accuracy_metrics(pred_labels, labels, True)
    kappa, precision, recall, f1, accuracy = accuracy_metrics(pred_labels, labels, False)

    print('== Test on big_test_data -- '+model_path+' == kappa - {0:.4f}, precision - {1:.4f}, recall - {2:.4f}, f1 - {3:.4f}, accuracy - {4:.4f} =='.format(kappa, precision, recall, f1, accuracy))

    return kappa
