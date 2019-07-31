import time
import code
import os, torch
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler
from torch import optim
from data import DogData
from model import Generator, Discriminator
from hparams import hparams
from functools import reduce
import operator
from tqdm import tqdm


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def train(resume=False):

    train_dataset = DogData(csv_file=hparams.train_csv,
                        root_dir=hparams.images_dir,
                        file_format='.jpg',
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ]))

    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size,
                            num_workers=2, shuffle=True)

    train_loaders = [train_loader]

    print('loaded train data of length :'+str(reduce(operator.add, map(len, train_loaders))))

    if hparams.cuda:
        auxiliary_loss = torch.nn.CrossEntropyLoss().cuda(hparams.gpu_device)
        adversarial_loss = torch.nn.BCELoss().cuda(hparams.gpu_device)
        generator = Generator().cuda(hparams.gpu_device)
        discriminator = Discriminator().cuda(hparams.gpu_device)
    else:
        auxiliary_loss = torch.nn.CrossEntropyLoss()
        adversarial_loss = torch.nn.BCELoss()
        generator = Generator()
        discriminator = Discriminator()

    params_count = 0
    for model in [generator, discriminator]:
        for param in model.parameters():
            params_count += np.prod(param.size())
    print('Model has {0} trainable parameters'.format(params_count))

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=hparams.learning_rate, betas=(hparams.momentum1, hparams.momentum2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=hparams.learning_rate/2, betas=(hparams.momentum1, hparams.momentum2))

    FloatTensor = torch.cuda.FloatTensor if hparams.cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if hparams.cuda else torch.LongTensor

    train_log = np.zeros((4, 1+int(hparams.num_epochs*reduce(operator.add, map(len, train_loaders)))))
    snap_log = np.zeros((4, 1+int(hparams.num_epochs*reduce(operator.add, map(len, train_loaders))/hparams.snap_interval)))

    fixed_noise = Variable(FloatTensor(np.random.normal(0, 1, (hparams.n_classes, hparams.num_sample, hparams.latent_dim))))

    def validation(generator, gen_dir=hparams.gen_imgs, name='gen', send_stats=False):
        print('Sampling {0} images from generator.'.format(hparams.num_sample*hparams.n_classes))
        os.makedirs(gen_dir, exist_ok=True)
        with torch.no_grad():
            generated_images = []
            for num in tqdm(range(hparams.n_classes)):
                labels = np.array([num for _ in range(hparams.num_sample)])
                labels = Variable(LongTensor(labels))
                gen_imgs = generator(fixed_noise[num], labels)
                generated_images.append(gen_imgs)
                if (num+1) % 30 == 0:
                    generated_images = torch.cat(generated_images, dim=0)
                    save_image(generated_images, gen_dir+str(name)+'_'+str(num//30)+'.png', nrow=30, normalize=True)
                    generated_images = []

    print('Starting training..')
    batch, snap, batch_to_snap = 0, 0, 0
    n_one = 0
    start_time = time.time()

    # print(model)

    for epoch in range(hparams.num_epochs):
        for loader in train_loaders:
            for i, (imgs, labels, img_names) in enumerate(loader):

                batch_size = imgs.shape[0]

                # Adversarial ground truths
                valid = Variable(FloatTensor(batch_size, 1).uniform_(0.8, 1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).uniform_(0.0, 0.2), requires_grad=False)
                # if hparams.cuda:
                #     valid = Variable(FloatTensor(torch.cuda.FloatTensor(imgs.shape[0], 1)).uniform_(0.8, 1.0)), requires_grad=False)
                #     fake = Variable(FloatTensor(torch.cuda.FloatTensor(imgs.shape[0], 1)).uniform_(0.0, 0.2)), requires_grad=False)
                # else:
                #     valid = Variable(FloatTensor(torch.FloatTensor(imgs.shape[0]).uniform_(0.8, 1.0)), requires_grad=False)
                #     fake = Variable(FloatTensor(torch.FloatTensor(imgs.shape[0]).uniform_(0.0, 0.2)), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(FloatTensor))
                labels = Variable(labels.type(LongTensor))

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, hparams.latent_dim))))
                gen_labels = Variable(LongTensor(np.random.randint(0, hparams.n_classes, batch_size)))

                # Generate a batch of images
                gen_imgs = generator(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity, pred_label = discriminator(gen_imgs)
                g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                if i % hparams.skip_disc_update == 0:
                    optimizer_D.zero_grad()

                    # Loss for real images
                    real_pred, real_aux = discriminator(real_imgs)
                    d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

                    # Loss for fake images
                    fake_pred, fake_aux = discriminator(gen_imgs.detach())
                    d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

                    # Total discriminator loss
                    d_loss = (d_real_loss + d_fake_loss) / 2

                    # Calculate discriminator accuracy
                    pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                    gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                    d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                    d_loss.backward()
                    optimizer_D.step()


                train_log[0, batch] = g_loss.item()
                train_log[1, batch] = d_loss.item()
                snap_log[0,snap] += g_loss.item()
                snap_log[1,snap] += d_loss.item()
                batch_to_snap += 1

                if batch % hparams.print_interval == 0 and batch > 0:
                    print('[Epoch - {0}/{1}, batch - {2:.3f}, generator_loss - {3:.6f}, discriminator_loss - {4:.3f}, aux_accuracy - {5:.3f}]'.\
                    format(epoch, hparams.num_epochs, 100.0*i/len(loader), g_loss.item(), d_loss.item(), d_acc))
                if batch % hparams.snap_interval == 0 and batch > 0:
                    batch_to_snap = 0
                    validation(generator, gen_dir=hparams.gen_imgs, name=str(snap+1))
                    # print('SNAP -- {0:.3f} === [Epoch - {1:.1f}, Batch No - {2:.1f}, Snap No. - {3:.1f}, train_kappa - {4:.3f}, precision - {5:.6f}, recall - {6:.6f}, F1-score - {7:.6f},  train_accuracy - {8:.3f},\n\
                    #  validation_kappa - {9:.3f}, val_precision - {10:.6f}, val_recall - {11:.6f}, val_F1-score - {12:.6f}, validation_accuracy - {13:.3f},\n\
                    #  big_kappa - {14:.3f}, big_precision - {15:.6f}, big_recall - {16:.6f}, big_F1-score - {17:.6f}, big_accuracy - {18:.3f}]'\
                    #     .format(time.time()-start_time, 1.0*epoch, 1.0*batch, 1.0*snap, snap_log[1, snap], snap_log[2, snap], snap_log[3, snap], snap_log[4, snap],
                    #      snap_log[5, snap], valid_kappa, valid_precision, valid_recall, valid_f1, valid_acc, big_valid_kappa, big_valid_precision, big_valid_recall, big_valid_f1, big_valid_acc))
                    snap += 1
                    start_time = time.time()
                if batch % hparams.model_interval == 0 and batch > 0:
                    np.savetxt(hparams.result_dir+'train_log.csv', train_log, delimiter=",")
                    np.savetxt(hparams.result_dir+'snap_log.csv', snap_log, delimiter=",")
                    torch.save({
                        'epoch': epoch,
                        'batch': batch,
                        'model_state_dict': generator.state_dict(),
                        'optimizer_state_dict': optimizer_G.state_dict(),
                        'loss': g_loss,
                        }, hparams.generator+'.'+str(batch//hparams.model_interval))
                    torch.save({
                        'epoch': epoch,
                        'batch': batch,
                        'model_state_dict': discriminator.state_dict(),
                        'optimizer_state_dict': optimizer_D.state_dict(),
                        'loss': d_loss,
                        }, hparams.discriminator+'.'+str(batch//hparams.model_interval))
                    print('model saved. log files saved.')

                batch += 1
