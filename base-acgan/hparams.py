import os
import torch

class Hparams():
    def __init__(self):

        self.cuda = True if torch.cuda.is_available() else False

        """
        Data Parameters
        """

        # os.makedirs('../input', exist_ok=True)
        os.makedirs('../model', exist_ok=True)
        os.makedirs('../results/', exist_ok=True)

        self.exp_name = 'base-acgan/'

        self.annotation_dir = '../input/annotation/Annotation/'
        self.images_dir = '../input/all-dogs/all-dogs/'

        self.images_csv = '../input/images.csv'
        self.train_csv = '../input/train.csv'

        """
        Model Parameters
        """

        os.makedirs('../model/', exist_ok=True)

        self.image_shape = (64, 64)
        self.img_shape = (3, 64, 64)
        self.img_size = 64
        self.channels = 3

        self.n_classes = 120

        self.latent_dim = 100


        """
        Training parameters
        """

        self.num_epochs = 300
        self.batch_size = 64

        self.learning_rate = 0.0002
        self.n_critic = 2
        self.skip_disc_update = 1
        self.num_sample = 10

        self.momentum1 = 0.5
        self.momentum2 = 0.999

        self.model_interval = 900
        self.print_interval = 50
        self.snap_interval = 300

        self.gpu_device = 'cuda:0'

        self.model_dir = '../model/'+self.exp_name
        self.result_dir = '../results/'+self.exp_name

        os.makedirs(self.model_dir, exist_ok=True)

        self.generator = self.model_dir + 'generator'
        self.discriminator = self.model_dir + 'discriminator'

        self.gen_imgs = '../results/'+self.exp_name+'gen_imgs/'


hparams = Hparams()
