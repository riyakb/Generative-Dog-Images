import argparse
import os
from train import *
from hparams import hparams

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')


args = parser.parse_args()


if args.mode == 'train':
    train()

# best_kappa = 0
#
#
# if args.mode == 'test':
#     try:
#         for i in range(1, 100):
#             kappa = test(hparams.model_dir+'model.'+str(i))
#             if kappa > best_kappa:
#                 best_kappa = kappa
#                 os.system('cp '+hparams.model_dir+'model.'+str(i)+' '+hparams.model_dir+'best_model')
#                 print(' -------->>>>> Best model updated with kappa = {0} -- using model.'.format(best_kappa)+str(i)+' <<<<<-------- ')
#     except:
#         test(hparams.model_dir+'best_model')
#         test_on_big(hparams.model_dir+'best_model')
