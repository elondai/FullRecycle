import os
import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

from fid_score import *
from inception import InceptionV3
from inception_score import *

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--real-path', type=str,
                    help='Path to real images')
parser.add_argument('--generated-path', type=str,
                    help='Path to generated images')
parser.add_argument('--batch-size', type=int, default=4,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')
parser.add_argument('--impl', type=str, default='torch', help='Implementation detail.')

# Inception score: https://github.com/sbarratt/inception-score-pytorch
# FID score: https://github.com/mseitzer/pytorch-fid

# python evaluate.py --real-path=../pytorch-CycleGAN-and-pix2pix/testdata/CelebA/
#                    --generated-path=../pytorch-CycleGAN-and-pix2pix/results/rescale64/test_latest/images/
#                    --batch-size=4 -c GPU
if __name__ == '__main__':
    args = parser.parse_args()
    if args.gpu != '':
        #print('Use CUDA')
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    #print('using impl %s' % args.impl)

    if args.impl == 'torch':
        class IgnoreLabelDataset(torch.utils.data.Dataset):
            def __init__(self, path, transforms_):
                self.imgs = np.asarray(list(glob.glob(path + '*')))
                self.transform = transforms_

            def __getitem__(self, index):
                img = Image.open(self.imgs[index])
                img = self.transform(img)
                return img

            def __len__(self):
                return len(self.imgs)

        transforms_ = transforms.Compose([transforms.Resize(32, Image.BICUBIC),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = IgnoreLabelDataset(args.generated_path, transforms_)

        inception_score = inception_score(dataset, cuda=True, batch_size=args.batch_size, resize=True, splits=10)
    else:
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        imgs = np.asarray(list(glob.glob(args.generated_path + '*_fake_B*')))
        # resize & open
        imgs = np.array([np.array(Image.open(i).resize((224, 224), Image.BICUBIC)) for i in imgs])
        # normalize
        imgs = imgs / 255.0
        # switch the axes
        # imgs = np.rollaxis(imgs, 3, 1)
        inception_score = inception_score_keras(imgs, batch_size=args.batch_size, splits=10)

    fid_value = calculate_fid_given_paths([args.real_path, args.generated_path],
                                          args.batch_size,
                                          args.gpu != '',
                                          args.dims)
    # print(fid_value)
    # IS mean, IS stdev, py, pyx, FID
    print('result:', '\t', inception_score[0], '\t', inception_score[1], '\t', fid_value)