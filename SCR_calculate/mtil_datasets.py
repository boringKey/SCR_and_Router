import os
import torch.nn as nn
import torch
import sys

from MTIL_datasets.caltech101 import Caltech101
from MTIL_datasets.cifar100 import CIFAR100
from MTIL_datasets.dtd import DescribableTextures as DTD
from MTIL_datasets.eurosat import EuroSAT
from MTIL_datasets.fgvc_aircraft import FGVCAircraft as Aircraft
from MTIL_datasets.food101 import Food101 as Food
from MTIL_datasets.mnist import MNIST
from MTIL_datasets.oxford_flowers import OxfordFlowers as Flowers
from MTIL_datasets.oxford_pets import OxfordPets as OxfordPet
from MTIL_datasets.stanford_cars import StanfordCars
from MTIL_datasets.sun397 import SUN397
from MTIL_datasets.country211 import Country211
from MTIL_datasets.utils import DatasetWrapper
from MTIL_datasets.sst2 import SST2
from MTIL_datasets.hatefulmemes import HatefulMemes
from MTIL_datasets.gtsrb import GTSRB
from MTIL_datasets.resisc import RESISC45
from MTIL_datasets.fer2013 import FER2013
from MTIL_datasets.ucf101 import UCF101
from MTIL_datasets.cifar10 import CIFAR10
from MTIL_datasets.stl10 import STL10
from MTIL_datasets.voc2007 import VOC2007
from MTIL_datasets.imagenet_r import ImageNetR
from MTIL_datasets.kitti_distance import KittiDistance
from MTIL_datasets.pcam import PCam
from MTIL_datasets.clevr_count import CLEVRCount


def get_dataset(cfg, split, transforms=None):
    if split == 'val' and (not cfg.use_validation):
        return None, None, None

    is_train = (split == 'train')
    templates = None
    if cfg.dataset == "MTIL":
        if cfg.MTIL_order_2:
            base_sets = [StanfordCars, Food, MNIST, OxfordPet, Flowers, SUN397, Aircraft, Caltech101, DTD, EuroSAT, CIFAR100, Country211, SST2, HatefulMemes, GTSRB, RESISC45, FER2013, UCF101, CIFAR10, STL10, VOC2007, ImageNetR, KittiDistance, PCam, CLEVRCount]
        else:
            base_sets = [Aircraft, Caltech101, CIFAR100, DTD, EuroSAT, Flowers, Food, MNIST, OxfordPet, StanfordCars, SUN397, Country211, SST2, HatefulMemes, GTSRB, RESISC45, FER2013, UCF101, CIFAR10, STL10, VOC2007, ImageNetR, KittiDistance, PCam, CLEVRCount]
        if cfg.train_one_dataset >= 0:
            base_sets = base_sets[cfg.train_one_dataset: cfg.train_one_dataset+1]
        dataset = []
        classes_names = []
        templates = []
        for base_set in base_sets:
            # VOC2007 special handling: when selected as downstream (train_one_dataset >= 0),
            # we need single-label data for CIL; for ZS (train_one_dataset == -1), keep multi-label.
            if base_set is VOC2007 and getattr(cfg, 'train_one_dataset', -1) >= 0:
                base = base_set(cfg.dataset_root, seed=cfg.seed, single_label=True)
            else:
                base = base_set(cfg.dataset_root, seed=cfg.seed)
            classes_names.append(base.classnames)
            # each dataset exposes a list of template callables/strings via `templates`
            templates.append(getattr(base, 'templates', None))
            if split == 'train':
                dataset.append(DatasetWrapper(base.train_x, transform=transforms, is_train=is_train))
            elif split == 'val':
                dataset.append(DatasetWrapper(base.val, transform=transforms, is_train=is_train))
            elif split == 'test':
                dataset.append(DatasetWrapper(base.test, transform=transforms, is_train=is_train))
    else:
        ValueError(f"'{cfg.dataset}' is a invalid dataset.")
    return dataset, classes_names, templates



def parse_sample(sample, is_train, task_id, cfg):
    return sample[0], sample[1], torch.IntTensor([task_id]).repeat(sample[0].size(0))