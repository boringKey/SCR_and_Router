import os
from typing import List

from .utils import *  
from .oxford_pets import OxfordPets

try:
    from torchvision.datasets import CIFAR10 as TorchCIFAR10
except Exception as e:
    TorchCIFAR10 = None
    print(f"Warning: torchvision not available for CIFAR10: {e}")


CIFAR10_CLASSES: List[str] = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

CIFAR10_TEMPLATES: List[str] = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]


class CIFAR10(DatasetBase):

    dataset_dir = "cifar10"

    def __init__(self, root, num_shots=0, seed=1, subsample_classes='all'):
        if TorchCIFAR10 is None:
            raise ImportError("torchvision is required for CIFAR10 dataset. Please install torchvision.")

        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        # Use torchvision to download/load data to dataset_dir
        train_ds = TorchCIFAR10(root=self.dataset_dir, train=True, download=True)
        test_ds = TorchCIFAR10(root=self.dataset_dir, train=False, download=True)

        trainval = []
        for idx in range(len(train_ds.data)):
            img = Image.fromarray(train_ds.data[idx])
            label = int(train_ds.targets[idx])
            classname = CIFAR10_CLASSES[label]
            trainval.append(Datum(impath=img, label=label, classname=classname))

        test = []
        for idx in range(len(test_ds.data)):
            img = Image.fromarray(test_ds.data[idx])
            label = int(test_ds.targets[idx])
            classname = CIFAR10_CLASSES[label]
            test.append(Datum(impath=img, label=label, classname=classname))

        train, val = OxfordPets.split_trainval(trainval)

        if num_shots >= 1:
            train = self.generate_fewshot_dataset(train, num_shots=num_shots)
            val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))

        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample_classes)

        self.templates = CIFAR10_TEMPLATES

        super().__init__(train_x=train, val=val, test=test)
