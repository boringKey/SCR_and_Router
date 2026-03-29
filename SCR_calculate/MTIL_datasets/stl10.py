import os
import math
import random
from typing import List, Tuple

from .utils import *  # Datum, DatasetBase, listdir_nohidden
from .oxford_pets import OxfordPets

STL10_CLASSES: List[str] = [
    'airplane',
    'bird',
    'car',
    'cat',
    'deer',
    'dog',
    'horse',
    'monkey',
    'ship',
    'truck',
]

# keep templates as strings with {}
STL10_TEMPLATES: List[str] = [
    'a photo of a {}.',
    'a photo of the {}.',
]


class STL10(DatasetBase):

    dataset_dir = "stl10"

    def __init__(self, root, num_shots=0, seed=1, subsample_classes='all', test_ratio: float = 0.2):
        """
        Expect directory structure:
            {root}/stl10/<class_name>/*.png|jpg|jpeg|bmp|webp
        No official test set provided; we split per-class into train/test.
        Then we further split train into train/val using OxfordPets.split_trainval.
        """
        rnd = random.Random(seed)

        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_custom_STL10.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # If we have a saved split, reuse for reproducibility
        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            trainval, test = self._read_from_folders(self.dataset_dir, rnd=rnd, test_ratio=test_ratio)
            train, val = self._split_trainval_safe(trainval)
            OxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)

        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                import pickle
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                import pickle
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample_classes)

        self.templates = STL10_TEMPLATES

        super().__init__(train_x=train, val=val, test=test)

    def _read_from_folders(self, data_dir: str, rnd: random.Random, test_ratio: float) -> Tuple[List[Datum], List[Datum]]:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        class_to_label = {name: i for i, name in enumerate(STL10_CLASSES)}

        items_by_label = {i: [] for i in range(len(STL10_CLASSES))}
        for cname in STL10_CLASSES:
            cdir = os.path.join(data_dir, cname)
            if not os.path.isdir(cdir):
                raise FileNotFoundError(f"Class folder not found: {cdir}")
            for fname in listdir_nohidden(cdir, sort=True):
                fext = os.path.splitext(fname)[1].lower()
                if fext not in exts:
                    continue
                impath = os.path.join(cdir, fname)
                label = class_to_label[cname]
                items_by_label[label].append(Datum(impath=impath, label=label, classname=cname))

        trainval, test = [], []
        for label, items in items_by_label.items():
            if not items:
                continue
            rnd.shuffle(items)
            if len(items) == 1:
                # keep the only sample for training to preserve class presence in train
                trainval.extend(items)
                continue
            n_test = max(1, int(round(len(items) * test_ratio)))
            if n_test >= len(items):
                n_test = len(items) - 1
            test.extend(items[:n_test])
            trainval.extend(items[n_test:])

        return trainval, test

    @staticmethod
    def _split_trainval_safe(trainval: List[Datum], p_val: float = 0.2) -> Tuple[List[Datum], List[Datum]]:
        from collections import defaultdict
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            tracker[item.label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n = len(idxs)
            if n <= 1:
                # not enough to create a val sample for this class
                for idx in idxs:
                    train.append(trainval[idx])
                continue
            n_val = max(1, int(round(n * p_val)))
            if n_val >= n:
                n_val = n - 1
            random.shuffle(idxs)
            for i, idx in enumerate(idxs):
                if i < n_val:
                    val.append(trainval[idx])
                else:
                    train.append(trainval[idx])

        # If val ended up empty (degenerate tiny dataset), move one from train to val
        if len(val) == 0 and len(train) > 0:
            val.append(train[-1])
            train = train[:-1]

        return train, val
