import os
import pickle
import random
from typing import List, Tuple

import h5py
import numpy as np

from .utils import Datum, DatasetBase, mkdir_if_missing, read_json, write_json
from .oxford_pets import OxfordPets


PCAM_CLASSES = [
    'lymph node tissue without metastatic tumor',
    'metastatic tumor in lymph node tissue',
]

# Multiple domain-specific templates for histopathology microscopy images
PCAM_TEMPLATES = [
    'a microscopy image patch of {}',
    'a histopathology image of {}',
    'a hematoxylin and eosin stained image of {}',
    'a high-resolution histology patch of {}',
    'a digital pathology slide patch of {}',
    'this is a microscopy image of {}',
    'this is a histopathology image of {}',
]


class PCam(DatasetBase):

    dataset_dir = 'pcam'

    def __init__(self, root, num_shots: int = 0, seed: int = 1, subsample_classes: str = 'all', val_ratio: float = 0.2):
        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        mkdir_if_missing(self.dataset_dir)

        # HDF5 file paths
        self.train_x_path = os.path.join(self.dataset_dir, 'camelyonpatch_level_2_split_train_x.h5')
        self.train_y_path = os.path.join(self.dataset_dir, 'camelyonpatch_level_2_split_train_y.h5')
        self.test_x_path = os.path.join(self.dataset_dir, 'camelyonpatch_level_2_split_test_x.h5')
        self.test_y_path = os.path.join(self.dataset_dir, 'camelyonpatch_level_2_split_test_y.h5')

        for p in [self.train_x_path, self.train_y_path, self.test_x_path, self.test_y_path]:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"PCam: expected file not found: {p}")

        self.split_path = os.path.join(self.dataset_dir, 'split_custom_PCam.json')
        self.split_fewshot_dir = os.path.join(self.dataset_dir, 'split_fewshot')
        mkdir_if_missing(self.split_fewshot_dir)

        random.seed(seed)
        np.random.seed(seed)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path)
        else:
            trainval = self._read_train(self.train_x_path, self.train_y_path)
            test = self._read_test(self.test_x_path, self.test_y_path)
            train, val = self.split_trainval(trainval, p_val=val_ratio)
            self.save_split(train, val, test, self.split_path)

        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, 'rb') as f:
                    data = pickle.load(f)
                    train, val = data['train'], data['val']
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {'train': train, 'val': val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Optional class subsampling (kept for consistency with other datasets)
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample_classes)

        # Templates per user specification
        self.templates = PCAM_TEMPLATES

        # Debug: print split sizes and label histograms
        def _hist(items):
            from collections import Counter
            cnt = Counter([it.label for it in items])
            # ensure keys present for both binary classes
            out = {i: int(cnt.get(i, 0)) for i in range(len(PCAM_CLASSES))}
            return out
        try:
            print(
                "PCam stats: "
                f"train={len(train)} (hist={_hist(train)}), "
                f"val={len(val)} (hist={_hist(val)}), "
                f"test={len(test)} (hist={_hist(test)})"
            )
        except Exception as e:
            print(f"PCam stats printing failed: {e}")

        super().__init__(train_x=train, val=val, test=test)
        # Ensure stable binary classification metadata
        self._classnames = PCAM_CLASSES
        self._lab2cname = {i: c for i, c in enumerate(PCAM_CLASSES)}
        self._num_classes = len(PCAM_CLASSES)

    @staticmethod
    def _first_key(h5_path: str) -> str:
        with h5py.File(h5_path, 'r') as f:
            keys = list(f.keys())
            if not keys:
                raise RuntimeError(f"No datasets found in H5 file: {h5_path}")
            return keys[0]

    @staticmethod
    def _read_labels(h5_path: str) -> np.ndarray:
        key = PCam._first_key(h5_path)
        with h5py.File(h5_path, 'r') as f:
            y = f[key][...]
        y = np.asarray(y).squeeze()
        y = y.astype(np.int64)
        return y

    def _read_train(self, x_path: str, y_path: str) -> List[Datum]:
        x_key = self._first_key(x_path)
        y = self._read_labels(y_path)
        items: List[Datum] = []
        for i, label in enumerate(y.tolist()):
            label_i = int(label)
            classname = PCAM_CLASSES[label_i]
            # Lazy image reference: ('h5', abs_h5_path, dataset_key, index)
            abs_path = os.path.abspath(x_path)
            impath = ('h5', abs_path, x_key, i)
            items.append(Datum(impath=impath, label=label_i, classname=classname))
        return items

    def _read_test(self, x_path: str, y_path: str) -> List[Datum]:
        x_key = self._first_key(x_path)
        y = self._read_labels(y_path)
        items: List[Datum] = []
        for i, label in enumerate(y.tolist()):
            label_i = int(label)
            classname = PCAM_CLASSES[label_i]
            abs_path = os.path.abspath(x_path)
            impath = ('h5', abs_path, x_key, i)
            items.append(Datum(impath=impath, label=label_i, classname=classname))
        return items

    @staticmethod
    def split_trainval(trainval: List[Datum], p_val: float = 0.2) -> Tuple[List[Datum], List[Datum]]:
        from collections import defaultdict
        p_trn = 1 - p_val
        print(f"Splitting PCam train into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            tracker[item.label].append(idx)
        train, val = [], []
        for _, idxs in tracker.items():
            n_val = max(1, round(len(idxs) * p_val))
            random.shuffle(idxs)
            for n, i in enumerate(idxs):
                if n < n_val:
                    val.append(trainval[i])
                else:
                    train.append(trainval[i])
        return train, val

    def save_split(self, train: List[Datum], val: List[Datum], test: List[Datum], filepath: str):
        def _ser(items: List[Datum]):
            out = []
            for it in items:
                impath = it.impath
                if isinstance(impath, tuple) and len(impath) == 4 and impath[0] == 'h5':
                    tag, abs_path, key, idx = impath
                    # store relative path for portability
                    rel = os.path.relpath(abs_path, self.dataset_dir)
                    impath_ser = [tag, rel, key, int(idx)]
                else:
                    raise ValueError('PCam expects H5 tuple paths')
                out.append((impath_ser, int(it.label), it.classname))
            return out
        split = {
            'train': _ser(train),
            'val': _ser(val),
            'test': _ser(test),
        }
        write_json(split, filepath)
        print(f"Saved PCam split to {filepath}")

    def read_split(self, filepath: str):
        def _deser(items):
            out = []
            for impath_ser, label, classname in items:
                if isinstance(impath_ser, (list, tuple)) and len(impath_ser) == 4 and impath_ser[0] == 'h5':
                    tag, rel, key, idx = impath_ser
                    fpath = os.path.join(self.dataset_dir, rel)
                    impath = (tag, fpath, key, int(idx))
                else:
                    raise ValueError('PCam split contains invalid path entries')
                out.append(Datum(impath=impath, label=int(label), classname=classname))
            return out
        print(f"Reading PCam split from {filepath}")
        split = read_json(filepath)
        train = _deser(split['train'])
        val = _deser(split['val'])
        test = _deser(split['test'])
        return train, val, test
