import os
import random
from typing import List, Tuple, Dict

from .utils import *  # Datum, DatasetBase, listdir_nohidden, mkdir_if_missing
from .oxford_pets import OxfordPets


IMAGENETR_TEMPLATES: List[str] = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


class ImageNetR(DatasetBase):

    dataset_dir = "imagenet-r"

    def __init__(self, root: str, num_shots: int = 0, seed: int = 1, subsample_classes: str = 'all', test_ratio: float = 0.2):
        """
        Expect directory structure:
            {root}/imagenet-r/<wnid>/*.jpg|png|jpeg|bmp|webp
            {root}/imagenet-r/classname.txt  # lines: "<wnid> <human_readable_name>"
        There is no official test split; we perform a per-class split into train/test, then
        split train into train/val using OxfordPets.split_trainval.
        """
        rnd = random.Random(seed)

        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_custom_ImageNetR.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # Read class mapping from classname.txt
        class_map_path = os.path.join(self.dataset_dir, "classname.txt")
        if not os.path.exists(class_map_path):
            raise FileNotFoundError(f"Class mapping file not found: {class_map_path}")
        wnids, classnames = self._read_class_map(class_map_path)
        self._wnids = wnids
        self._classnames_ref = classnames

        # Build or load split
        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            trainval, test = self._read_from_folders(self.dataset_dir, wnids, classnames, rnd=rnd, test_ratio=test_ratio)
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

        # Expose templates
        self.templates = IMAGENETR_TEMPLATES

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def _read_class_map(filepath: str) -> Tuple[List[str], List[str]]:
        wnids: List[str] = []
        cnames: List[str] = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                wnid = parts[0]
                cname = ' '.join(parts[1:]) if len(parts) > 1 else wnid
                wnids.append(wnid)
                cnames.append(cname)
        return wnids, cnames

    @staticmethod
    def _split_trainval_safe(trainval: List[Datum], p_val: float = 0.2) -> Tuple[List[Datum], List[Datum]]:
        from collections import defaultdict
        tracker: Dict[int, List[int]] = defaultdict(list)
        for idx, item in enumerate(trainval):
            tracker[item.label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n = len(idxs)
            if n <= 1:
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
        if len(val) == 0 and len(train) > 0:
            val.append(train[-1])
            train = train[:-1]
        return train, val

    def _read_from_folders(self, data_dir: str, wnids: List[str], classnames: List[str], rnd: random.Random, test_ratio: float) -> Tuple[List[Datum], List[Datum]]:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        class_to_label = {wnid: i for i, wnid in enumerate(wnids)}
        label_to_cname = {i: classnames[i] for i in range(len(classnames))}

        items_by_label: Dict[int, List[Datum]] = {i: [] for i in range(len(wnids))}
        for wnid in wnids:
            cdir = os.path.join(data_dir, wnid)
            if not os.path.isdir(cdir):
                # If a class listed in mapping has no folder, skip gracefully
                continue
            label = class_to_label[wnid]
            cname = label_to_cname[label]
            for fname in listdir_nohidden(cdir, sort=True):
                fext = os.path.splitext(fname)[1].lower()
                if fext not in exts:
                    continue
                impath = os.path.join(cdir, fname)
                items_by_label[label].append(Datum(impath=impath, label=label, classname=cname))

        trainval: List[Datum] = []
        test: List[Datum] = []
        for label, items in items_by_label.items():
            if not items:
                continue
            rnd.shuffle(items)
            if len(items) == 1:
                trainval.extend(items)
                continue
            n_test = max(1, int(round(len(items) * test_ratio)))
            if n_test >= len(items):
                n_test = len(items) - 1
            test.extend(items[:n_test])
            trainval.extend(items[n_test:])
        return trainval, test
