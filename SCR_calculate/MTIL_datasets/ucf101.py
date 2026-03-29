import os
import pickle
import random
import math
from collections import defaultdict
from typing import List, Tuple

from .utils import *  # Datum, DatasetBase, mkdir_if_missing, read_json, write_json, listdir_nohidden


# Canonical class names as provided
UCF101_CANONICAL: List[str] = [
    'Apply Eye Makeup',
    'Apply Lipstick',
    'Archery',
    'Baby Crawling',
    'Balance Beam',
    'Band Marching',
    'Baseball Pitch',
    'Basketball',
    'Basketball Dunk',
    'Bench Press',
    'Biking',
    'Billiards',
    'Blow Dry Hair',
    'Blowing Candles',
    'Body Weight Squats',
    'Bowling',
    'Boxing Punching Bag',
    'Boxing Speed Bag',
    'Breast Stroke',
    'Brushing Teeth',
    'Clean And Jerk',
    'Cliff Diving',
    'Cricket Bowling',
    'Cricket Shot',
    'Cutting In Kitchen',
    'Diving',
    'Drumming',
    'Fencing',
    'Field Hockey Penalty',
    'Floor Gymnastics',
    'Frisbee Catch',
    'Front Crawl',
    'Golf Swing',
    'Haircut',
    'Hammer Throw',
    'Hammering',
    'Hand Stand Pushups',
    'Handstand Walking',
    'Head Massage',
    'High Jump',
    'Horse Race',
    'Horse Riding',
    'Hula Hoop',
    'Ice Dancing',
    'Javelin Throw',
    'Juggling Balls',
    'Jump Rope',
    'Jumping Jack',
    'Kayaking',
    'Knitting',
    'Long Jump',
    'Lunges',
    'Military Parade',
    'Mixing',
    'Mopping Floor',
    'Nunchucks',
    'Parallel Bars',
    'Pizza Tossing',
    'Playing Cello',
    'Playing Daf',
    'Playing Dhol',
    'Playing Flute',
    'Playing Guitar',
    'Playing Piano',
    'Playing Sitar',
    'Playing Tabla',
    'Playing Violin',
    'Pole Vault',
    'Pommel Horse',
    'Pull Ups',
    'Punch',
    'Push Ups',
    'Rafting',
    'Rock Climbing Indoor',
    'Rope Climbing',
    'Rowing',
    'Salsa Spin',
    'Shaving Beard',
    'Shotput',
    'Skate Boarding',
    'Skiing',
    'Skijet',
    'Sky Diving',
    'Soccer Juggling',
    'Soccer Penalty',
    'Still Rings',
    'Sumo Wrestling',
    'Surfing',
    'Swing',
    'Table Tennis Shot',
    'Tai Chi',
    'Tennis Swing',
    'Throw Discus',
    'Trampoline Jumping',
    'Typing',
    'Uneven Bars',
    'Volleyball Spiking',
    'Walking With Dog',
    'Wall Pushups',
    'Writing On Board',
    'Yo Yo',
]


UCF101_TEMPLATES: List[str] = [
    'a photo of a person {}.',
    'a video of a person {}.',
    'a example of a person {}.',
    'a demonstration of a person {}.',
    'a photo of the person {}.',
    'a video of the person {}.',
    'a example of the person {}.',
    'a demonstration of the person {}.',
    'a photo of a person using {}.',
    'a video of a person using {}.',
    'a example of a person using {}.',
    'a demonstration of a person using {}.',
    'a photo of the person using {}.',
    'a video of the person using {}.',
    'a example of the person using {}.',
    'a demonstration of the person using {}.',
    'a photo of a person doing {}.',
    'a video of a person doing {}.',
    'a example of a person doing {}.',
    'a demonstration of a person doing {}.',
    'a photo of the person doing {}.',
    'a video of the person doing {}.',
    'a example of the person doing {}.',
    'a demonstration of the person doing {}.',
    'a photo of a person during {}.',
    'a video of a person during {}.',
    'a example of a person during {}.',
    'a demonstration of a person during {}.',
    'a photo of the person during {}.',
    'a video of the person during {}.',
    'a example of the person during {}.',
    'a demonstration of the person during {}.',
    'a photo of a person performing {}.',
    'a video of a person performing {}.',
    'a example of a person performing {}.',
    'a demonstration of a person performing {}.',
    'a photo of the person performing {}.',
    'a video of the person performing {}.',
    'a example of the person performing {}.',
    'a demonstration of the person performing {}.',
    'a photo of a person practicing {}.',
    'a video of a person practicing {}.',
    'a example of a person practicing {}.',
    'a demonstration of a person practicing {}.',
    'a photo of the person practicing {}.',
    'a video of the person practicing {}.',
    'a example of the person practicing {}.',
    'a demonstration of the person practicing {}.',
]


class UCF101(DatasetBase):
    """
    UCF101 midframes classification dataset adapter for MTIL.
    Expected structure:
      root/ucf101/
        UCF-101-midframes/<Class_Folder>/<image>.jpg
        split_zhou_UCF101.json
    """

    dataset_dir = "ucf101"

    def __init__(self, root, num_shots=0, seed=1, subsample_classes='all'):
        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "UCF-101-midframes")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_UCF101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            # Fallback: build from directory and split train/val; use val as test too.
            trainval = self._read_from_dir()
            train, val = self.split_trainval(trainval)
            test = list(val)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        # Optionally subsample classes (keep interface consistent)
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample_classes)

        self.templates = UCF101_TEMPLATES

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args

        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]
        else:
            selected = labels[m:]
        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)

        return output

    # ---- IO helpers (compatible with OxfordPets) ----
    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)
        split = {"train": train, "val": val, "test": test}
        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])
        return train, val, test

    @staticmethod
    def split_trainval(trainval: List[Datum], p_val=0.2) -> Tuple[List[Datum], List[Datum]]:
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            tracker[item.label].append(idx)
        train, val = [], []
        for label, idxs in tracker.items():
            n_val = max(1, round(len(idxs) * p_val))
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)
        return train, val

    # ---- Directory reader (fallback if JSON is missing) ----
    def _read_from_dir(self) -> List[Datum]:
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Build canonical key mapping (lowercased, remove spaces/underscores)
        def canon_key(s: str) -> str:
            return ''.join(ch for ch in s.lower() if ch.isalnum())

        canonical_to_label = {name: idx for idx, name in enumerate(UCF101_CANONICAL)}
        key_to_canonical = {canon_key(name): name for name in UCF101_CANONICAL}

        items: List[Datum] = []
        class_dirs = listdir_nohidden(self.image_dir, sort=True)
        for cls_dir in class_dirs:
            cls_path = os.path.join(self.image_dir, cls_dir)
            if not os.path.isdir(cls_path):
                continue
            # Try to map folder name to canonical class
            k = canon_key(cls_dir.replace('_', ' '))
            cname = key_to_canonical.get(k, None)
            if cname is None:
                # Try removing underscores without space
                k2 = canon_key(cls_dir.replace('_', ''))
                cname = key_to_canonical.get(k2, None)
            if cname is None:
                # As a last resort, use the folder name with underscores replaced
                cname = cls_dir.replace('_', ' ').strip()
                if cname not in canonical_to_label:
                    # Unknown class; skip
                    continue
            label = canonical_to_label[cname]
            # Collect images
            for vid in listdir_nohidden(cls_path, sort=False):
                vpath = os.path.join(cls_path, vid)
                if os.path.isdir(vpath):
                    # some datasets might nest frames under video folder; include all frames
                    for frame in listdir_nohidden(vpath, sort=False):
                        impath = os.path.join(vpath, frame)
                        if os.path.isfile(impath):
                            items.append(Datum(impath=impath, label=label, classname=cname))
                else:
                    # direct frames under class folder
                    if os.path.isfile(vpath):
                        items.append(Datum(impath=vpath, label=label, classname=cname))
        return items
