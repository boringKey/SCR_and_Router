import os
import pickle
import random
import warnings
from collections import defaultdict

from .utils import *
from .oxford_pets import OxfordPets

# Synonym sets per class (label index is the list index)
FER2013_CLASSES_SYNONYMS = [
    ['angry'],
    ['disgusted', 'disgust'],
    ['fearful', 'fear'],
    ['happy', 'smiling'],
    ['sad', 'depressed'],
    ['surprised', 'surprise', 'shocked', 'spooked'],
    ['neutral', 'bored'],
]

# Canonical class names are the first synonym in each list
FER2013_CANONICAL = [syns[0] for syns in FER2013_CLASSES_SYNONYMS]

# Prompt templates
FER2013_TEMPLATES = [
    'a photo of a {} looking face.',
    'a photo of a face showing the emotion: {}.',
    'a photo of a face looking {}.',
    'a face that looks {}.',
    'they look {}.',
    'look at how {} they are.',
]

FER2013_DEBUG = os.environ.get("FER2013_DEBUG", "0") not in ("0", "false", "False", "")

def _dbg(msg: str):
    if FER2013_DEBUG:
        print(f"[FER2013][DEBUG] {msg}")


def _norm(s: str) -> str:
    s = s.lower().strip()
    for ch in [" ", "_", "-", "."]:
        s = s.replace(ch, "")
    return s


def _build_syn_map():
    m = {}
    for y, syns in enumerate(FER2013_CLASSES_SYNONYMS):
        for s in syns:
            m[_norm(s)] = y
    # add common canonical variants for safety
    aliases = {
        'disgust': 1,
        'fear': 2,
        'surprise': 5,
    }
    for k, v in aliases.items():
        m[_norm(k)] = v
    return m


class FER2013(DatasetBase):

    dataset_dir = "fer2013"

    def __init__(self, root, num_shots=0, seed=1, subsample_classes='all'):
        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_custom_FER2013.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        train_dir = os.path.join(self.dataset_dir, "train")
        test_dir = os.path.join(self.dataset_dir, "test")
        if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
            raise ValueError(
                f"FER2013: expected train/test folders under '{self.dataset_dir}'. Got train={os.path.isdir(train_dir)}, test={os.path.isdir(test_dir)}"
            )

        # try cache
        if os.path.exists(self.split_path):
            try:
                train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
            except Exception as e:
                warnings.warn(f"FER2013: failed to read cached split; rebuilding. Error: {e}")
                train, val, test = self._build_split(train_dir, test_dir)
                try:
                    OxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)
                except Exception as e2:
                    warnings.warn(f"FER2013: failed to save split: {e2}")
        else:
            train, val, test = self._build_split(train_dir, test_dir)
            try:
                OxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)
            except Exception as e:
                warnings.warn(f"FER2013: failed to save split: {e}")

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

        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample_classes)
        self.templates = FER2013_TEMPLATES
        super().__init__(train_x=train, val=val, test=test)

    def _build_split(self, train_dir, test_dir, p_val=0.2):
        syn_map = _build_syn_map()
        # read train per class
        tr_items_by_label = defaultdict(list)
        class_dirs = listdir_nohidden(train_dir, sort=True)
        if not class_dirs:
            warnings.warn(f"FER2013: no class folders found in {train_dir}")
        for cls in class_dirs:
            full = os.path.join(train_dir, cls)
            if not os.path.isdir(full):
                continue
            key = _norm(cls)
            y = syn_map.get(key)
            if y is None:
                warnings.warn(f"FER2013: unexpected class folder in train: '{cls}'")
                continue
            cname = FER2013_CANONICAL[y]
            for fname in listdir_nohidden(full):
                impath = os.path.join(full, fname)
                tr_items_by_label[y].append(Datum(impath=impath, label=y, classname=cname))

        # stratified split train->(train,val)
        train, val = [], []
        for y, items in tr_items_by_label.items():
            random.shuffle(items)
            n_val = max(1, round(len(items) * p_val)) if len(items) > 1 else 0
            val.extend(items[:n_val])
            train.extend(items[n_val:])

        # read test
        test = []
        class_dirs = listdir_nohidden(test_dir, sort=True)
        if not class_dirs:
            warnings.warn(f"FER2013: no class folders found in {test_dir}")
        for cls in class_dirs:
            full = os.path.join(test_dir, cls)
            if not os.path.isdir(full):
                continue
            key = _norm(cls)
            y = syn_map.get(key)
            if y is None:
                warnings.warn(f"FER2013: unexpected class folder in test: '{cls}'")
                continue
            cname = FER2013_CANONICAL[y]
            for fname in listdir_nohidden(full):
                impath = os.path.join(full, fname)
                test.append(Datum(impath=impath, label=y, classname=cname))

        # basic sanity
        if not train or not val or not test:
            warnings.warn(f"FER2013: split sizes train={len(train)} val={len(val)} test={len(test)}")
        return train, val, test
