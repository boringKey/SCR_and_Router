import os
import pickle
import random
import warnings

from .utils import *
from .oxford_pets import OxfordPets

# Canonical class list for RESISC45 (order defines label ids 0..44)
RESISC_CLASSES = [
    'airplane',
    'airport',
    'baseball diamond',
    'basketball court',
    'beach',
    'bridge',
    'chaparral',
    'church',
    'circular farmland',
    'cloud',
    'commercial area',
    'dense residential',
    'desert',
    'forest',
    'freeway',
    'golf course',
    'ground track field',
    'harbor',
    'industrial area',
    'intersection',
    'island',
    'lake',
    'meadow',
    'medium residential',
    'mobile home park',
    'mountain',
    'overpass',
    'palace',
    'parking lot',
    'railway',
    'railway station',
    'rectangular farmland',
    'river',
    'roundabout',
    'runway',
    'sea ice',
    'ship',
    'snowberg',
    'sparse residential',
    'stadium',
    'storage tank',
    'tennis court',
    'terrace',
    'thermal power station',
    'wetland',
]

# Prompt templates (strings) as provided
RESISC_TEMPLATES = [
    'satellite imagery of {}.',
    'aerial imagery of {}.',
    'satellite photo of {}.',
    'aerial photo of {}.',
    'satellite view of {}.',
    'aerial view of {}.',
    'satellite imagery of a {}.',
    'aerial imagery of a {}.',
    'satellite photo of a {}.',
    'aerial photo of a {}.',
    'satellite view of a {}.',
    'aerial view of a {}.',
    'satellite imagery of the {}.',
    'aerial imagery of the {}.',
    'satellite photo of the {}.',
    'aerial photo of the {}.',
    'satellite view of the {}.',
    'aerial view of the {}.',
]

RESISC_DEBUG = os.environ.get("RESISC_DEBUG", "0") not in ("0", "false", "False", "")

def _dbg(msg: str):
    if RESISC_DEBUG:
        print(f"[RESISC45][DEBUG] {msg}")


def _norm_name(s: str) -> str:
    s = s.lower().strip()
    for ch in [" ", "_", "-", "."]:
        s = s.replace(ch, "")
    return s


class RESISC45(DatasetBase):

    dataset_dir = "resisc45"

    def __init__(self, root, num_shots=0, seed=1, subsample_classes='all'):
        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = self.dataset_dir  # images are under class subfolders directly
        self.split_path = os.path.join(self.dataset_dir, "split_custom_RESISC45.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        _dbg(f"dataset_dir={self.dataset_dir}")

        if os.path.exists(self.split_path):
            try:
                train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
            except Exception as e:
                warnings.warn(f"RESISC45: failed to read split file '{self.split_path}'; rebuilding. Error: {e}")
                train, val, test = self.read_and_split_data(self.image_dir)
                try:
                    OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)
                except Exception as e2:
                    warnings.warn(f"RESISC45: failed to save rebuilt split to '{self.split_path}': {e2}")
        else:
            train, val, test = self.read_and_split_data(self.image_dir)
            try:
                OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)
            except Exception as e:
                warnings.warn(f"RESISC45: failed to save split to '{self.split_path}': {e}")

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
        self.templates = RESISC_TEMPLATES
        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_and_split_data(image_dir, p_trn=0.5, p_val=0.2):
        try:
            categories = listdir_nohidden(image_dir, sort=True)
        except Exception as e:
            warnings.warn(f"RESISC45: failed to list directory '{image_dir}': {e}")
            raise
        categories = [c for c in categories if os.path.isdir(os.path.join(image_dir, c))]
        cat_norm = {_norm_name(c): c for c in categories}

        missing = []
        class_to_dir = {}
        for cname in RESISC_CLASSES:
            key = _norm_name(cname)
            if key in cat_norm:
                class_to_dir[cname] = cat_norm[key]
            else:
                missing.append(cname)
        if missing:
            warnings.warn(f"RESISC45: missing class folders for {len(missing)} classes: {missing[:5]}{' ...' if len(missing)>5 else ''}")

        def _collate(paths, y, cname):
            return [Datum(impath=p, label=y, classname=cname) for p in paths]

        train, val, test = [], [], []
        for y, cname in enumerate(RESISC_CLASSES):
            if cname not in class_to_dir:
                continue
            cdir = os.path.join(image_dir, class_to_dir[cname])
            try:
                images = listdir_nohidden(cdir, sort=False)
            except Exception as e:
                warnings.warn(f"RESISC45: failed to list class folder '{cdir}': {e}")
                images = []
            images = [os.path.join(cdir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            if n_total == 0:
                warnings.warn(f"RESISC45: empty class folder {cdir}; skipping")
                continue
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            if not (n_train > 0 and n_val > 0 and n_test > 0):
                # Fallback to keep all splits non-empty where possible
                if n_total >= 3:
                    n_train, n_val, n_test = 1, 1, n_total - 2
                elif n_total == 2:
                    n_train, n_val, n_test = 1, 1, 0
                else:  # 1
                    n_train, n_val, n_test = 1, 0, 0

            train.extend(_collate(images[:n_train], y, cname))
            val.extend(_collate(images[n_train:n_train + n_val], y, cname))
            test.extend(_collate(images[n_train + n_val:], y, cname))

        return train, val, test
