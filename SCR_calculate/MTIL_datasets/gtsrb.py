import os
import pickle
import random
import warnings

from .utils import *
from .oxford_pets import OxfordPets

# 43 classes for GTSRB with human-readable names
classes = [
    'red and white circle 20 kph speed limit',
    'red and white circle 30 kph speed limit',
    'red and white circle 50 kph speed limit',
    'red and white circle 60 kph speed limit',
    'red and white circle 70 kph speed limit',
    'red and white circle 80 kph speed limit',
    'end / de-restriction of 80 kph speed limit',
    'red and white circle 100 kph speed limit',
    'red and white circle 120 kph speed limit',
    'red and white circle red car and black car no passing',
    'red and white circle red truck and black car no passing',
    'red and white triangle road intersection warning',
    'white and yellow diamond priority road',
    'red and white upside down triangle yield right-of-way',
    'stop',
    'empty red and white circle',
    'red and white circle no truck entry',
    'red circle with white horizonal stripe no entry',
    'red and white triangle with exclamation mark warning',
    'red and white triangle with black left curve approaching warning',
    'red and white triangle with black right curve approaching warning',
    'red and white triangle with black double curve approaching warning',
    'red and white triangle rough / bumpy road warning',
    'red and white triangle car skidding / slipping warning',
    'red and white triangle with merging / narrow lanes warning',
    'red and white triangle with person digging / construction / road work warning',
    'red and white triangle with traffic light approaching warning',
    'red and white triangle with person walking warning',
    'red and white triangle with child and person walking warning',
    'red and white triangle with bicyle warning',
    'red and white triangle with snowflake / ice warning',
    'red and white triangle with deer warning',
    'white circle with gray strike bar no speed limit',
    'blue circle with white right turn arrow mandatory',
    'blue circle with white left turn arrow mandatory',
    'blue circle with white forward arrow mandatory',
    'blue circle with white forward or right turn arrow mandatory',
    'blue circle with white forward or left turn arrow mandatory',
    'blue circle with white keep right arrow mandatory',
    'blue circle with white keep left arrow mandatory',
    'blue circle with white arrows indicating a traffic circle',
    'white circle with gray strike bar indicating no passing for cars has ended',
    'white circle with gray strike bar indicating no passing for trucks has ended',
]


DEBUG = os.environ.get("GTSRB_DEBUG", "0") not in ("0", "false", "False", "")


def _dbg(msg: str):
    if DEBUG:
        print(f"[GTSRB][DEBUG] {msg}")


def _is_image_file(name):
    name = name.lower()
    return any(name.endswith(ext) for ext in ['.ppm', '.png', '.jpg', '.jpeg', '.bmp', '.webp'])


class GTSRB(DatasetBase):
    """
    German Traffic Sign Recognition Benchmark (GTSRB)

    Expected structure:
      <root>/gtsrb/
        ├─ 00000/*.ppm
        ├─ 00001/*.ppm
        └─ ... up to 00042/

    Note: The dataset has no official test split in this layout, so we will
    randomly split each class into train/val/test (50%/20%/30%), cached to json.
    """

    dataset_dir = "gtsrb"

    def __init__(self, root, num_shots=0, seed=1, subsample_classes='all'):
        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = self.dataset_dir
        self.split_path = os.path.join(self.dataset_dir, "split_custom_GTSRB.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        _dbg(f"dataset_dir={self.dataset_dir}")
        _dbg(f"image_dir={self.image_dir}")
        _dbg(f"split_path exists? {os.path.exists(self.split_path)}")

        if os.path.exists(self.split_path):
            try:
                train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
            except Exception as e:
                warnings.warn(f"GTSRB: failed to read split file '{self.split_path}'; rebuilding split. Error: {e}")
                train, val, test = self.read_and_split_data(self.image_dir)
                try:
                    OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)
                except Exception as e2:
                    warnings.warn(f"GTSRB: failed to save rebuilt split to '{self.split_path}': {e2}")
        else:
            train, val, test = self.read_and_split_data(self.image_dir)
            try:
                OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)
            except Exception as e:
                warnings.warn(f"GTSRB: failed to save split to '{self.split_path}': {e}")

        _dbg(f"loaded counts: train={len(train)}, val={len(val)}, test={len(test)}")

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
            _dbg(f"few-shot applied: train={len(train)}, val={len(val)} (shots={num_shots})")

        # Ensure class subsampling behavior matches others
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample_classes)
        _dbg(f"after subsample='{subsample_classes}': train={len(train)}, val={len(val)}, test={len(test)})")

        # Prompt templates provided by user
        self.templates = [
            lambda c: f'a zoomed in photo of a "{c}" traffic sign.',
            lambda c: f'a centered photo of a "{c}" traffic sign.',
            lambda c: f'a close up photo of a "{c}" traffic sign.',
        ]

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_and_split_data(image_root, p_trn=0.5, p_val=0.2):
        # Discover class directories (e.g., 00000 .. 00042)
        try:
            all_entries = listdir_nohidden(image_root, sort=True)
        except Exception as e:
            warnings.warn(f"GTSRB: failed to list directory '{image_root}': {e}")
            raise
        class_dirs = [d for d in all_entries if os.path.isdir(os.path.join(image_root, d))]
        class_dirs.sort()
        assert len(class_dirs) > 0, f"GTSRB: no class folders found under {image_root}"
        _dbg(f"found {len(class_dirs)} class folders; head={class_dirs[:5]}")

        # Map sorted class dirs to labels 0..N-1 and names from 'classes'
        if len(class_dirs) != len(classes):
            print(f"Warning: detected {len(class_dirs)} class folders but classes list has {len(classes)} entries. Proceeding with min overlap.")
        num_labels = min(len(class_dirs), len(classes))

        def _collate(paths, y, cname):
            return [Datum(impath=p, label=y, classname=cname) for p in paths]

        train, val, test = [], [], []
        for y, cls_dir in enumerate(class_dirs[:num_labels]):
            cname = classes[y]
            cdir = os.path.join(image_root, cls_dir)
            try:
                files = listdir_nohidden(cdir, sort=False)
            except Exception as e:
                warnings.warn(f"GTSRB: failed to list class folder '{cdir}': {e}")
                files = []
            imgs = [os.path.join(cdir, f) for f in files if _is_image_file(f)]
            random.shuffle(imgs)
            n_total = len(imgs)
            if n_total == 0:
                warnings.warn(f"GTSRB: empty or unreadable class folder: {cdir}; skipping this class")
                continue
            if n_total < 5:
                warnings.warn(f"GTSRB: very few images in class '{cls_dir}' (n={n_total}); splits may be unstable")
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            if not (n_train > 0 and n_val > 0 and n_test > 0):
                warnings.warn(f"GTSRB: split would create empty split for class {cls_dir} (n={n_total}); adjusting strategy to keep at least 1 per split")
                # Fallback: enforce at least 1 per split if possible
                if n_total >= 3:
                    n_train, n_val, n_test = 1, 1, n_total - 2
                elif n_total == 2:
                    n_train, n_val, n_test = 1, 1, 0
                else:  # n_total == 1
                    n_train, n_val, n_test = 1, 0, 0

            train.extend(_collate(imgs[:n_train], y, cname))
            val.extend(_collate(imgs[n_train:n_train + n_val], y, cname))
            test.extend(_collate(imgs[n_train + n_val:], y, cname))
            _dbg(f"class {cls_dir} -> label {y}: total={n_total}, train={n_train}, val={n_val}, test={n_test}")

        _dbg(f"aggregate sizes: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test
