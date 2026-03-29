import os
import random
from typing import List, Tuple

from .utils import *  # Datum, DatasetBase, listdir_nohidden, mkdir_if_missing
from .oxford_pets import OxfordPets

# Class names per user specification (ordered)
KITTI_DISTANCE_CLASSES: List[str] = [
    'a photo i took of a car nearby',
    'a photo i took with a car in the middle distance',
    'a photo i took with a car faraway',
    'a photo i took with no car.',
]

# Keep templates as strings with {}
KITTI_DISTANCE_TEMPLATES: List[str] = [
    '{}',
]


class KittiDistance(DatasetBase):

    dataset_dir = "kitti"

    def __init__(self, root, num_shots=0, seed=1, subsample_classes='all', test_ratio: float = 0.2):
        """
        Predict the distance category of the closest car from KITTI labels.

        Expected structure:
            {root}/kitti/
                image/000004.png
                label/000004.txt

        Label file format: KITTI object detection label lines.
        We retain only lines with type == 'Car'.
        For these, we read the 14th column (index 13, loc_z) as the camera-depth in meters.
        We take the minimum positive loc_z across all 'Car' lines as z_min for the image.
        If there is no valid positive loc_z for 'Car', we assign the 'no car' class.

        Discretization into classes:
            0: 0 < z_min < 10         -> 'nearby'
            1: 10 <= z_min < 30       -> 'middle distance'
            2: z_min >= 30            -> 'faraway'
            3: no car                 -> 'no car.'

        No official test set; we split randomly (per class) into train/test by test_ratio.
        Then we split train into train/val using a safe per-class split.
        Splits are saved to JSON for reproducibility.
        """
        rnd = random.Random(seed)

        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        # Image and label directories
        image_dir = os.path.join(self.dataset_dir, 'image')
        if not os.path.isdir(image_dir):
            # Graceful fallback to standard KITTI naming variants if provided
            alt1 = os.path.join(self.dataset_dir, 'image_2')
            alt2 = os.path.join(self.dataset_dir, 'images')
            if os.path.isdir(alt1):
                image_dir = alt1
            elif os.path.isdir(alt2):
                image_dir = alt2
        label_dir = os.path.join(self.dataset_dir, 'label')
        if not os.path.isdir(label_dir):
            alt_l1 = os.path.join(self.dataset_dir, 'label_2')
            alt_l2 = os.path.join(self.dataset_dir, 'labels')
            if os.path.isdir(alt_l1):
                label_dir = alt_l1
            elif os.path.isdir(alt_l2):
                label_dir = alt_l2

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"KittiDistance: image dir not found: {image_dir}")
        if not os.path.isdir(label_dir):
            raise FileNotFoundError(f"KittiDistance: label dir not found: {label_dir}")

        self.image_dir = image_dir
        self.label_dir = label_dir

        self.split_path = os.path.join(self.dataset_dir, "split_custom_KittiDistance.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            trainval, test = self._read_and_split(self.image_dir, self.label_dir, rnd=rnd, test_ratio=test_ratio)
            train, val = self._split_trainval_safe(trainval)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

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

        # Allow class subsampling if requested
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample_classes)

        # Templates per user specification
        self.templates = KITTI_DISTANCE_TEMPLATES

        super().__init__(train_x=train, val=val, test=test)
        # Override inferred metadata to ensure stable 4-way classification regardless of train label coverage
        self._classnames = KITTI_DISTANCE_CLASSES
        self._lab2cname = {i: c for i, c in enumerate(KITTI_DISTANCE_CLASSES)}
        self._num_classes = len(KITTI_DISTANCE_CLASSES)

    def _read_and_split(self, image_dir: str, label_dir: str, rnd: random.Random, test_ratio: float) -> Tuple[List[Datum], List[Datum]]:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        # Collect items with computed labels
        items_by_label = {i: [] for i in range(len(KITTI_DISTANCE_CLASSES))}

        for fname in listdir_nohidden(image_dir, sort=True):
            fext = os.path.splitext(fname)[1].lower()
            if fext not in exts:
                continue
            impath = os.path.join(image_dir, fname)
            stem = os.path.splitext(fname)[0]
            label_path = os.path.join(label_dir, stem + '.txt')
            label_idx = self._compute_label_from_kitti(label_path)
            cname = KITTI_DISTANCE_CLASSES[label_idx]
            items_by_label[label_idx].append(Datum(impath=impath, label=label_idx, classname=cname))

        # Per-class balanced split into trainval/test
        trainval, test = [], []
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

    @staticmethod
    def _compute_label_from_kitti(label_path: str) -> int:
        """
        Parse KITTI label file; focus only on lines with type == 'Car'.
        Extract the 14th column (index 13, loc_z) as meters; consider only positive values.
        If no positive loc_z found -> class 'no car' (index 3).

        Thresholds:
            0: 0 < z < 10
            1: 10 <= z < 30
            2: z >= 30
        """
        z_vals = []
        if os.path.isfile(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    obj_type = parts[0]
                    if obj_type != 'Car':
                        continue
                    if len(parts) < 14:
                        # Not a valid KITTI detection line; skip
                        continue
                    try:
                        # parts[13] is loc_z (0-based index), per KITTI format
                        z = float(parts[13])
                        if z > 0:
                            z_vals.append(z)
                    except Exception:
                        continue
        # Determine class index
        if not z_vals:
            return 3  # no car
        z_min = min(z_vals)
        if z_min < 10:
            return 0
        elif z_min < 20:
            return 1
        else:
            return 2

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
