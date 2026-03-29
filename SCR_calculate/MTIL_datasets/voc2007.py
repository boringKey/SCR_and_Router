import os
import os.path as osp
from typing import List, Dict, Tuple
from collections import defaultdict

from .utils import Datum, DatasetBase, listdir_nohidden


VOC2007_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'dog', 'horse', 'motorbike', 'person', 'sheep',
    'sofa', 'diningtable', 'pottedplant', 'train', 'tvmonitor',
]

VOC2007_TEMPLATES = [
    'a photo of a {}.',
]


class VOC2007(DatasetBase):
    """
    VOC2007 multi-label classification dataset adapter for MTIL evaluation.

    Expects directory structure:
      data/VOC2007/
        JPEGImages/
        Main/
          <class>_trainval.txt
          <class>_test.txt

    Each *_split.txt has lines: "<image_id> <label>", where label in {1, 0, -1}.
    We treat 1 as positive and others as negative for that class.
    """

    dataset_dir = 'VOC2007'

    def __init__(self, root: str, seed: int = 32, single_label: bool = False):
        self.root = os.path.expanduser(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.images_dir = osp.join(self.dataset_dir, 'JPEGImages')
        self.main_dir = osp.join(self.dataset_dir, 'Main')

        classes = VOC2007_CLASSES
        class_to_label = {c: i for i, c in enumerate(classes)}

        if not single_label:
            # Multi-label mode (for Zero-Shot evaluation)
            test_list = self._read_split('test', classes)
            test = self._build_data_list(test_list, class_to_label)

            # Create a meta train set with one sample per class (integer labels) for proper initialization
            # Try to point to a real image containing that class as positive; fallback to any available image or empty path
            meta_train: List[Datum] = []
            for ci, cname in enumerate(classes):
                impath = None
                for d in test:
                    try:
                        if isinstance(d.label, (list, tuple)) and ci < len(d.label) and int(d.label[ci]) == 1:
                            impath = d.impath
                            break
                    except Exception:
                        continue
                if impath is None:
                    impath = test[0].impath if len(test) > 0 else ''
                meta_train.append(Datum(impath=impath, label=ci, classname=cname))

            train_x: List[Datum] = meta_train
            val: List[Datum] = []
            self.templates = VOC2007_TEMPLATES
            super().__init__(train_x=train_x, val=val, test=test)
        else:
            # Single-label mode (for CIL downstream training/evaluation)
            # Build trainval and test splits as single-label datasets by selecting the first positive class per image
            trainval_multi = self._read_split('trainval', classes)
            test_multi = self._read_split('test', classes)
            train_x = self._build_single_label_list(trainval_multi, classes)
            val = []
            test = self._build_single_label_list(test_multi, classes)
            self.templates = VOC2007_TEMPLATES
            super().__init__(train_x=train_x, val=val, test=test)

    def _read_split(self, split: str, classes: List[str]) -> Dict[str, List[int]]:
        """Return mapping: image_id -> multi-hot list for the given split."""
        # Collect image ids present in this split
        img_ids = set()
        by_class_labels: Dict[str, Dict[str, int]] = {}
        for cname in classes:
            split_file = osp.join(self.main_dir, f"{cname}_{split}.txt")
            if not osp.isfile(split_file):
                # Try alternate common path name
                split_file = osp.join(self.dataset_dir, 'ImageSets', 'Main', f"{cname}_{split}.txt")
            if not osp.isfile(split_file):
                # If missing, skip this class
                continue
            class_map: Dict[str, int] = {}
            with open(split_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    img_id, label_str = parts[0], parts[1]
                    try:
                        label = int(label_str)
                    except Exception:
                        continue
                    class_map[img_id] = 1 if label == 1 else 0
                    img_ids.add(img_id)
            by_class_labels[cname] = class_map

        # Aggregate into multi-hot vectors
        result: Dict[str, List[int]] = {}
        for img_id in img_ids:
            vec = [0] * len(classes)
            for ci, cname in enumerate(classes):
                cmap = by_class_labels.get(cname, {})
                vec[ci] = int(cmap.get(img_id, 0))
            result[img_id] = vec
        return result

    def _build_data_list(self, id_to_vec: Dict[str, List[int]], class_to_label: Dict[str, int]) -> List[Datum]:
        data: List[Datum] = []
        for img_id, vec in id_to_vec.items():
            # Try common image extensions
            impath = None
            for ext in ['.jpg', '.jpeg', '.png']:
                p = osp.join(self.images_dir, img_id + ext)
                if osp.isfile(p):
                    impath = p
                    break
            if impath is None:
                # As a fallback, if there's exactly one file starting with img_id
                try:
                    candidates = [f for f in listdir_nohidden(self.images_dir) if f.startswith(img_id + '.')]
                    if candidates:
                        impath = osp.join(self.images_dir, candidates[0])
                except Exception:
                    pass
            if impath is None:
                # Skip missing images
                continue
            # For multi-label, we store the full vector as the label
            data.append(Datum(impath=impath, label=vec, classname=''))
        return data

    def _build_single_label_list(self, id_to_vec: Dict[str, List[int]], classes: List[str]) -> List[Datum]:
        data: List[Datum] = []
        for img_id, vec in id_to_vec.items():
            # choose the first positive class; skip if none
            try:
                cls_idx = next((i for i, v in enumerate(vec) if int(v) == 1), None)
            except Exception:
                cls_idx = None
            if cls_idx is None:
                continue
            # image path resolution
            impath = None
            for ext in ['.jpg', '.jpeg', '.png']:
                p = osp.join(self.images_dir, img_id + ext)
                if osp.isfile(p):
                    impath = p
                    break
            if impath is None:
                try:
                    candidates = [f for f in listdir_nohidden(self.images_dir) if f.startswith(img_id + '.')]
                    if candidates:
                        impath = osp.join(self.images_dir, candidates[0])
                except Exception:
                    pass
            if impath is None:
                continue
            data.append(Datum(impath=impath, label=int(cls_idx), classname=classes[cls_idx]))
        return data
