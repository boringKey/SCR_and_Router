import os
import pickle
from typing import List

from .utils import Datum, DatasetBase, mkdir_if_missing
from .oxford_pets import OxfordPets
from .utils import read_json, write_json

# Class list and templates as specified
CLEVR_COUNT_CLASSES: List[str] = [
    '10', '3', '4', '5', '6', '7', '8', '9'
]

CLEVR_COUNT_TEMPLATES: List[str] = [
    'a photo of {} objects.',
]


class CLEVRCount(DatasetBase):

    dataset_dir = 'clevr'

    def __init__(self, root, num_shots: int = 0, seed: int = 1, subsample_classes: str = 'all'):
        # Root and directories
        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.images_dir = os.path.join(self.dataset_dir, 'images')
        self.scenes_dir = os.path.join(self.dataset_dir, 'scenes')
        self.split_path = os.path.join(self.dataset_dir, 'split_custom_CLEVRCount.json')
        self.split_fewshot_dir = os.path.join(self.dataset_dir, 'split_fewshot')
        mkdir_if_missing(self.split_fewshot_dir)

        # Required files
        train_scenes = os.path.join(self.scenes_dir, 'CLEVR_train_scenes.json')
        val_scenes = os.path.join(self.scenes_dir, 'CLEVR_val_scenes.json')
        if not os.path.isfile(train_scenes) or not os.path.isfile(val_scenes):
            raise FileNotFoundError(
                f"CLEVRCount expects scenes JSON at {train_scenes} and {val_scenes}"
            )

        # Load or build split
        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            trainval = self._read_scenes(train_scenes, split='train')
            test = self._read_scenes(val_scenes, split='val')
            train, val = OxfordPets.split_trainval(trainval)
            OxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)

        # Few-shot
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

        # Optional class subsampling (base/new)
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample_classes)

        # Templates
        self.templates = CLEVR_COUNT_TEMPLATES

        # Debug stats
        try:
            def _hist(items: List[Datum]):
                from collections import Counter
                cnt = Counter([it.label for it in items])
                out = {i: int(cnt.get(i, 0)) for i in range(len(CLEVR_COUNT_CLASSES))}
                return out
            print(
                "CLEVRCount stats: "
                f"train={len(train)} (hist={_hist(train)}), "
                f"val={len(val)} (hist={_hist(val)}), "
                f"test={len(test)} (hist={_hist(test)})"
            )
        except Exception as e:
            print(f"CLEVRCount stats printing failed: {e}")

        super().__init__(train_x=train, val=val, test=test)
        # Ensure class metadata is stable and matches the provided list
        self._classnames = CLEVR_COUNT_CLASSES
        self._lab2cname = {i: c for i, c in enumerate(CLEVR_COUNT_CLASSES)}
        self._num_classes = len(CLEVR_COUNT_CLASSES)

    def _read_scenes(self, json_path: str, split: str) -> List[Datum]:
        """Read CLEVR scenes JSON and construct a list of Datum entries.
        Only images whose object count appears in CLEVR_COUNT_CLASSES are kept.
        """
        obj = read_json(json_path)
        scenes = obj.get('scenes', [])
        # Map count -> class index
        lab2idx = {int(c): i for i, c in enumerate(CLEVR_COUNT_CLASSES)}
        items: List[Datum] = []
        for sc in scenes:
            # Some files use key 'split', some typos list 'spit'; be robust
            image_filename = sc.get('image_filename', None)
            if not image_filename:
                continue
            n_objects = sc.get('objects', [])
            try:
                num = int(len(n_objects))
            except Exception:
                continue
            if num not in lab2idx:
                # skip counts not in the configured class list
                continue
            label_i = lab2idx[num]
            # Build absolute path to the image based on split
            if split not in ['train', 'val', 'test']:
                split_dir = 'train'
            else:
                split_dir = split
            impath = os.path.join(self.dataset_dir, 'images', split_dir, image_filename)
            items.append(Datum(impath=impath, label=label_i, classname=str(num)))
        return items
