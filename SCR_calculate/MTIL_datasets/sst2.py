import os
import pickle

from .utils import *
from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD


class SST2(DatasetBase):
    """
    SST2 (rendered) dataset loader for MTIL.

    Expected structure (either of the following roots):
      <root>/sst2/
        ├─ train/{negative,positive}/*.png
        ├─ valid/{negative,positive}/*.png
        └─ test/{negative,positive}/*.png

      <root>/rendered-sst2/  (alias supported)
        ├─ train/{negative,positive}/*.png
        ├─ valid/{negative,positive}/*.png
        └─ test/{negative,positive}/*.png

    Classes: ['negative', 'positive']
    Templates: ['a {} review of a movie.']
    """

    dataset_dir = "sst2"

    def __init__(self, root, num_shots=0, seed=1, subsample_classes='all'):
        root = os.path.abspath(os.path.expanduser(root))
        primary_dir = os.path.join(root, self.dataset_dir)
        alt_dir = os.path.join(root, "rendered-sst2")

        if os.path.isdir(primary_dir):
            self.dataset_dir = primary_dir
        elif os.path.isdir(alt_dir):
            self.dataset_dir = alt_dir
        else:
            raise ValueError(
                "SST2: dataset folder not found. Expected one of: '{}' or '{}'".format(primary_dir, alt_dir)
            )

        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        train_dir = os.path.join(self.dataset_dir, "train")
        valid_dir = os.path.join(self.dataset_dir, "valid")
        test_dir = os.path.join(self.dataset_dir, "test")

        use_folder_splits = os.path.isdir(train_dir) and os.path.isdir(valid_dir) and os.path.isdir(test_dir)

        # fixed class order and validation
        classes = ["negative", "positive"]
        class_to_label = {c: i for i, c in enumerate(classes)}

        if use_folder_splits:
            train = self._read_split_dir(train_dir, class_to_label)
            val = self._read_split_dir(valid_dir, class_to_label)
            test = self._read_split_dir(test_dir, class_to_label)
        else:
            # Fallbacks consistent with other datasets: look for a JSON split, else naive split
            image_dir = os.path.join(self.dataset_dir, "images")
            self.image_dir = image_dir if os.path.isdir(image_dir) else self.dataset_dir
            split_path = os.path.join(self.dataset_dir, "split_zhou_SST2.json")
            if os.path.exists(split_path):
                train, val, test = OxfordPets.read_split(split_path, self.image_dir)
            else:
                train, val, test = DTD.read_and_split_data(self.image_dir)
                OxfordPets.save_split(train, val, test, split_path, self.image_dir)

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

        subsample = subsample_classes
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        # Template per user spec
        self.templates = [
            lambda c: f'a {c} review of a movie.',
        ]

        super().__init__(train_x=train, val=val, test=test)

    def _read_split_dir(self, split_dir, class_to_label):
        items = []
        if not os.path.isdir(split_dir):
            return items
        codes = listdir_nohidden(split_dir)
        # Validate folders: must be subset of expected classes and cover at least one class
        unexpected = sorted([c for c in codes if c not in class_to_label])
        if unexpected:
            raise ValueError(
                f"SST2: Found unexpected class folders in '{split_dir}': {unexpected}. "
                f"Expected only {list(class_to_label.keys())}."
            )
        for cls in codes:
            class_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(class_dir):
                continue
            label = class_to_label.get(cls)
            if label is None:
                raise ValueError(
                    f"SST2: Inconsistent label mapping for class '{cls}' in split '{split_dir}'."
                )
            cname = cls
            for fname in listdir_nohidden(class_dir):
                impath = os.path.join(class_dir, fname)
                items.append(Datum(impath=impath, label=label, classname=cname))
        return items
