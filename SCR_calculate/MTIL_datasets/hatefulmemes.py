import os
import json
import pickle
import logging

from .utils import *
from .oxford_pets import OxfordPets


class HatefulMemes(DatasetBase):
    """
    Hateful Memes dataset loader (image-only for MTIL).

    Expected structure:
      <root>/hatefulmemes/
        ├─ img/*.png|jpg
        ├─ train.jsonl
        ├─ dev.jsonl
        └─ test.jsonl

    JSONL lines example:
      {"id":85362, "img":"img/85362.png", "label":0, "text":"..."}

    We DO NOT use the 'text' field. Only image path and label are used.

    Classes: ['meme', 'hatespeech meme']
    Templates: ['a {}.']
    """

    dataset_dir = "hatefulmemes"

    def __init__(self, root, num_shots=0, seed=1, subsample_classes='all'):
        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        if not os.path.isdir(self.dataset_dir):
            # allow alias without trailing 's'
            alt = os.path.join(root, "hatefulmeme")
            if os.path.isdir(alt):
                self.dataset_dir = alt
            else:
                raise ValueError(
                    f"HatefulMemes: dataset folder not found at '{self.dataset_dir}' or '{alt}'"
                )
        # logging.info(f"HatefulMemes: using dataset_dir={self.dataset_dir}")

        split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(split_fewshot_dir)

        # paths
        train_json = os.path.join(self.dataset_dir, "train.jsonl")
        dev_json = os.path.join(self.dataset_dir, "dev.jsonl")
        test_json = os.path.join(self.dataset_dir, "test.jsonl")

        if not os.path.isfile(train_json) or not os.path.isfile(dev_json) or not os.path.isfile(test_json):
            raise ValueError(
                f"HatefulMemes: missing jsonl files. Expected train/dev/test at '{self.dataset_dir}'."
            )

        # fixed classes
        classes = ["meme", "hatespeech meme"]
        lab_to_name = {0: classes[0], 1: classes[1]}

        # read splits
        train = self._read_jsonl(train_json, lab_to_name)
        val = self._read_jsonl(dev_json, lab_to_name)
        test = self._read_jsonl(test_json, lab_to_name)
        # logging.info(f"HatefulMemes: parsed sizes -> train={len(train)}, val={len(val)}, test={len(test)}")
        # Some public releases of Hateful Memes do not include labels for test.jsonl.
        # In that case, fallback to use dev.jsonl for evaluation so zero-shot works.
        if len(test) == 0 and len(val) > 0:
            logging.warning(
                "HatefulMemes: test split yielded 0 items (likely unlabeled test.jsonl). Falling back to dev.jsonl as test for evaluation."
            )
            test = list(val)
            # logging.info(f"HatefulMemes: after test->dev fallback -> train={len(train)}, val={len(val)}, test={len(test)}")

        # few-shot
        if num_shots >= 1:
            preprocessed = os.path.join(split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
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

        # subsample behavior consistent with others
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample_classes)
        # logging.info(
        #     "HatefulMemes: after subsample='%s' -> train=%d, val=%d, test=%d",
        #     subsample_classes, len(train), len(val), len(test)
        # )

        # templates per user spec
        self.templates = [
            lambda c: f'a {c}.',
        ]

        super().__init__(train_x=train, val=val, test=test)
        # Log class mapping and sanity samples
        # logging.info("HatefulMemes: classnames=%s", self.classnames)
        try:
            exs = []
            for split_name, split in [("train", self.train_x), ("val", self.val), ("test", self.test)]:
                if split and len(split) > 0:
                    s0 = split[0]
                    exs.append(f"{split_name}: (label={s0.label}, cname={s0.classname}, exists={os.path.isfile(s0.impath)}) {s0.impath}")
            if exs:
                pass
                # logging.info("HatefulMemes: sample items -> %s", " | ".join(exs))
        except Exception as e:
            logging.warning(f"HatefulMemes: failed to log sample items: {e}")

    def _read_jsonl(self, filepath, lab_to_name):
        items = []
        total = 0
        added = 0
        missing = 0
        bad = 0
        per_label = {0: 0, 1: 0}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                total += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                # fields: id, img, label, text (ignored)
                img_rel = obj.get('img', '')
                label = obj.get('label', None)
                if img_rel is None or label is None:
                    bad += 1
                    continue
                if label not in (0, 1):
                    raise ValueError(f"HatefulMemes: unexpected label {label} in {filepath}")
                impath = os.path.join(self.dataset_dir, img_rel.replace('/', os.sep))
                if not os.path.isfile(impath):
                    missing += 1
                    if missing <= 5:
                        logging.warning(f"HatefulMemes: missing file referenced in {os.path.basename(filepath)}: {impath}")
                    continue
                cname = lab_to_name[label]
                items.append(Datum(impath=impath, label=label, classname=cname))
                per_label[label] += 1
                added += 1
        # logging.info(
        #     # "HatefulMemes: read %s -> total_lines=%d, added=%d, missing_files=%d, bad_lines=%d, label_counts=%s",
        #     os.path.basename(filepath), total, added, missing, bad, per_label
        # )
        # Log a couple of examples
        for i in range(min(2, len(items))):
            it = items[i]
            # logging.info("HatefulMemes: example[%d] label=%d cname=%s path=%s", i, it.label, it.classname, it.impath)
        return items
