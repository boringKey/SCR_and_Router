import random
from typing import List, Sequence

import torch
from torch.utils.data import Dataset


class ClassSubsetTriplet(Dataset):
    """
    Wrap a base dataset (returning (img, label)) to only keep samples whose label is in class_ids.
    Optionally remap labels to local [0..K-1] for training. Always returns (img, label, task_id).
    """
    def __init__(self, base_ds: Dataset, class_ids: Sequence[int], task_id: int, remap_local: bool = True):
        self.base = base_ds
        self.task_id = int(task_id)
        self.class_ids = list(class_ids)
        self.idx_map = []  # indices in base
        # Build label mapping
        if remap_local:
            self.label_map = {c: i for i, c in enumerate(self.class_ids)}
        else:
            self.label_map = {c: c for c in self.class_ids}
        # Precompute indices into base dataset
        # The base is expected to have attribute `data_source` with items carrying `.label`.
        data_source = getattr(self.base, 'data_source', None)
        if data_source is None:
            # Fallback: iterate once to collect labels (may be slower)
            for i in range(len(self.base)):
                _, lab = self.base[i]
                if int(lab) in self.label_map:
                    self.idx_map.append(i)
        else:
            keep = set(self.class_ids)
            for i, item in enumerate(data_source):
                if int(item.label) in keep:
                    self.idx_map.append(i)

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):
        base_idx = self.idx_map[idx]
        img, label = self.base[base_idx]
        label = int(label)
        label = self.label_map[label]
        return img, label, self.task_id


class SeenEvalDataset(Dataset):
    """
    Evaluation dataset over seen classes up to a certain task.
    - Maps original labels to indices within the union-of-seen class order.
    - Emits task_ids per-sample corresponding to the task assignment of the class.
    """
    def __init__(self, base_ds: Dataset, seen_order: Sequence[int], class_to_task: Sequence[int]):
        self.base = base_ds
        self.seen_order = list(seen_order)
        self.class_to_task = class_to_task
        # Build mapping from original label -> seen index
        self.seen_index = {c: i for i, c in enumerate(self.seen_order)}
        # Precompute indices to keep and their target labels / task_ids
        self.keep_indices: List[int] = []
        self.targets: List[int] = []
        self.task_ids: List[int] = []

        data_source = getattr(self.base, 'data_source', None)
        if data_source is None:
            for i in range(len(self.base)):
                _, lab = self.base[i]
                lab = int(lab)
                if lab in self.seen_index:
                    self.keep_indices.append(i)
                    self.targets.append(self.seen_index[lab])
                    self.task_ids.append(int(self.class_to_task[lab]))
        else:
            keep = set(self.seen_order)
            for i, item in enumerate(data_source):
                lab = int(item.label)
                if lab in keep:
                    self.keep_indices.append(i)
                    self.targets.append(self.seen_index[lab])
                    self.task_ids.append(int(self.class_to_task[lab]))

    def __len__(self):
        return len(self.keep_indices)

    def __getitem__(self, idx):
        base_idx = self.keep_indices[idx]
        img, _ = self.base[base_idx]
        tgt = self.targets[idx]
        tid = int(self.task_ids[idx])
        return img, tgt, tid


class MTILCILTrainScenario:
    def __init__(self, train_tasks: List[ClassSubsetTriplet]):
        self._tasks = train_tasks

    def __len__(self):
        return len(self._tasks)

    def __iter__(self):
        for _ in range(len(self._tasks)):
            yield None

    def __getitem__(self, s):
        if isinstance(s, slice):
            assert (s.stop - (0 if s.start is None else s.start)) == 1, "Train slice must select exactly one task"
            idx = s.start or 0
            return self._tasks[idx]
        raise TypeError("Train scenario expects slicing with a single-task slice, e.g., [t:t+1]")


class MTILCILEvalScenario:
    def __init__(self, eval_seen_datasets: List[SeenEvalDataset]):
        self._seen = eval_seen_datasets

    def __len__(self):
        return len(self._seen)

    def __iter__(self):
        for _ in range(len(self._seen)):
            yield None

    def __getitem__(self, s):
        # Expect slices like [:t+1] used by main.py
        if isinstance(s, slice):
            if s.start is None and isinstance(s.stop, int):
                t = s.stop - 1
                return self._seen[t]
        raise TypeError("Eval scenario expects slicing like [:t+1]")


def build_mtil_cil_scenarios(
    base_train: Dataset,
    base_test: Dataset,
    classnames: Sequence[str],
    cil_splits: int,
    seed: int = 32,
):
    """
    Returns:
      train_scn: MTILCILTrainScenario
      eval_scn: MTILCILEvalScenario
      class_order: List[int]
      class_ids_per_task: List[List[int]]
      class_to_task: List[int] map from original class id -> task id
    """
    n_cls = len(classnames)
    assert cil_splits > 0, "cil_splits must be > 0"

    # Deterministic class order (keep natural order). If you want random, uncomment:
    class_order = list(range(n_cls))
    # random.Random(seed).shuffle(class_order)

    q, r = divmod(n_cls, cil_splits)
    sizes = [q + 1] * r + [q] * (cil_splits - r)
    # Remove trailing zero-size tasks if any (when cil_splits > n_cls)
    sizes = [s for s in sizes if s > 0]

    class_ids_per_task: List[List[int]] = []
    start = 0
    for s in sizes:
        class_ids_per_task.append(class_order[start:start + s])
        start += s

    # Map each original class id to its task id
    class_to_task = [-1] * n_cls
    for tidx, cls_ids in enumerate(class_ids_per_task):
        for c in cls_ids:
            class_to_task[c] = tidx

    # Build per-task train datasets (labels remapped to local 0..K-1)
    train_tasks = [
        ClassSubsetTriplet(base_train, cls_ids, task_id=tidx, remap_local=True)
        for tidx, cls_ids in enumerate(class_ids_per_task)
    ]

    # Build eval datasets per seen level (labels remapped to union-of-seen order)
    eval_seen = []
    for t in range(len(class_ids_per_task)):
        seen = [c for k in range(t + 1) for c in class_ids_per_task[k]]
        eval_seen.append(SeenEvalDataset(base_test, seen, class_to_task))

    train_scn = MTILCILTrainScenario(train_tasks)
    eval_scn = MTILCILEvalScenario(eval_seen)
    return train_scn, eval_scn, class_order, class_ids_per_task, class_to_task
