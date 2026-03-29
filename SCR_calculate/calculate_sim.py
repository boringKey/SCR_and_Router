import argparse
import os
from typing import List, Dict, Tuple

import numpy as np
import torch

# Project-local imports
from mtil_datasets import get_dataset as get_mtil_dataset
from continual_clip.mtil_cil import build_mtil_cil_scenarios
from continual_clip.clip_original import load as load_orig_clip, tokenize as tokenize_orig


DEFAULT_NAMES_ORDER1 = [
    "Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT",
    "OxfordFlowers", "Food101", "MNIST", "OxfordPets", "StanfordCars",
    "SUN397", "Country211", "SST2", "HatefulMemes", "GTSRB",
    "RESISC45", "FER2013", "UCF101", "CIFAR10", "STL10",
    "VOC2007", "ImageNetR", "KittiDistance", "PCam", "CLEVRCount",
]


def encode_texts(model, device, sentences: List[str], batch_size: int = 256) -> torch.Tensor:
    feats_all = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            chunk = sentences[i:i+batch_size]
            tokens = tokenize_orig(chunk).to(device)
            feats = model.encode_text(tokens)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
            feats_all.append(feats)
    if not feats_all:
        return torch.zeros((0, model.text_projection.shape[1]), device=device)
    return torch.cat(feats_all, dim=0)


def build_text_class_prototypes(model, device, classnames: List[str], templates, batch_size: int = 256) -> torch.Tensor:
    D = model.text_projection.shape[1]
    class_vecs: List[torch.Tensor] = []
    for cname in classnames:
        sents: List[str] = []
        if templates and len(templates) > 0:
            for t in templates:
                try:
                    sents.append(t(cname) if callable(t) else str(t).format(cname))
                except Exception:
                    continue
        else:
            sents = [f"a photo of a {cname}."]
        feats = encode_texts(model, device, sents, batch_size=batch_size)
        if feats.numel() == 0:
            v = torch.zeros(D, device=device)
        else:
            v = feats.mean(dim=0)
            v = v / (v.norm() + 1e-12)
        class_vecs.append(v)
    if not class_vecs:
        return torch.zeros((0, D), device=device)
    return torch.stack(class_vecs, dim=0)


def normalize_labels_to_list(lab) -> List[int]:
    if isinstance(lab, (int, np.integer)):
        return [int(lab)]
    import torch as _torch
    if _torch.is_tensor(lab):
        arr = lab.detach().cpu().numpy()
        if arr.ndim == 0:
            return [int(arr)]
        if arr.ndim == 1 and arr.size > 0 and set(np.unique(arr)).issubset({0,1}):
            return [int(x) for x in np.where(arr > 0.5)[0].tolist()]
        return [int(x) for x in arr.flatten().tolist()]
    if isinstance(lab, (list, tuple, np.ndarray)):
        arr = np.asarray(lab)
        if arr.ndim == 0:
            return [int(arr)]
        if arr.ndim == 1 and arr.size > 0 and set(np.unique(arr)).issubset({0,1}):
            return [int(x) for x in np.where(arr > 0.5)[0].tolist()]
        if arr.ndim == 1:
            return [int(x) for x in arr.tolist()]
        return [int(x) for x in np.where(arr.flatten() > 0.5)[0].tolist()]
    return []


def build_visual_class_prototypes(model, device, ds_wrapper, num_classes: int, max_per_class: int, batch_size: int, preprocess_eval) -> torch.Tensor:
    D = None
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {i: 0 for i in range(num_classes)}
    imgs_batch: List[torch.Tensor] = []
    labels_batch_multi: List[List[int]] = []
    with torch.no_grad():
        for i in range(len(ds_wrapper)):
            img, lab = ds_wrapper[i]
            lab_ids = [lid for lid in normalize_labels_to_list(lab) if 0 <= lid < num_classes and counts[lid] < max_per_class]
            if not lab_ids:
                continue
            if isinstance(img, torch.Tensor):
                tensor_img = img
            else:
                tensor_img = preprocess_eval(img)
            imgs_batch.append(tensor_img.unsqueeze(0))
            labels_batch_multi.append(lab_ids)
            if len(imgs_batch) >= max(1, batch_size):
                batch = torch.cat(imgs_batch, dim=0).to(device)
                feats = model.encode_image(batch)
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
                if D is None:
                    D = int(feats.shape[1])
                for f, ls in zip(feats, labels_batch_multi):
                    for l in ls:
                        if counts[l] >= max_per_class:
                            continue
                        if l not in sums:
                            sums[l] = f.detach().clone()
                        else:
                            sums[l] = sums[l] + f.detach()
                        counts[l] += 1
                imgs_batch.clear()
                labels_batch_multi.clear()
            if all(counts[l] >= max_per_class for l in range(num_classes)):
                break
        if imgs_batch:
            batch = torch.cat(imgs_batch, dim=0).to(device)
            feats = model.encode_image(batch)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
            if D is None:
                D = int(feats.shape[1])
            for f, ls in zip(feats, labels_batch_multi):
                for l in ls:
                    if counts[l] >= max_per_class:
                        continue
                    if l not in sums:
                        sums[l] = f.detach().clone()
                    else:
                        sums[l] = sums[l] + f.detach()
                    counts[l] += 1
            imgs_batch.clear()
            labels_batch_multi.clear()
    class_vecs: List[torch.Tensor] = []
    for lid in range(num_classes):
        c = counts.get(lid, 0)
        if c <= 0:
            class_vecs.append(torch.zeros(int(D or 0), device=device))
        else:
            m = sums[lid] / float(c)
            m = m / (m.norm() + 1e-12)
            class_vecs.append(m)
    if not class_vecs:
        return torch.zeros((0, int(D or 0)), device=device)
    return torch.stack(class_vecs, dim=0)


def build_visual_class_prototypes_subset(model, device, ds_wrapper, label_ids: List[int], max_per_class: int, batch_size: int, preprocess_eval) -> torch.Tensor:
    D = None
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {lid: 0 for lid in label_ids}
    label_set = set(label_ids)
    imgs_batch: List[torch.Tensor] = []
    labels_batch_multi: List[List[int]] = []
    with torch.no_grad():
        for i in range(len(ds_wrapper)):
            img, lab = ds_wrapper[i]
            lab_all = [lid for lid in normalize_labels_to_list(lab) if lid in label_set and counts.get(lid, 0) < max_per_class]
            if not lab_all:
                continue
            tensor_img = img if isinstance(img, torch.Tensor) else preprocess_eval(img)
            imgs_batch.append(tensor_img.unsqueeze(0))
            labels_batch_multi.append(lab_all)
            if len(imgs_batch) >= max(1, batch_size):
                batch = torch.cat(imgs_batch, dim=0).to(device)
                feats = model.encode_image(batch)
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
                if D is None:
                    D = int(feats.shape[1])
                for f, ls in zip(feats, labels_batch_multi):
                    for l in ls:
                        if counts[l] >= max_per_class:
                            continue
                        if l not in sums:
                            sums[l] = f.detach().clone()
                        else:
                            sums[l] = sums[l] + f.detach()
                        counts[l] += 1
                imgs_batch.clear()
                labels_batch_multi.clear()
            if all(counts[lid] >= max_per_class for lid in label_ids):
                break
        if imgs_batch:
            batch = torch.cat(imgs_batch, dim=0).to(device)
            feats = model.encode_image(batch)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
            if D is None:
                D = int(feats.shape[1])
            for f, ls in zip(feats, labels_batch_multi):
                for l in ls:
                    if counts[l] >= max_per_class:
                        continue
                    if l not in sums:
                        sums[l] = f.detach().clone()
                    else:
                        sums[l] = sums[l] + f.detach()
                    counts[l] += 1
            imgs_batch.clear()
            labels_batch_multi.clear()
    class_vecs: List[torch.Tensor] = []
    for lid in label_ids:
        c = counts.get(lid, 0)
        if c <= 0:
            class_vecs.append(torch.zeros(int(D or 0), device=device))
        else:
            m = sums[lid] / float(c)
            m = m / (m.norm() + 1e-12)
            class_vecs.append(m)
    if not class_vecs:
        return torch.zeros((0, int(D or 0)), device=device)
    return torch.stack(class_vecs, dim=0)


def cosine_distance_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [K1, D], B: [K2, D]; assume rows L2-normalized
    S = (A @ B.t()).clamp(-1.0, 1.0)
    return 1.0 - S  # in [0, 2]


def directed_covering_distance(T_src: torch.Tensor, V_src: torch.Tensor,
                               T_tgt: torch.Tensor, V_tgt: torch.Tensor) -> float:
    """Compute directed Chamfer distance A->B = mean_i min_j d(a_i, b_j)."""
    if V_src.size(0) == 0 or V_tgt.size(0) == 0:
        return 0.0
    F_src = torch.cat([V_src, T_src], dim=1)
    F_tgt = torch.cat([V_tgt, T_tgt], dim=1)
    F_src = F_src / (F_src.norm(dim=-1, keepdim=True) + 1e-12)
    F_tgt = F_tgt / (F_tgt.norm(dim=-1, keepdim=True) + 1e-12)
    C = cosine_distance_matrix(F_src, F_tgt)
    d_row_min = C.min(dim=1).values
    return float(d_row_min.mean().item())


def tasks_to_upstream_similarity_matrix(vision_model, device,
                                        upstream_names: List[str], name_to_idx: Dict[str, int],
                                        dataset_list, classes_names_list, templates_list,
                                        downstream_name: str, cil_splits: int,
                                        preprocess_eval,
                                        max_images_per_class: int, vision_batch_size: int,
                                        dataset_root: str,
                                        beta: float,
                                        clip_text_model=None) -> Tuple[np.ndarray, List[str]]:
    # Build upstream prototypes
    T_up_list: List[torch.Tensor] = []
    V_up_list: List[torch.Tensor] = []
    for nm in upstream_names:
        i = name_to_idx[nm]
        classnames = classes_names_list[i]
        templates = templates_list[i]
        ds_wrapper = dataset_list[i]
        K = len(classnames)
        print(f"Building upstream prototypes for {nm}: {K} classes")
        T_k = build_text_class_prototypes(clip_text_model, device, classnames, templates, batch_size=256)
        V_k = build_visual_class_prototypes(vision_model, device, ds_wrapper, K, max_per_class=int(max_images_per_class), batch_size=int(vision_batch_size), preprocess_eval=preprocess_eval)
        T_up_list.append(T_k)
        V_up_list.append(V_k)

    ds_idx = name_to_idx[downstream_name]
    cfg_ds = type("Cfg", (), {})()
    cfg_ds.dataset = "MTIL"
    cfg_ds.dataset_root = dataset_root
    cfg_ds.MTIL_order_2 = False
    cfg_ds.train_one_dataset = ds_idx
    cfg_ds.seed = 32
    cfg_ds.use_validation = False
    train_list, train_classes_names, train_templates = get_mtil_dataset(cfg_ds, 'train', transforms=preprocess_eval)
    test_list, _, _ = get_mtil_dataset(cfg_ds, 'test', transforms=preprocess_eval)
    assert len(train_list) == 1 and len(test_list) == 1, "Expected single selected dataset for downstream"
    classnames_single = train_classes_names[0]
    templates_single = train_templates[0]
    _, _, _, class_ids_per_task, _ = build_mtil_cil_scenarios(train_list[0], test_list[0], classnames_single, cil_splits, seed=32)

    distances = np.zeros((cil_splits, len(upstream_names)), dtype=float)
    for t, cls_ids in enumerate(class_ids_per_task):
        cls_names_t = [classnames_single[c] for c in cls_ids]
        print(f"Building downstream task-{t} prototypes: {len(cls_names_t)} classes")
        T_t = build_text_class_prototypes(clip_text_model, device, cls_names_t, templates_single, batch_size=256)
        ds_down = train_list[0]
        V_t = build_visual_class_prototypes_subset(vision_model, device, ds_down, list(cls_ids), max_per_class=int(max_images_per_class), batch_size=int(vision_batch_size), preprocess_eval=preprocess_eval)
        for j, nm in enumerate(upstream_names):
            distances[t, j] = directed_covering_distance(T_t, V_t, T_up_list[j], V_up_list[j])

    similarity = np.exp(-float(beta) * distances)
    similarity = np.clip(similarity, 0.0, 1.0)
    return similarity, upstream_names


def main():
    parser = argparse.ArgumentParser(description="Dataset similarity based on CLIP text and visual prototypes")
    parser.add_argument("--dataset-root", type=str, default=os.environ.get("DATASET_ROOT", "data"))
    parser.add_argument("--model-name", type=str, default="ViT-B/16")
    parser.add_argument("--vision-batch-size", type=int, default=64)
    parser.add_argument("--max-images-per-class", type=int, default=64, help="Maximum number of samples to use for each class when building visual prototypes. For efficiency, Chamfer distance is not computed using all samples in a dataset; instead, each class is sampled with at most this many examples.")
    parser.add_argument("--beta", type=float, default=1.0, help="Similarity mapping exp(-beta * dist)")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--downstream-dataset", type=str, default="", help="Name of downstream dataset (e.g., CIFAR100, PCam, FGVCAircraft, DescribableTextures)")
    parser.add_argument("--cil-split", type=int, default=0, help="Number of CIL splits for downstream dataset; if >0, output [cil_split x 24] matrix")
    args = parser.parse_args()

    names = DEFAULT_NAMES_ORDER1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load CLIP once (for clip vision and/or clip text)
    clip_model, _, preprocess_eval_clip = load_orig_clip(args.model_name, device=device, jit=False)
    clip_model.eval()

    vision_model = clip_model
    preprocess_eval = preprocess_eval_clip
    clip_text_model = clip_model

    # Load MTIL test datasets to build class/name lists and wrappers
    cfg = type("Cfg", (), {})()
    cfg.dataset = "MTIL"
    cfg.dataset_root = args.dataset_root
    cfg.MTIL_order_2 = False
    cfg.train_one_dataset = -1
    cfg.seed = 32
    cfg.use_validation = False

    dataset_list, classes_names_list, templates_list = get_mtil_dataset(cfg, 'test', transforms=preprocess_eval)

    n = min(len(names), len(classes_names_list))
    names = names[:n]
    classes_names_list = classes_names_list[:n]
    templates_list = templates_list[:n] if templates_list is not None else [None] * n

    # Downstream task-to-upstream similarity matrix mode
    if args.downstream_dataset and int(args.cil_split) > 0:
        alias_in = (args.downstream_dataset or '').strip()
        alias_map = {"FGVCAircraft": "Aircraft", "DescribableTextures": "DTD"}
        ds_internal = alias_map.get(alias_in, alias_in) if alias_in else None
        name_to_idx = {nm: idx for idx, nm in enumerate(names)}
        if ds_internal not in name_to_idx:
            raise ValueError(f"Downstream dataset '{args.downstream_dataset}' (mapped to '{ds_internal}') not found in MTIL names.")
        upstream_names = [nm for nm in names if nm != ds_internal]
        similarity_matrix, upstream_names = tasks_to_upstream_similarity_matrix(
            vision_model, device,
            upstream_names, name_to_idx,
            dataset_list, classes_names_list, templates_list,
            ds_internal, int(args.cil_split),
            preprocess_eval,
            int(args.max_images_per_class), int(args.vision_batch_size),
            args.dataset_root, float(args.beta), clip_text_model=clip_text_model
        )
        print("DEFAULT_SIM_MATRIX = [")
        fmt = "{:.3f}"
        for i in range(similarity_matrix.shape[0]):
            row_str = ", ".join(fmt.format(float(x)) for x in similarity_matrix[i])
            print(f"    [{row_str}],")
        print("]")
        names_py = ", ".join([f'"{n}"' for n in upstream_names])
        print(f"DEFAULT_SIM_UPSTREAM_NAMES = [{names_py}]")
        if args.output:
            import json
            out = {
                "downstream": args.downstream_dataset,
                "cil_split": int(args.cil_split),
                "upstream_names": upstream_names,
                "similarity_matrix": similarity_matrix.tolist(),
                "beta": float(args.beta),
                "max_images_per_class": int(args.max_images_per_class),
                "vision_encoder": "clip",
                "text_encoder": "clip",
            }
            base, ext = os.path.splitext(args.output)
            out2 = base + "_sim" + ext
            with open(out2, 'w', encoding='utf-8') as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"Saved similarity JSON to {out2}")
        return

if __name__ == "__main__":
    main()
