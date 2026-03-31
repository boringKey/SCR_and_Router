from __future__ import annotations

import argparse
import json
from typing import List, Dict, Any, Tuple


def format_matrix(mat: List[List[float]], decimals: int = 3) -> str:
    if not mat:
        return "[]"
    fmt = f"{{:.{decimals}f}}"
    lines = []
    for row in mat:
        row_str = ", ".join(fmt.format(v) for v in row)
        lines.append(f"[{row_str}]")
    return "[" + "\n" + ",\n".join(lines) + "\n]"


def load_lines(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] skip invalid JSON line: {e}")
    rows.sort(key=lambda r: int(r.get('task', 0)))
    return rows


def build_zs_matrix(rows: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[str]]:
    if not rows:
        return [], []

    def get_zs_dict(r: Dict[str, Any]) -> Dict[str, float] | None:
        for key in ('zs_mtil', 'zs', 'zs_pre'):
            zs = r.get(key)
            if isinstance(zs, dict) and len(zs) > 0:
                return zs
        return None

    rows_zs = []
    for r in rows:
        zs = get_zs_dict(r)
        if zs is not None:
            rows_zs.append((r, zs))
    if not rows_zs:
        return [], []

    names = list(rows_zs[0][1].keys())
    matrix: List[List[float]] = []
    for r, zs in rows_zs:
        row = [float(zs.get(name, 0.0)) for name in names]
        matrix.append(row)
    return matrix, names


def load_similarity_json(path: str) -> Tuple[List[List[float]], List[str]]:
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    sim_mat_raw = payload.get("similarity_matrix", [])
    sim_names_raw = payload.get("upstream_names", [])
    sim_mat = [[float(x) for x in row] for row in sim_mat_raw]
    sim_names = [str(x) for x in sim_names_raw]
    return sim_mat, sim_names


def compute_scores_with_matrix(
    zs_mat: List[List[float]],
    zs_names: List[str],
    sim_mat: List[List[float]],
    sim_names: List[str],
) -> List[float]:
    if not zs_mat or not sim_mat:
        return []

    zs_name_to_idx = {nm: i for i, nm in enumerate(zs_names)}
    sim_name_to_idx = {nm: i for i, nm in enumerate(sim_names)}
    common = [nm for nm in zs_names if nm in sim_name_to_idx]
    if len(common) < len(zs_names):
        missing = [nm for nm in zs_names if nm not in sim_name_to_idx]
        print(f"[WARN] Missing similarity columns for: {missing}")
    zs_cols = [zs_name_to_idx[nm] for nm in common]
    sim_cols = [sim_name_to_idx[nm] for nm in common]

    base = zs_mat[0]
    scores: List[float] = []
    T = min(len(sim_mat), max(0, len(zs_mat) - 1))
    if T < (len(zs_mat) - 1):
        print(f"[WARN] sim-matrix rows ({len(sim_mat)}) < tasks ({len(zs_mat) - 1}); truncating to {T}")
    for t in range(T):
        row = zs_mat[t + 1]
        total = 0.0
        sum_w = 0.0
        sims = [float(sim_mat[t][sim_cols[k]]) for k in range(len(common))]
        risks = [max(0.0, 1-s) for s in sims]
        norm = sum(risks) if risks else 0.0
        default_weight = 1.0 / max(1, len(common))
        for k in range(len(common)):
            i = zs_cols[k]
            diff = float(row[i]) - float(base[i])
            w = (risks[k] / norm) if norm > 0.0 else default_weight
            total += diff * w
            sum_w += w
        avg = (total / sum_w) if sum_w > 0.0 else 0.0
        scores.append(round(avg, 2))
    return scores


def compute_scores_grouped(
    zs_mat: List[List[float]],
    zs_names: List[str],
    sim_mat: List[List[float]],
    sim_names: List[str],
) -> Dict[str, Any]:
    if not zs_mat or not sim_mat:
        return {
            "weighted": {"low": [], "mid": [], "high": []},
        }

    zs_name_to_idx = {nm: i for i, nm in enumerate(zs_names)}
    sim_name_to_idx = {nm: i for i, nm in enumerate(sim_names)}
    common = [nm for nm in zs_names if nm in sim_name_to_idx]
    if len(common) < len(zs_names):
        missing = [nm for nm in zs_names if nm not in sim_name_to_idx]
        print(f"[WARN] Missing similarity columns for grouped score: {missing}")
    zs_cols = [zs_name_to_idx[nm] for nm in common]
    sim_cols = [sim_name_to_idx[nm] for nm in common]

    base = zs_mat[0]
    T = min(len(sim_mat), max(0, len(zs_mat) - 1))
    if T < (len(zs_mat) - 1):
        print(f"[WARN] sim-matrix rows ({len(sim_mat)}) < tasks ({len(zs_mat) - 1}); truncating to {T}")

    def calc_weighted(items):
        total = sum(item["diff"] * item["weight"] for item in items)
        sum_w = sum(item["weight"] for item in items)
        return round((total / sum_w) if sum_w > 0.0 else 0.0, 2)

    w_low, w_mid, w_high = [], [], []

    for t in range(T):
        row = zs_mat[t + 1]
        sims = [float(sim_mat[t][sim_cols[k]]) for k in range(len(common))]
        risks = [max(0.0, 1.0 - s) for s in sims]
        norm = sum(risks) if risks else 0.0
        default_weight = 1.0 / max(1, len(common))
        triplets = []
        for k in range(len(common)):
            i = zs_cols[k]
            sim_val = sims[k]
            diff = float(row[i]) - float(base[i])
            w = (risks[k] / norm) if norm > 0.0 else default_weight
            triplets.append({
                "sim": sim_val,
                "diff": diff,
                "weight": w,
                "index": i + 1,
            })
        triplets.sort(key=lambda x: x["sim"])
        n = len(triplets)
        q = n // 3
        low = triplets[:q]
        mid = triplets[q:2 * q]
        high = triplets[2 * q:]

        w_low.append(calc_weighted(low))
        w_mid.append(calc_weighted(mid))
        w_high.append(calc_weighted(high))

    return {
        "weighted": {"low": w_low, "mid": w_mid, "high": w_high},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute SCR metrics from metric.json and calculate_sim similarity JSON.",
    )
    parser.add_argument("--metric-path", type=str, default="metric.json", help="Path to the metric.json file")
    parser.add_argument("--sim-json", type=str, required=True, help="Path to the similarity JSON generated by calculate_sim.py")
    parser.add_argument("--zs-decimals", type=int, default=2, help="Decimals when printing ZS matrix")
    args = parser.parse_args()

    rows = load_lines(args.metric_path)
    zs_mat, zs_names = build_zs_matrix(rows)
    sim_mat, sim_names = load_similarity_json(args.sim_json)

    if not zs_mat:
        print("[ERROR] No zero-shot matrix could be extracted from metric.json.")
        return
    if not sim_mat:
        print("[ERROR] No similarity matrix could be extracted from the calculate_sim JSON.")
        return

    num_cols = len(zs_mat[0]) if zs_mat else 0
    if any(len(row) != num_cols for row in zs_mat):
        print("[ERROR] Extracted zero-shot matrix rows must all have the same length.")
        return

    sim_cols = len(sim_mat[0]) if sim_mat else 0
    if any(len(row) != sim_cols for row in sim_mat):
        print("[ERROR] similarity_matrix rows in the calculate_sim JSON must all have the same length.")
        return
    if sim_names and len(sim_names) != sim_cols:
        print("[ERROR] upstream_names length must match similarity_matrix columns.")
        return
    if not sim_names:
        sim_names = zs_names

    print("1. Zero-shot matrix (baseline + tasks):")
    print(format_matrix(zs_mat, decimals=args.zs_decimals))

    scores_weighted = compute_scores_with_matrix(zs_mat, zs_names, sim_mat, sim_names)
    avg_weighted = round(sum(scores_weighted) / len(scores_weighted), 2) if scores_weighted else 0.0
    print("\nSCR:")
    print(avg_weighted)

    grouped = compute_scores_grouped(zs_mat, zs_names, sim_mat, sim_names)

    def mean_or_zero(arr: list) -> float:
        return round(sum(arr) / len(arr), 2) if arr else 0.0

    weighted = grouped["weighted"]
    low_w_mean = mean_or_zero(weighted["low"])
    mid_w_mean = mean_or_zero(weighted["mid"])
    high_w_mean = mean_or_zero(weighted["high"])

    print("\nLow-Similarity SCR:", low_w_mean)
    print("Mid-Similarity SCR:", mid_w_mean)
    print("High-Similarity SCR:", high_w_mean)

if __name__ == "__main__":
    main()
