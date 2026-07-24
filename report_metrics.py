from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

EXPECTED_FOLDS = {"smic": 16, "samm": 28, "casme2": 24}
EXPECTED_SAMPLES = {"smic": 164, "samm": 133, "casme2": 145, "all": 442}
PUBLIC_CONFIG_SCHEMA = {
    "auxiliary_pair_sampling": {"beta_left", "beta_right", "delta_min", "enabled", "max_resample"},
    "channel_swap": {
        "channel_swap_ramp_frac",
        "channel_swap_warmup_frac",
        "enabled",
        "lambda_channel_swap",
        "lambda_var",
        "mode",
        "partition",
        "partition_source",
        "sampling_source",
        "var_min_std",
    },
    "closed_loop": {"enabled"},
    "cross_view_mask": {"enabled"},
    "data": {"csv_path", "datasets", "extra_frames", "image_mode", "image_size", "roots", "subject_split_csv"},
    "diagnostics": {"t_flow_near0_thresh"},
    "finalize": {"primary_selector"},
    "gradient_classifier": {"enabled", "grad_scale", "hidden_dim", "mode", "num_layers"},
    "logging": {"exp_name", "save_dir"},
    "loso": {"enabled"},
    "loss": {
        "ce_ramp_frac",
        "lambda_cons",
        "lambda_cons_img",
        "lambda_photo",
        "lambda_rec",
        "lambda_sc",
        "lambda_smooth_D",
        "lambda_smooth_T",
        "lambda_smooth_flow_off",
        "lambda_smooth_flow_on",
        "lambda_warp_auxiliary_pair",
        "lambda_warp_endpoints",
        "photo_anneal_end_frac",
        "photo_anneal_start_frac",
        "photo_warmup_frac",
        "schedule",
    },
    "model": {
        "classifier_input_channels",
        "embed_channels",
        "include_r_in_classifier_input",
        "name",
        "num_classes",
        "use_reconstruction",
    },
    "model_selection": {"health_beta", "pareto_imp_min"},
    "motion": {
        "base_channels",
        "classifier_in_channels",
        "cls_rep",
        "compose_mode",
        "descriptor_detach_scale",
        "descriptor_norm_topk_ratio",
        "flow_downscale",
        "max_disp",
        "motion_mode",
        "stopgrad_cls_to_motion",
        "with_confidence",
    },
    "multiframe": {
        "enabled",
        "extra_rec_in_loss",
        "extra_rec_ramp_frac",
        "extra_rec_warmup_frac",
        "lambda_extra_rec",
        "lambda_rank",
        "rank_margin",
        "rank_ramp_frac",
        "rank_warmup_frac",
        "variant",
    },
    "photo": {
        "charbonnier_eps",
        "ssim_weight",
        "type",
        "use_grad",
        "w_dyn_dilate_kernel",
        "w_dyn_mode",
        "w_dyn_topk_ratio",
        "w_grad",
        "w_int",
        "w_ssim",
    },
    "reco": {"channel_swap", "cross_view_mask", "multiframe", "region", "trusted_transport"},
    "round_trip_composition": {"enabled"},
    "round_trip_consistency": {
        "cycle_ramp_frac",
        "cycle_warmup_frac",
        "enabled",
        "fb_conf_enabled",
        "fb_conf_min",
        "fb_conf_tau",
        "lambda_cycle",
        "occ_thresh",
        "variant",
    },
    "routing": {"feature_source", "gate_thr", "token_hw"},
    "sc": {"allow_dynamic_regions", "enabled", "mode", "region_partition", "region_source", "tau"},
    "train": {"batch_size", "epochs", "lr", "optimizer", "scheduler", "seed", "weight_decay"},
    "trust_propagation": {"enabled"},
}
PUBLIC_RECO_SCHEMA = {
    "channel_swap": {"enabled"},
    "cross_view_mask": {"enabled"},
    "multiframe": {"enabled"},
    "region": {"mode"},
    "trusted_transport": {
        "align_teacher",
        "align_unit",
        "conf_tok_min",
        "conf_tok_source",
        "enabled",
        "feat_dim",
        "lambda",
        "lambda_align",
        "temp",
        "token_hw",
        "use_conf_tok",
        "variant",
    },
}
PUBLIC_STATE_PREFIXES = {
    "adapter",
    "channel_swap_pred",
    "channel_swap_proj",
    "classifier",
    "gradient_classifier_head",
    "pair_flow",
    "reco",
}
REQUIRED_STATE_PREFIXES = {"gradient_classifier_head", "channel_swap_proj", "reco"}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "UF1": float(f1_score(y_true, y_pred, average="macro")),
        "UAR": float(recall_score(y_true, y_pred, average="macro")),
        "n_samples": int(y_true.shape[0]),
    }


def group_key_from_fold_name(name: str) -> str:
    low = name.lower()
    if low.startswith("smic__"):
        return "smic"
    if low.startswith("samm__"):
        return "samm"
    if low.startswith("casme2__"):
        return "casme2"
    raise RuntimeError(f"Unknown fold directory: {name}")


def collect_group_arrays(root: Path) -> dict[str, tuple[list[np.ndarray], list[np.ndarray]]]:
    groups: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        "all": ([], []),
        "smic": ([], []),
        "samm": ([], []),
        "casme2": ([], []),
    }

    for fold_dir in sorted(root.iterdir()):
        if not fold_dir.is_dir():
            continue
        y_true_path = fold_dir / "best_y_true.npy"
        y_pred_path = fold_dir / "best_y_pred.npy"
        if not (y_true_path.exists() and y_pred_path.exists()):
            continue

        y_true = np.load(y_true_path)
        y_pred = np.load(y_pred_path)
        groups["all"][0].append(y_true)
        groups["all"][1].append(y_pred)

        key = group_key_from_fold_name(fold_dir.name)
        groups[key][0].append(y_true)
        groups[key][1].append(y_pred)

    return groups


def summarize_metrics(root: Path) -> dict[str, dict[str, float | int]]:
    groups = collect_group_arrays(root)
    out: dict[str, dict[str, float | int]] = {}
    for key, (ys, ps) in groups.items():
        y_true = np.concatenate(ys, axis=0)
        y_pred = np.concatenate(ps, axis=0)
        out[key] = compute_metrics(y_true, y_pred)
    return out


def audit_predictions(root: Path) -> dict[str, object]:
    fold_counts: dict[str, int] = {"smic": 0, "samm": 0, "casme2": 0}
    sample_counts: dict[str, int] = {"smic": 0, "samm": 0, "casme2": 0}
    true_counts: dict[str, Counter[int]] = {
        "all": Counter(),
        "smic": Counter(),
        "samm": Counter(),
        "casme2": Counter(),
    }
    pred_counts: dict[str, Counter[int]] = {
        "all": Counter(),
        "smic": Counter(),
        "samm": Counter(),
        "casme2": Counter(),
    }

    for fold_dir in sorted(root.iterdir()):
        if not fold_dir.is_dir():
            continue
        key = group_key_from_fold_name(fold_dir.name)
        y_true_path = fold_dir / "best_y_true.npy"
        y_pred_path = fold_dir / "best_y_pred.npy"
        pt_path = fold_dir / "best.pt"
        if not y_true_path.exists() or not y_pred_path.exists() or not pt_path.exists():
            raise RuntimeError(f"Missing release artifacts in {fold_dir}")

        y_true = np.load(y_true_path)
        y_pred = np.load(y_pred_path)
        if y_true.shape != y_pred.shape:
            raise RuntimeError(f"Shape mismatch in {fold_dir}: {y_true.shape} vs {y_pred.shape}")
        if y_true.ndim != 1:
            raise RuntimeError(f"Expected 1-D labels in {fold_dir}, got shape {y_true.shape}")
        labels = set(map(int, np.concatenate([y_true, y_pred], axis=0).tolist()))
        if not labels.issubset({0, 1, 2}):
            raise RuntimeError(f"Unexpected class ids in {fold_dir}: {sorted(labels)}")

        fold_counts[key] += 1
        sample_counts[key] += int(y_true.shape[0])
        true_counts[key].update(map(int, y_true.tolist()))
        pred_counts[key].update(map(int, y_pred.tolist()))
        true_counts["all"].update(map(int, y_true.tolist()))
        pred_counts["all"].update(map(int, y_pred.tolist()))

    sample_counts["all"] = sum(sample_counts.values())
    if fold_counts != EXPECTED_FOLDS:
        raise RuntimeError(f"Unexpected fold counts: {fold_counts}, expected {EXPECTED_FOLDS}")
    for key, expected in EXPECTED_SAMPLES.items():
        if sample_counts[key] != expected:
            raise RuntimeError(f"Unexpected sample count for {key}: {sample_counts[key]}, expected {expected}")

    return {
        "fold_counts": fold_counts,
        "sample_counts": sample_counts,
        "true_counts": {k: dict(sorted(v.items())) for k, v in true_counts.items()},
        "pred_counts": {k: dict(sorted(v.items())) for k, v in pred_counts.items()},
    }


def audit_manifest(root: Path, manifest_path: Path) -> dict[str, object]:
    if not manifest_path.exists():
        raise RuntimeError(f"Missing release manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"fold_name", "Dataset", "Sub", "Which", "OnsetFrame", "Apex", "OffsetFrame", "Emo"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise RuntimeError(f"Manifest is missing columns: {sorted(missing)}")
    if len(rows) != EXPECTED_SAMPLES["all"]:
        raise RuntimeError(f"Unexpected manifest length: {len(rows)}")

    by_fold: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_fold.setdefault(row["fold_name"], []).append(row)

    fold_counts: dict[str, int] = {"smic": 0, "samm": 0, "casme2": 0}
    sample_counts: dict[str, int] = {"smic": 0, "samm": 0, "casme2": 0}
    for fold_name, fold_rows in sorted(by_fold.items()):
        key = group_key_from_fold_name(fold_name)
        y_true_path = root / fold_name / "best_y_true.npy"
        if not y_true_path.exists():
            raise RuntimeError(f"Manifest fold has no released labels: {fold_name}")
        manifest_y = np.asarray([int(r["Emo"]) for r in fold_rows], dtype=np.int64)
        released_y = np.load(y_true_path).astype(np.int64)
        if manifest_y.shape != released_y.shape or not np.array_equal(manifest_y, released_y):
            raise RuntimeError(f"Manifest label order does not match released y_true for {fold_name}")
        fold_counts[key] += 1
        sample_counts[key] += len(fold_rows)

    sample_counts["all"] = sum(sample_counts.values())
    if fold_counts != EXPECTED_FOLDS:
        raise RuntimeError(f"Unexpected manifest fold counts: {fold_counts}")
    for key, expected in EXPECTED_SAMPLES.items():
        if sample_counts[key] != expected:
            raise RuntimeError(f"Unexpected manifest sample count for {key}: {sample_counts[key]}")

    return {
        "path": str(manifest_path.resolve()),
        "rows": len(rows),
        "folds": len(by_fold),
        "fold_counts": fold_counts,
        "sample_counts": sample_counts,
        "label_order_matches_released_y_true": True,
    }


def audit_checkpoint_schema(pt_path: Path, ck: dict[str, object]) -> None:
    cfg = ck["cfg"]
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Checkpoint cfg is not a dict: {pt_path}")

    unexpected_top = sorted(set(cfg.keys()).difference(PUBLIC_CONFIG_SCHEMA.keys()))
    if unexpected_top:
        raise RuntimeError(f"Checkpoint contains non-public top-level config sections {unexpected_top}: {pt_path}")

    required_sections = {"routing", "trust_propagation", "gradient_classifier", "channel_swap", "cross_view_mask", "reco"}
    missing_sections = sorted(required_sections.difference(cfg.keys()))
    if missing_sections:
        raise RuntimeError(f"Checkpoint is missing public config sections {missing_sections}: {pt_path}")

    for section, allowed_keys in PUBLIC_CONFIG_SCHEMA.items():
        value = cfg.get(section, {}) or {}
        if not isinstance(value, dict):
            continue
        unexpected = sorted(set(value.keys()).difference(allowed_keys))
        if unexpected:
            raise RuntimeError(f"Checkpoint contains non-public keys in {section}: {unexpected} in {pt_path}")

    reco_cfg = cfg.get("reco", {}) or {}
    if not isinstance(reco_cfg, dict) or "trusted_transport" not in reco_cfg:
        raise RuntimeError(f"Checkpoint does not use the public ReCo schema: {pt_path}")
    for section, allowed_keys in PUBLIC_RECO_SCHEMA.items():
        value = reco_cfg.get(section, {}) or {}
        if not isinstance(value, dict):
            continue
        unexpected = sorted(set(value.keys()).difference(allowed_keys))
        if unexpected:
            raise RuntimeError(f"Checkpoint contains non-public keys in reco.{section}: {unexpected} in {pt_path}")

    state = ck["model_state"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint model_state is not a dict: {pt_path}")
    state_prefixes = {str(key).split(".", 1)[0] for key in state.keys()}
    unexpected_state_prefixes = sorted(state_prefixes.difference(PUBLIC_STATE_PREFIXES))
    if unexpected_state_prefixes:
        raise RuntimeError(f"Checkpoint contains non-public state_dict prefixes {unexpected_state_prefixes}: {pt_path}")
    missing_state_prefixes = sorted(REQUIRED_STATE_PREFIXES.difference(state_prefixes))
    if missing_state_prefixes:
        raise RuntimeError(f"Checkpoint is missing public state_dict prefixes {missing_state_prefixes}: {pt_path}")

    if not any(k.startswith("reco.transport.") for k in state.keys()):
        raise RuntimeError(f"Checkpoint does not expose the public transport module keys: {pt_path}")
    if not any(k.startswith("gradient_classifier_head.") for k in state.keys()):
        raise RuntimeError(f"Checkpoint does not expose the gradient classifier head keys: {pt_path}")
    if not any(k.startswith("channel_swap_proj.") for k in state.keys()):
        raise RuntimeError(f"Checkpoint does not expose the channel-swap projection keys: {pt_path}")


def check_load(root: Path) -> int:
    import torch

    from model.builder import build_model_from_cfg

    count = 0
    for pt_path in sorted(root.glob("*/best.pt")):
        ck = torch.load(pt_path, map_location="cpu")
        audit_checkpoint_schema(pt_path, ck)
        model = build_model_from_cfg(ck["cfg"])
        model.load_state_dict(ck["model_state"], strict=True)
        count += 1
    return count


def smoke_forward(root: Path) -> dict[str, object]:
    import torch

    from model.builder import build_model_from_cfg

    pt_path = next(iter(sorted(root.glob("*/best.pt"))))
    ck = torch.load(pt_path, map_location="cpu")
    cfg = ck["cfg"]
    image_size = int((cfg.get("data", {}) or {}).get("image_size", 224))
    n_pre = int(((cfg.get("data", {}) or {}).get("extra_frames", {}) or {}).get("num_pre", 2))
    n_post = int(((cfg.get("data", {}) or {}).get("extra_frames", {}) or {}).get("num_post", 2))

    model = build_model_from_cfg(cfg)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()

    with torch.no_grad():
        frame = torch.zeros(1, 3, image_size, image_size)
        extra_pre = torch.zeros(1, max(n_pre, 2), 3, image_size, image_size)
        extra_post = torch.zeros(1, max(n_post, 2), 3, image_size, image_size)
        dt_pre = torch.arange(max(n_pre, 2), 0, -1, dtype=torch.float32).view(1, -1)
        dt_post = torch.arange(1, max(n_post, 2) + 1, dtype=torch.float32).view(1, -1)
        out = model(
            frame,
            frame,
            frame,
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([2]),
            extra_pre=extra_pre,
            extra_post=extra_post,
            dt_pre=dt_pre,
            dt_post=dt_post,
        )
    return {
        "fold": pt_path.parent.name,
        "logits_shape": list(out.logits.shape),
        "aux_keys": len(out.aux),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "main_loso",
    )
    parser.add_argument(
        "--check-load",
        action="store_true",
        help="Additionally verify that all released checkpoints load with strict=True.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Validate released fold counts, sample counts, label ids, class distributions, and manifest alignment.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "release_manifest.csv",
        help="No-image release manifest aligned to the released fold order.",
    )
    parser.add_argument(
        "--smoke-forward",
        action="store_true",
        help="Run a no-data tensor forward pass through one released checkpoint.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output: dict[str, object] = {
        "root": str(root),
        "metrics": summarize_metrics(root),
    }
    if args.audit:
        output["audit"] = audit_predictions(root)
        output["manifest"] = audit_manifest(root, args.manifest)
    if args.check_load:
        output["strict_load_ok_folds"] = check_load(root)
    if args.smoke_forward:
        output["smoke_forward"] = smoke_forward(root)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
