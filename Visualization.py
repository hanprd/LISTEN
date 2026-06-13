from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "Logs"
DEFAULT_OUTPUT_DIR = LOG_DIR / "Visualizations"

# Keep matplotlib from trying to write its cache under the user's home directory.
MPLCONFIGDIR = PROJECT_DIR / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
XDG_CACHE_HOME = PROJECT_DIR / ".cache"
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_ORDER = [
    "LISTEN",
    "MobileNetV4-Small",
    "MobileViT-XXS",
    "BC-ResNet-3",
]

MODE_ORDER = [
    "Mode1",
    "Mode2",
    "Mode3",
    "Mode4",
    "Mode5",
    "Mode6",
    "Mode7",
    "Mode8",
    "Off",
    "On",
]

METRICS_CSVS = {
    "BC-ResNet-3": LOG_DIR / "BC-ResNet-3_inference_log_20260613_131736_1038_class_metrics.csv",
    "LISTEN": LOG_DIR / "LISTEN_inference_log_20260613_131701_1038_class_metrics.csv",
    "MobileNetV4-Small": LOG_DIR / "MobileNetV4-Small_inference_log_20260613_131711_1038_class_metrics.csv",
    "MobileViT-XXS": LOG_DIR / "MobileViT-XXS_inference_log_20260613_131722_1038_class_metrics.csv",
}

INFERENCE_CSVS = {
    "BC-ResNet-3": LOG_DIR / "BC-ResNet-3_inference_log_20260613_131736_1038.csv",
    "LISTEN": LOG_DIR / "LISTEN_inference_log_20260613_131701_1038.csv",
    "MobileNetV4-Small": LOG_DIR / "MobileNetV4-Small_inference_log_20260613_131711_1038.csv",
    "MobileViT-XXS": LOG_DIR / "MobileViT-XXS_inference_log_20260613_131722_1038.csv",
}

MODEL_COLORS = {
    "LISTEN": "#2F6B9A",
    "MobileNetV4-Small": "#E07A3F",
    "MobileViT-XXS": "#4D9A57",
    "BC-ResNet-3": "#8B5FBF",
}

TIMESTAMP_PATTERN = re.compile(r"_inference_log_(\d{8})_(\d{6})_")


def require_columns(df: pd.DataFrame, columns: list[str], path: Path):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def ordered_modes(modes: list[str]) -> list[str]:
    known = [mode for mode in MODE_ORDER if mode in modes]
    extra = sorted(mode for mode in modes if mode not in MODE_ORDER)
    return known + extra


def log_sort_key(path: Path) -> tuple[str, float, str]:
    match = TIMESTAMP_PATTERN.search(path.name)
    timestamp = "".join(match.groups()) if match else ""
    return timestamp, path.stat().st_mtime, path.name


def find_latest_log(model_name: str, class_metrics: bool) -> Path:
    if class_metrics:
        candidates = list(LOG_DIR.glob(f"{model_name}_inference_log_*_class_metrics.csv"))
    else:
        candidates = [
            path
            for path in LOG_DIR.glob(f"{model_name}_inference_log_*.csv")
            if not path.name.endswith("_class_metrics.csv")
        ]

    if not candidates:
        log_type = "class metrics" if class_metrics else "inference"
        raise FileNotFoundError(f"No {log_type} log found for {model_name} in {LOG_DIR}")
    return max(candidates, key=log_sort_key)


def discover_latest_logs() -> tuple[dict[str, Path], dict[str, Path]]:
    metrics_csvs = {model: find_latest_log(model, class_metrics=True) for model in MODEL_ORDER}
    inference_csvs = {model: find_latest_log(model, class_metrics=False) for model in MODEL_ORDER}
    return metrics_csvs, inference_csvs


def load_f1_table(metrics_csvs: dict[str, Path]) -> pd.DataFrame:
    records = []
    for model_name, path in metrics_csvs.items():
        if not path.exists():
            raise FileNotFoundError(f"Metrics CSV not found: {path}")
        df = pd.read_csv(path)
        require_columns(df, ["class_name", "f1"], path)
        for _, row in df.iterrows():
            records.append({
                "model": model_name,
                "mode": str(row["class_name"]),
                "f1": float(row["f1"]),
            })

    if not records:
        raise RuntimeError("No F1 records loaded.")
    return pd.DataFrame(records)


def load_inference_times(inference_csvs: dict[str, Path]) -> dict[str, np.ndarray]:
    times_by_model = {}
    for model_name, path in inference_csvs.items():
        if not path.exists():
            raise FileNotFoundError(f"Inference CSV not found: {path}")
        df = pd.read_csv(path)
        require_columns(df, ["inference_time_sec"], path)
        times = pd.to_numeric(df["inference_time_sec"], errors="coerce").dropna().to_numpy(dtype=float)
        if times.size == 0:
            raise RuntimeError(f"No valid inference_time_sec values in: {path}")
        times_by_model[model_name] = times
    return times_by_model


def compute_time_ylim(times_by_model: dict[str, np.ndarray], std_factor: float) -> tuple[float, float]:
    retained: list[np.ndarray] = []
    for times in times_by_model.values():
        mean = float(np.mean(times))
        std = float(np.std(times))
        lower = max(0.0, mean - std_factor * std)
        upper = mean + std_factor * std
        filtered = times[(times >= lower) & (times <= upper)]
        if filtered.size > 0:
            retained.append(filtered)

    if not retained:
        combined = np.concatenate(list(times_by_model.values()))
    else:
        combined = np.concatenate(retained)

    y_min = 0.0
    y_max = float(np.max(combined))
    padding = max(y_max * 0.08, 1e-6)
    return y_min, y_max + padding


def plot_f1_by_mode(f1_table: pd.DataFrame, output_path: Path):
    modes = ordered_modes(f1_table["mode"].unique().tolist())
    models = [model for model in MODEL_ORDER if model in f1_table["model"].unique()]
    pivot = f1_table.pivot_table(index="mode", columns="model", values="f1", aggfunc="mean").reindex(modes)

    x = np.arange(len(modes))
    width = min(0.18, 0.78 / max(1, len(models)))

    fig, ax = plt.subplots(figsize=(14, 6.5), constrained_layout=True)
    for idx, model in enumerate(models):
        offsets = x + (idx - (len(models) - 1) / 2) * width
        values = pivot[model].to_numpy(dtype=float)
        ax.bar(
            offsets,
            values,
            width=width,
            label=model,
            color=MODEL_COLORS.get(model),
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_title("F1 Score by Mode and Backbone", fontsize=15, pad=14)
    ax.set_xlabel("Mode", fontsize=12)
    ax.set_ylabel("F1 score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylim(0.50, 1.05)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(title="Backbone", ncols=2, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_inference_time_boxplot(times_by_model: dict[str, np.ndarray], output_path: Path, outlier_std_factor: float):
    models = [model for model in MODEL_ORDER if model in times_by_model]
    data = [times_by_model[model] for model in models]

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    box = ax.boxplot(
        data,
        tick_labels=models,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.55,
        medianprops={"color": "#222222", "linewidth": 1.5},
        meanprops={"color": "#B00020", "linewidth": 1.4},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "#777777",
            "markeredgecolor": "none",
            "alpha": 0.35,
        },
    )

    for patch, model in zip(box["boxes"], models, strict=False):
        patch.set_facecolor(MODEL_COLORS.get(model, "#777777"))
        patch.set_alpha(0.72)
        patch.set_edgecolor("#444444")

    ax.set_title("Inference Time Distribution by Backbone", fontsize=15, pad=14)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Inference time per file (sec)", fontsize=12)
    ax.set_ylim(*compute_time_ylim(times_by_model, outlier_std_factor))
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.tick_params(axis="x", labelrotation=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize downstream F1 scores and inference-time distributions.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where plot images are saved.")
    parser.add_argument("--f1-output", default="f1_by_mode.png", help="F1 bar chart filename.")
    parser.add_argument("--time-output", default="inference_time_boxplot.png", help="Inference-time box plot filename.")
    parser.add_argument("--fixed-logs", action="store_true", help="Use the hard-coded CSV paths instead of discovering the latest logs.")
    parser.add_argument("--time-outlier-std-factor", type=float, default=3.0, help="Exclude values outside mean +/- this many stds when setting the time-plot y-axis.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir

    metrics_csvs, inference_csvs = (METRICS_CSVS, INFERENCE_CSVS) if args.fixed_logs else discover_latest_logs()
    print("Metrics logs:")
    for model_name, path in metrics_csvs.items():
        print(f"  {model_name}: {path}")
    print("Inference logs:")
    for model_name, path in inference_csvs.items():
        print(f"  {model_name}: {path}")

    f1_table = load_f1_table(metrics_csvs)
    times_by_model = load_inference_times(inference_csvs)

    f1_output = output_dir / args.f1_output
    time_output = output_dir / args.time_output
    plot_f1_by_mode(f1_table, f1_output)
    plot_inference_time_boxplot(times_by_model, time_output, args.time_outlier_std_factor)

    print(f"Saved F1 chart: {f1_output}")
    print(f"Saved inference-time box plot: {time_output}")


if __name__ == "__main__":
    main()
