from __future__ import annotations

import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TEST_DIR = SCRIPT_DIR / "Datasets" / "Onsite_Test"
DEFAULT_TRAIN_CSV = SCRIPT_DIR / "Datasets" / "Onsite_Train" / "train_list.csv"


def safe_backbone_name(backbone_name: str) -> str:
    return backbone_name.replace(" ", "_").replace("/", "_")


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    kind: str
    embed_dim: int
    default_encoder_path: Path
    default_classifier_path: Path
    student_class_name: str | None = None


BACKBONE_SPECS = {
    "LISTEN": BackboneSpec(
        name="LISTEN",
        kind="listen",
        embed_dim=64,
        default_encoder_path=SCRIPT_DIR / "Models" / "LISTEN.pth",
        default_classifier_path=SCRIPT_DIR / "Models" / "LISTEN-classifier.pth",
    ),
    "MobileNetV4-Small": BackboneSpec(
        name="MobileNetV4-Small",
        kind="student",
        embed_dim=192,
        default_encoder_path=SCRIPT_DIR / "Models" / "MobileNetV4-Small.pth",
        default_classifier_path=SCRIPT_DIR / "Models" / "MobileNetV4-Small-classifier.pth",
        student_class_name="MobileNetV4Small_Student",
    ),
    "MobileViT-XXS": BackboneSpec(
        name="MobileViT-XXS",
        kind="student",
        embed_dim=192,
        default_encoder_path=SCRIPT_DIR / "Models" / "MobileViT-XXS.pth",
        default_classifier_path=SCRIPT_DIR / "Models" / "MobileViT-XXS-classifier.pth",
        student_class_name="MobileViT_XXS_Student",
    ),
    "BC-ResNet-3": BackboneSpec(
        name="BC-ResNet-3",
        kind="student",
        embed_dim=192,
        default_encoder_path=SCRIPT_DIR / "Models" / "BC-ResNet-3.pth",
        default_classifier_path=SCRIPT_DIR / "Models" / "BC-ResNet-3-classifier.pth",
        student_class_name="BCResNet3_Student",
    ),
}


def get_default_device() -> str:
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class DistillConfig:
    sr: int = 48_000
    sample_len: int = 48_000
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 376
    n_mels: int = 128
    top_db: int = 80

    patch_size: Tuple[int, int] = (16, 16)
    nhead: int = 4
    distill_dim: int = 64
    depth: int = 2
    multiplier: int = 1

    batch_size: int = 1
    num_workers: int = 0
    seed: int = 42
    device: str = get_default_device()


class WavInferenceDataset(Dataset):
    def __init__(self, test_dir: str | os.PathLike, cfg: DistillConfig):
        super().__init__()
        self.test_dir = Path(test_dir)
        self.cfg = cfg
        self.wav_paths = sorted(self.test_dir.rglob("*.wav"))
        if not self.wav_paths:
            raise FileNotFoundError(f"No wav files found in: {self.test_dir}")

        self.mel = torchaudio.transforms.MelSpectrogram(
            cfg.sr, cfg.n_fft, cfg.win_length, cfg.hop_length,
            cfg.n_mels, power=2.
        )
        self.db = torchaudio.transforms.AmplitudeToDB("power", top_db=cfg.top_db)

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if sr != self.cfg.sr:
            raise RuntimeError(f"SR mismatch for {wav_path}: expected {self.cfg.sr}, got {sr}")

        if audio.ndim > 1:
            audio = audio.mean(1)

        original_num_samples = len(audio)
        audio = np.pad(audio, (0, max(0, self.cfg.sample_len - len(audio))))[:self.cfg.sample_len]

        x = torch.tensor(audio).unsqueeze(0)
        rms = torch.sqrt((x ** 2).mean())
        x = (x / (rms + 1e-6) - x.mean()) / (x.std() + 1e-6)

        spec = self.db(self.mel(x)).squeeze(0)[:, :128]
        spec = (spec + self.cfg.top_db) / (2 * self.cfg.top_db)

        return {
            "spec": spec.unsqueeze(0),
            "filepath": str(wav_path),
            "relative_path": str(wav_path.relative_to(self.test_dir)),
            "source_folder": wav_path.parent.name,
            "filename": wav_path.name,
            "sample_rate": sr,
            "num_samples": original_num_samples,
        }


class DistillModel(nn.Module):
    def __init__(self, cfg: DistillConfig, dim_feedforward: int | None = None):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = cfg.distill_dim

        self.conv1 = nn.Conv2d(1, cfg.distill_dim, 16, 16)
        self.bn1 = nn.BatchNorm2d(cfg.distill_dim)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                cfg.distill_dim,
                cfg.nhead,
                dim_feedforward,
                0.0,
                batch_first=True,
                activation="relu",
                norm_first=True,
            )
            for _ in range(cfg.depth)
        ])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        for blk in self.blocks:
            x = blk(x)

        return x.mean(dim=1)


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MNv4ConvBN(nn.Module):
    def __init__(self, in_c, out_c, k, s=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, (k - 1) // 2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UniversalInvertedBottleneck(nn.Module):
    def __init__(self, in_c, out_c, expand_ratio, start_dw, middle_dw, stride):
        super().__init__()
        self.start_dw = start_dw
        self.middle_dw = middle_dw

        if start_dw:
            self.start_conv = nn.Sequential(
                nn.Conv2d(
                    in_c,
                    in_c,
                    start_dw,
                    stride if not middle_dw else 1,
                    (start_dw - 1) // 2,
                    groups=in_c,
                    bias=False,
                ),
                nn.BatchNorm2d(in_c),
            )

        exp_c = make_divisible(int(in_c * expand_ratio), 8)
        self.exp_conv = nn.Sequential(
            nn.Conv2d(in_c, exp_c, 1, bias=False),
            nn.BatchNorm2d(exp_c),
            nn.ReLU(inplace=True),
        )

        if middle_dw:
            self.mid_conv = nn.Sequential(
                nn.Conv2d(exp_c, exp_c, middle_dw, stride, (middle_dw - 1) // 2, groups=exp_c, bias=False),
                nn.BatchNorm2d(exp_c),
                nn.ReLU(inplace=True),
            )

        self.proj_conv = nn.Sequential(
            nn.Conv2d(exp_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.identity = stride == 1 and in_c == out_c

    def forward(self, x):
        shortcut = x
        if self.start_dw:
            x = self.start_conv(x)
        x = self.exp_conv(x)
        if self.middle_dw:
            x = self.mid_conv(x)
        x = self.proj_conv(x)
        return x + shortcut if self.identity else x


class MobileNetV4Small_Student(nn.Module):
    def __init__(self, output_dim=192):
        super().__init__()
        specs = [
            ("conv", 3, 2, 32), ("conv", 3, 2, 32), ("conv", 1, 1, 32),
            ("conv", 3, 2, 96), ("conv", 1, 1, 64),
            ("uib", 5, 5, 2, 96, 3.0), ("uib", 0, 3, 1, 96, 2.0), ("uib", 0, 3, 1, 96, 2.0),
            ("uib", 0, 3, 1, 96, 2.0), ("uib", 0, 3, 1, 96, 2.0),
            ("uib", 3, 0, 1, 96, 4.0), ("uib", 3, 3, 2, 128, 6.0), ("uib", 5, 5, 1, 128, 4.0),
            ("uib", 0, 5, 1, 128, 4.0), ("uib", 0, 5, 1, 128, 3.0),
            ("uib", 0, 3, 1, 128, 4.0), ("uib", 0, 3, 1, 128, 4.0),
            ("conv", 1, 1, 960),
        ]

        layers = []
        channels = 1
        for block_type, *args in specs:
            if block_type == "conv":
                kernel, stride, filters = args
                layers.append(MNv4ConvBN(channels, filters, kernel, stride))
                channels = filters
            elif block_type == "uib":
                start_kernel, middle_kernel, stride, filters, expand = args
                layers.append(UniversalInvertedBottleneck(channels, filters, expand, start_kernel, middle_kernel, stride))
                channels = filters

        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(960, output_dim),
        )

    def forward(self, x):
        return self.head(self.features(x))


class SubSpectralNorm(nn.Module):
    def __init__(self, c, sub=5):
        super().__init__()
        self.sub = sub
        self.bn = nn.BatchNorm2d(c * sub)

    def forward(self, x):
        batch, channels, freq, time_steps = x.shape
        original_freq = freq

        if freq % self.sub != 0:
            pad = self.sub - (freq % self.sub)
            x = F.pad(x, (0, 0, 0, pad))
            freq = x.shape[2]

        x = x.view(batch, channels * self.sub, freq // self.sub, time_steps)
        x = self.bn(x)
        x = x.view(batch, channels, freq, time_steps)
        return x[:, :, :original_freq, :]


class BCResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, dilation=1, use_ssn=True):
        super().__init__()
        self.use_trans = (in_c != out_c) or (stride != 1)

        if self.use_trans:
            self.proj = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(True),
            )
            eff_in = out_c
        else:
            self.proj = nn.Identity()
            eff_in = in_c

        self.f2 = nn.Sequential(
            nn.Conv2d(eff_in, out_c, (3, 1), (stride, 1), (1, 0), groups=eff_in, bias=False),
            SubSpectralNorm(out_c) if use_ssn else nn.BatchNorm2d(out_c),
        )
        self.f1 = nn.Sequential(
            nn.Conv1d(out_c, out_c, 3, 1, dilation, dilation, groups=out_c, bias=False),
            nn.BatchNorm1d(out_c),
            nn.SiLU(),
            nn.Conv1d(out_c, out_c, 1, bias=False),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        if self.use_trans:
            x = self.proj(x)
        f2_out = self.f2(x)
        aux = f2_out if self.use_trans else (x + f2_out)
        time_out = self.f1(f2_out.mean(2))
        return F.relu(aux + time_out.unsqueeze(2))


class BCResNet3_Student(nn.Module):
    def __init__(self, output_dim=192):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, (5, 5), (2, 1), (2, 2), bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(True),
        )
        self.stages = nn.ModuleList()
        curr, cfgs = 24, [(24, 2, 1), (36, 2, 2), (48, 4, 2), (60, 4, 2)]
        dils = [[1, 1], [1, 1], [1, 2, 4, 8], [1, 2, 4, 8]]

        for idx, (channels, n_blocks, stride) in enumerate(cfgs):
            blocks = []
            for block_idx in range(n_blocks):
                block_stride = stride if block_idx == 0 else 1
                blocks.append(BCResBlock(curr if block_idx == 0 else channels, channels, block_stride, dils[idx][block_idx]))
            self.stages.append(nn.Sequential(*blocks))
            curr = channels

        self.head = nn.Sequential(
            nn.Conv2d(60, 60, 5, 1, 0, groups=60, bias=False),
            nn.Conv2d(60, 128, 1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)


class CustomTransformer(nn.Module):
    def __init__(self, dim, head, ff, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, head, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(ff, dim),
            nn.Dropout(drop),
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        return x + self.ff(self.ln2(x))


class MobileViTBlock(nn.Module):
    def __init__(self, in_c, dim, ff, n_layers, heads=2, patch=2):
        super().__init__()
        self.p = patch
        self.local = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.SiLU(),
            nn.Conv2d(in_c, dim, 1, bias=False),
        )
        self.global_trans = nn.Sequential(*[CustomTransformer(dim, heads, ff) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(dim)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, in_c, 1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_c * 2, in_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.SiLU(),
        )

    def forward(self, x):
        residual = x
        local = self.local(x)
        batch, channels, height, width = local.shape
        patch_h, patch_w = self.p, self.p
        n_h, n_w = height // patch_h, width // patch_w

        global_feat = local.view(batch, channels, n_h, patch_h, n_w, patch_w)
        global_feat = global_feat.permute(0, 2, 4, 3, 5, 1).reshape(-1, patch_h * patch_w, channels)
        global_feat = self.global_trans(global_feat)
        global_feat = self.ln(global_feat)
        global_feat = global_feat.view(batch, n_h, n_w, patch_h, patch_w, channels)
        global_feat = global_feat.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, height, width)
        return self.fuse(torch.cat([residual, self.proj(global_feat)], 1))


class MV2(nn.Module):
    def __init__(self, in_c, out_c, s, exp):
        super().__init__()
        hidden = int(in_c * exp)
        self.use_res = (s == 1 and in_c == out_c)
        layers = []
        if exp != 1:
            layers.extend([
                nn.Conv2d(in_c, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(),
            ])
        layers.extend([
            nn.Conv2d(hidden, hidden, 3, s, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class MobileViT_XXS_Student(nn.Module):
    def __init__(self, output_dim=192):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.SiLU())
        self.s1 = MV2(16, 16, 1, 2)
        self.s2 = nn.Sequential(MV2(16, 24, 2, 2), MV2(24, 24, 1, 2))
        self.s3 = nn.Sequential(MV2(24, 48, 2, 2), MobileViTBlock(48, 64, 128, 2))
        self.s4 = nn.Sequential(MV2(48, 64, 2, 2), MobileViTBlock(64, 80, 160, 4))
        self.s5 = nn.Sequential(MV2(64, 80, 2, 2), MobileViTBlock(80, 96, 192, 3))
        self.head = nn.Sequential(
            nn.Conv2d(80, 320, 1, bias=False),
            nn.BatchNorm2d(320),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(320, output_dim),
        )

    def forward(self, x):
        return self.head(self.s5(self.s4(self.s3(self.s2(self.s1(self.stem(x)))))))


STUDENT_MODEL_CLASSES = {
    "MobileNetV4Small_Student": MobileNetV4Small_Student,
    "MobileViT_XXS_Student": MobileViT_XXS_Student,
    "BCResNet3_Student": BCResNet3_Student,
}


class AudioClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, num_classes)
        self.norm = nn.LayerNorm(256)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.norm(x)
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_class_names(train_csv: str | os.PathLike) -> list[str]:
    class_names = set()
    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.add(row["class_name"])
    if not class_names:
        raise RuntimeError(f"No class names found in: {train_csv}")
    return sorted(class_names)


def resolve_script_path(path: str | os.PathLike | None) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else SCRIPT_DIR / path


def choose_backbone(backbone_arg: str | None) -> str:
    if backbone_arg is not None:
        return backbone_arg

    names = list(BACKBONE_SPECS)
    print("\nSelect backbone")
    for idx, name in enumerate(names, start=1):
        print(f"{idx}. {name}")

    while True:
        try:
            choice = input("Backbone number [1]: ").strip()
        except EOFError:
            return names[0]

        if choice == "":
            return names[0]
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        if choice in BACKBONE_SPECS:
            return choice
        print("Invalid selection. Enter a number or backbone name.")


def load_models(
    cfg: DistillConfig,
    num_classes: int,
    backbone_name: str,
    encoder_path: str | os.PathLike | None,
    classifier_path: str | os.PathLike | None,
) -> tuple[DistillModel, AudioClassifier]:
    spec = BACKBONE_SPECS[backbone_name]
    encoder_path = Path(encoder_path) if encoder_path is not None else spec.default_encoder_path
    classifier_path = Path(classifier_path) if classifier_path is not None else spec.default_classifier_path

    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found for {backbone_name}: {encoder_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found for {backbone_name}: {classifier_path}")

    if spec.kind == "listen":
        dim_feedforward = cfg.distill_dim * cfg.multiplier
        encoder = DistillModel(cfg, dim_feedforward).to(cfg.device)
        encoder.load_state_dict(torch.load(encoder_path, map_location=cfg.device), strict=False)
    elif spec.kind == "student":
        model_class = STUDENT_MODEL_CLASSES[spec.student_class_name]
        encoder = model_class(output_dim=spec.embed_dim).to(cfg.device)
        encoder.load_state_dict(torch.load(encoder_path, map_location=cfg.device), strict=True)
    else:
        raise ValueError(f"Unsupported backbone kind: {spec.kind}")

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    classifier = AudioClassifier(input_dim=spec.embed_dim, num_classes=num_classes).to(cfg.device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=cfg.device))
    classifier.eval()

    return encoder, classifier


def tensor_items(value):
    if torch.is_tensor(value):
        return value.cpu().tolist()
    return value


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def build_inference_time_stats(rows: list[dict]) -> dict[str, float]:
    inference_times = np.array(
        [float(row["inference_time_sec"]) for row in rows],
        dtype=np.float64,
    )
    if inference_times.size == 0:
        return {
            "inference_time_mean_sec": 0.0,
            "inference_time_std_sec": 0.0,
            "inference_time_min_sec": 0.0,
            "inference_time_max_sec": 0.0,
        }

    return {
        "inference_time_mean_sec": float(inference_times.mean()),
        "inference_time_std_sec": float(inference_times.std()),
        "inference_time_min_sec": float(inference_times.min()),
        "inference_time_max_sec": float(inference_times.max()),
    }


def build_class_metrics(rows: list[dict], class_names: list[str]) -> list[dict]:
    total = len(rows)
    inference_time_stats = build_inference_time_stats(rows)
    metrics = []

    for class_name in class_names:
        tp = sum(row["source_folder"] == class_name and row["predicted_label"] == class_name for row in rows)
        fp = sum(row["source_folder"] != class_name and row["predicted_label"] == class_name for row in rows)
        fn = sum(row["source_folder"] == class_name and row["predicted_label"] != class_name for row in rows)
        tn = total - tp - fp - fn

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        accuracy = safe_divide(tp + tn, total)

        metrics.append({
            "class_name": class_name,
            "support": tp + fn,
            "predicted_support": tp + fp,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            **inference_time_stats,
        })

    return metrics


def write_class_metrics_csv(metrics: list[dict], metrics_csv: str | os.PathLike) -> Path:
    metrics_csv = Path(metrics_csv)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "class_name",
        "support",
        "predicted_support",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "inference_time_mean_sec",
        "inference_time_std_sec",
        "inference_time_min_sec",
        "inference_time_max_sec",
    ]
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    return metrics_csv


def run_inference(
    backbone_name: str,
    test_dir: str | os.PathLike = DEFAULT_TEST_DIR,
    train_csv: str | os.PathLike = DEFAULT_TRAIN_CSV,
    encoder_path: str | os.PathLike | None = None,
    classifier_path: str | os.PathLike | None = None,
    output_csv: str | os.PathLike | None = None,
    metrics_csv: str | os.PathLike | None = None,
) -> Path:
    cfg = DistillConfig()
    set_seed(cfg.seed)

    test_dir = Path(test_dir)
    train_csv = Path(train_csv)
    encoder_path = Path(encoder_path) if encoder_path is not None else None
    classifier_path = Path(classifier_path) if classifier_path is not None else None

    class_names = load_class_names(train_csv)
    idx_to_class = {idx: name for idx, name in enumerate(class_names)}
    class_to_idx = {name: idx for idx, name in idx_to_class.items()}

    dataset = WavInferenceDataset(test_dir, cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device == "cuda",
        drop_last=False,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if output_csv is None:
        output_csv = SCRIPT_DIR / "Logs" / f"{safe_backbone_name(backbone_name)}_inference_log_{timestamp}_{len(dataset)}.csv"
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if metrics_csv is None:
        metrics_csv = output_csv.with_name(f"{output_csv.stem}_class_metrics.csv")

    encoder, classifier = load_models(
        cfg,
        len(class_names),
        backbone_name,
        encoder_path,
        classifier_path,
    )

    rows = []
    if cfg.device == "cuda":
        torch.cuda.synchronize()

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Inference"):
            spec = batch["spec"].to(cfg.device)

            if cfg.device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            features = encoder(spec)
            logits = classifier(features)
            probs = logits.softmax(dim=1)
            pred_probs, pred_indices = probs.max(dim=1)

            if cfg.device == "cuda":
                torch.cuda.synchronize()
            elapsed_per_sample = (time.perf_counter() - start_time) / spec.size(0)

            prob_values = probs.cpu().numpy()
            pred_indices = pred_indices.cpu().numpy()
            pred_probs = pred_probs.cpu().numpy()

            filepaths = tensor_items(batch["filepath"])
            relative_paths = tensor_items(batch["relative_path"])
            source_folders = tensor_items(batch["source_folder"])
            filenames = tensor_items(batch["filename"])
            sample_rates = tensor_items(batch["sample_rate"])
            num_samples = tensor_items(batch["num_samples"])

            for row_idx, pred_idx in enumerate(pred_indices):
                source_folder = source_folders[row_idx]
                true_idx = class_to_idx.get(source_folder)
                predicted_label = idx_to_class[int(pred_idx)]

                row = {
                    "backbone": backbone_name,
                    "filepath": filepaths[row_idx],
                    "relative_path": relative_paths[row_idx],
                    "source_folder": source_folder,
                    "filename": filenames[row_idx],
                    "sample_rate": sample_rates[row_idx],
                    "num_samples": num_samples[row_idx],
                    "predicted_index": int(pred_idx),
                    "predicted_label": predicted_label,
                    "predicted_probability": float(pred_probs[row_idx]),
                    "true_index": "" if true_idx is None else true_idx,
                    "is_correct": "" if true_idx is None else predicted_label == source_folder,
                    "inference_time_sec": elapsed_per_sample,
                }
                for class_idx, class_name in idx_to_class.items():
                    row[f"prob_{class_name}"] = float(prob_values[row_idx, class_idx])
                rows.append(row)

    fieldnames = [
        "backbone",
        "filepath",
        "relative_path",
        "source_folder",
        "filename",
        "sample_rate",
        "num_samples",
        "predicted_index",
        "predicted_label",
        "predicted_probability",
        "true_index",
        "is_correct",
        "inference_time_sec",
        *[f"prob_{class_name}" for class_name in class_names],
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics = build_class_metrics(rows, class_names)
    metrics_csv = write_class_metrics_csv(metrics, metrics_csv)

    print(f"Device: {cfg.device}")
    print(f"Backbone: {backbone_name}")
    print(f"Classes: {class_names}")
    print(f"Processed wav files: {len(rows)}")
    print(f"Saved inference CSV: {output_csv}")
    print(f"Saved class metrics CSV: {metrics_csv}")
    return output_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Run LISTEN downstream inference on wav files.")
    parser.add_argument("--backbone", choices=list(BACKBONE_SPECS), default=None, help="Backbone to use. Omit to select from an interactive menu.")
    parser.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR), help="Folder containing wav files.")
    parser.add_argument("--train-csv", default=str(DEFAULT_TRAIN_CSV), help="Training CSV used to recover class order.")
    parser.add_argument("--encoder", default=None, help="Override encoder checkpoint path.")
    parser.add_argument("--classifier", default=None, help="Override trained classifier checkpoint path.")
    parser.add_argument("--output-csv", default=None, help="Output CSV path. Defaults to Logs/<backbone>_inference_log_<timestamp>_<N>.csv.")
    parser.add_argument("--metrics-csv", default=None, help="Class metrics CSV path. Defaults to <output_csv_stem>_class_metrics.csv.")
    return parser.parse_args()


def main():
    args = parse_args()
    backbone_name = choose_backbone(args.backbone)
    run_inference(
        backbone_name=backbone_name,
        test_dir=resolve_script_path(args.test_dir),
        train_csv=resolve_script_path(args.train_csv),
        encoder_path=resolve_script_path(args.encoder),
        classifier_path=resolve_script_path(args.classifier),
        output_csv=resolve_script_path(args.output_csv),
        metrics_csv=resolve_script_path(args.metrics_csv),
    )


if __name__ == "__main__":
    main()
