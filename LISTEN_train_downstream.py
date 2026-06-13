from __future__ import annotations
import argparse
import os, random
import logging
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv
from torch.amp import autocast, GradScaler
import torchaudio
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
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


def setup_training_logger(backbone_name: str, log_dir: str | os.PathLike = "Logs") -> tuple[logging.Logger, Path]:
    """Create a file-based logger with a timestamped log file.

    Args:
        log_dir: Directory for log files. Relative paths are resolved
                 against the script's parent directory.

    Returns:
        A (logger, log_path) tuple.
    """
    log_dir = Path(log_dir)
    if not log_dir.is_absolute():
        log_dir = Path(__file__).resolve().parent / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{safe_backbone_name(backbone_name)}_train_downstream_{timestamp}.log"

    logger = logging.getLogger("LISTEN_train_downstream")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)

    return logger, log_path

def log_and_print(logger: logging.Logger | None, message: str):
    """Print *message* to stdout and write it to the log file (if available)."""
    print(message)
    if logger is not None:
        logger.info(message)


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


def get_default_device() -> str:
    """Return the best available device: cuda > mps > cpu."""
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
    clips_per_folder: int | None = None

    lr: float = 5e-4
    wd: float = 1e-3
    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 200
    seed: int = 42
    device: str = get_default_device()

class AudioDataset(Dataset):
    """Dataset that reads audio clips from a CSV manifest, converts them to
    log-mel spectrograms, and returns (spectrogram, label_idx, class_name)."""

    def __init__(self, csv_path: str | os.PathLike, cfg: DistillConfig):
        super().__init__()
        self.cfg = cfg
        self.sample_length = cfg.sample_len
        self.index_list = []
        self.class_to_idx = {}
        self.csv_path = csv_path
        class_names = set()

        # Parse CSV: each row must contain filepath, start_sample, label, class_name
        with open(self.csv_path, newline = "") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.index_list.append(
                    (row["filepath"], int(row["start_sample"]),
                     int(row["label"]), row["class_name"])
                )
                class_names.add(row["class_name"])

        self.class_to_idx = {name: idx for idx, name in enumerate(sorted(class_names))}

        # Mel spectrogram and dB conversion transforms
        self.mel = torchaudio.transforms.MelSpectrogram(
            cfg.sr, cfg.n_fft, cfg.win_length, cfg.hop_length,
            cfg.n_mels, power = 2.
        )
        self.db = torchaudio.transforms.AmplitudeToDB("power", top_db = cfg.top_db)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        fp, start, label, folder = self.index_list[idx]
        audio, _sr = sf.read(fp, start = start, frames = self.cfg.sample_len,
                             dtype = 'float32', always_2d = False)
        if _sr != self.cfg.sr:
            raise RuntimeError("SR mismatch")

        # Zero-pad short clips and truncate to fixed length
        audio = np.pad(audio, (0, max(0, self.cfg.sample_len - len(audio))))[:self.cfg.sample_len]
        if audio.ndim > 1:
            audio = audio.mean(1)  # Downmix to mono

        # RMS-normalise then z-score standardise
        x = torch.tensor(audio).unsqueeze(0)
        rms = torch.sqrt((x ** 2).mean())
        x = (x / (rms + 1e-6) - x.mean()) / (x.std() + 1e-6)

        # Convert to log-mel spectrogram and scale to [0, 1]
        spec = self.db(self.mel(x)).squeeze(0)[:, :128]
        spec = (spec + self.cfg.top_db) / (2 * self.cfg.top_db)
        return spec.unsqueeze(0), self.class_to_idx[folder], folder
    
class DistillModel(nn.Module):
    """Lightweight vision-transformer encoder: patch-embed via Conv2d,
    followed by a stack of TransformerEncoderLayers with global average pooling."""

    def __init__(self, cfg: DistillConfig, dim_feedforward: int = None):
        super().__init__()
        self.cfg = cfg
        # Default feedforward dimension equals the embedding dimension
        if dim_feedforward is None:
            dim_feedforward = cfg.distill_dim

        # Patch embedding: 16x16 non-overlapping patches
        self.conv1 = nn.Conv2d(1, cfg.distill_dim, 16, 16)
        self.bn1 = nn.BatchNorm2d(cfg.distill_dim)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(cfg.distill_dim, cfg.nhead, dim_feedforward, 0.0, batch_first=True, activation='relu', norm_first=True)
            for _ in range(cfg.depth)
        ])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, N, C)

        for blk in self.blocks:
            x = blk(x)

        x = x.mean(dim=1)  # Global average pooling over tokens
        return x


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
    """Two-hidden-layer MLP head for downstream classification."""

    def __init__(self, input_dim: int, num_classes = 4):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, num_classes)
        self.norm = nn.LayerNorm(256)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.norm(x)
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def run_epoch(enc_model, mlp, loader, criterion, optimizer, cfg: DistillConfig, scaler: GradScaler | None, desc: str):
    """Run a single training epoch.

    The frozen *enc_model* produces embeddings which are fed to the
    trainable *mlp* head.  Mixed-precision is enabled only on CUDA.

    Returns:
        Average loss over all batches.
    """
    mlp.train()

    running_loss = 0.
    pbar = tqdm(loader, desc=desc)
    use_amp = cfg.device == "cuda" and scaler is not None

    for batch_idx, batch in enumerate(pbar):
        spec, label = batch[0], batch[1]
        spec = spec.to(cfg.device)
        label = label.to(cfg.device)
        optimizer.zero_grad(set_to_none=True)

        amp_context = autocast(device_type=cfg.device, dtype=torch.float16) if use_amp else nullcontext()
        with amp_context:
            # Encoder is frozen — no gradient needed
            with torch.no_grad():
                enc_feat = enc_model(spec)

            logits = mlp(enc_feat)
            loss = criterion(logits, label)

        # Backward and optimiser step (outside autocast)
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({
            'batch_loss': f"{loss.item():.8f}",
            'avg_loss': f"{running_loss / (batch_idx + 1):.8f}"
        })

    return running_loss / len(loader)

def set_seed(seed_value):
    """Set random seeds across all frameworks for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_encoder(
    backbone_name: str,
    cfg: DistillConfig,
    encoder_path: str | os.PathLike | None,
):
    spec = BACKBONE_SPECS[backbone_name]
    encoder_path = Path(encoder_path) if encoder_path is not None else spec.default_encoder_path

    if not encoder_path.exists():
        raise FileNotFoundError(
            f"Encoder checkpoint not found for {backbone_name}: {encoder_path}"
        )

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

    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    return encoder


def train_classifier(
    backbone_name: str,
    csv_path: str | os.PathLike,
    encoder_path: str | os.PathLike | None,
    classifier_path: str | os.PathLike | None,
    logger: logging.Logger | None = None,
):
    cfg = DistillConfig()
    spec = BACKBONE_SPECS[backbone_name]
    set_seed(cfg.seed or 0)
    classifier_path = Path(classifier_path) if classifier_path is not None else spec.default_classifier_path
    classifier_path.parent.mkdir(parents=True, exist_ok=True)

    if logger is not None:
        logger.info("Backbone: %s", backbone_name)
        logger.info("Config: %s", asdict(cfg))
        logger.info("Seed: %s", cfg.seed)
        logger.info("Training CSV: %s", csv_path)
        logger.info("Encoder checkpoint: %s", encoder_path or spec.default_encoder_path)
        logger.info("Classifier checkpoint: %s", classifier_path)

    ds = AudioDataset(csv_path, cfg)
    dl = DataLoader(ds, cfg.batch_size, shuffle = True,
                    num_workers = cfg.num_workers,
                    pin_memory = cfg.device == "cuda", drop_last = False)
    
    num_classes = len(ds.class_to_idx)
    log_and_print(logger, f"Number of classes: {num_classes}")
    if logger is not None:
        logger.info("Dataset size: %d", len(ds))
        logger.info("Number of batches per epoch: %d", len(dl))
        logger.info("Class mapping: %s", ds.class_to_idx)

    enc_model = load_encoder(backbone_name, cfg, encoder_path)
    
    mlp = AudioClassifier(input_dim=spec.embed_dim, num_classes=num_classes).to(cfg.device)

    optimizer = torch.optim.AdamW(mlp.parameters(), lr = cfg.lr, weight_decay = cfg.wd)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 5)

    scaler = None
    if cfg.device == "cuda":
        scaler = GradScaler()

    for epoch in range(cfg.epochs):
        avg_loss = run_epoch(
            enc_model,
            mlp,
            dl,
            criterion,
            optimizer,
            cfg,
            scaler,
            desc=f"Epoch {epoch+1}/{cfg.epochs} ({backbone_name}, embed:{spec.embed_dim})",
        )
        current_lr = optimizer.param_groups[0]['lr']
        log_and_print(logger, f"[Epoch {epoch + 1}/{cfg.epochs}] Loss: {avg_loss:.8f}, LR: {current_lr:.8f}")
        scheduler.step(avg_loss)
        torch.save(mlp.state_dict(), classifier_path)

    log_and_print(logger, f"Completed training and saved classifier checkpoint: {classifier_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a downstream classifier for a selected frozen backbone.")
    parser.add_argument("--backbone", choices=list(BACKBONE_SPECS), default=None, help="Backbone to use. Omit to select from an interactive menu.")
    parser.add_argument("--csv-path", default=str(DEFAULT_TRAIN_CSV), help="Training CSV path.")
    parser.add_argument("--encoder-path", default=None, help="Override encoder checkpoint path.")
    parser.add_argument("--classifier-path", default=None, help="Override classifier checkpoint output path.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    backbone_name = choose_backbone(args.backbone)
    encoder_path = resolve_script_path(args.encoder_path)
    classifier_path = resolve_script_path(args.classifier_path)
    csv_path = resolve_script_path(args.csv_path)

    logger, log_path = setup_training_logger(backbone_name)
    log_and_print(logger, f"Training log: {log_path}")
    log_and_print(logger, f"Training backbone: {backbone_name}")
    log_and_print(logger, f"Training CSV: {csv_path}")
    train_classifier(
        backbone_name=backbone_name,
        csv_path=csv_path,
        encoder_path=encoder_path,
        classifier_path=classifier_path,
        logger=logger,
    )
    log_and_print(logger, f"Finished training for {backbone_name}")
    log_and_print(logger, "=====================================")
