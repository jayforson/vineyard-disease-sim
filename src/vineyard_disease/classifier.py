"""Pretrained image classifier wrapper (Hugging Face transformers, CPU first).

The default checkpoint is a MobileNetV2 fine-tuned on a 38-class
PlantVillage-style label space. It is loaded with ``trust_remote_code=False``
at a pinned revision. Outputs are the full softmax over all model labels;
nothing is renormalized onto the four grape classes.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Protocol

from .errors import InputError, SourceUnavailableError, require

DEFAULT_MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
DEFAULT_MODEL_REVISION = "c1861579a670fb6232258805b801cd4137cb7176"
EXPECTED_NUM_LABELS = 38
MODEL_CAVEATS = (
    (
        "The default checkpoint's card reports training on the 'New Plant Diseases Dataset', an "
        "augmented re-release derived from PlantVillage; PlantVillage test-split metrics are therefore "
        "not an independent estimate and are likely optimistic."
    ),
    "Model card license is 'other'; weights are downloaded at runtime and never redistributed here.",
    "Softmax scores are uncalibrated and the model was trained on lab single-leaf photos.",
)
ML_HINT = "Install the ML extras (torch>=2.6, transformers>=4.46,<5, Pillow)."


class Classifier(Protocol):
    """Anything that maps RGB PIL images to a probability vector per image."""

    labels: Sequence[str]
    info: dict

    def predict_proba(self, images: Sequence) -> list[list[float]]: ...


def softmax(logits: Sequence[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


class HFImageClassifier:
    """AutoImageProcessor + AutoModelForImageClassification on CPU by default."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = DEFAULT_MODEL_REVISION,
        *,
        device: str = "cpu",
        local_files_only: bool = False,
        cache_dir: str | None = None,
        num_threads: int | None = None,
    ):
        torch = require("torch", "model inference", ML_HINT)
        transformers = require("transformers", "model inference", ML_HINT)
        require("PIL.Image", "model inference", ML_HINT)
        if device != "cpu" and not device.startswith("cuda"):
            raise InputError(f"unsupported device '{device}' (use 'cpu' or 'cuda[:N]')")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise InputError("CUDA requested but torch.cuda.is_available() is False; use --device cpu")
        if num_threads:
            torch.set_num_threads(num_threads)
        common = {
            "revision": revision,
            "trust_remote_code": False,
            "local_files_only": local_files_only,
            "cache_dir": cache_dir,
        }
        try:
            with warnings.catch_warnings():
                # The pinned checkpoint names a deprecated processor class; harmless.
                warnings.filterwarnings("ignore", message=".*FeatureExtractor is deprecated.*", category=FutureWarning)
                try:
                    self.processor = transformers.AutoImageProcessor.from_pretrained(model_id, use_fast=False, **common)
                except TypeError:  # older transformers without use_fast
                    self.processor = transformers.AutoImageProcessor.from_pretrained(model_id, **common)
                self.model = transformers.AutoModelForImageClassification.from_pretrained(model_id, **common)
        except OSError as exc:
            raise SourceUnavailableError(
                f"could not load model '{model_id}'@{revision}: {exc}. "
                "Check network access or pre-populate the Hugging Face cache (then use --local-files-only)."
            ) from exc
        self.model.eval()
        self.model.to(device)
        self._torch = torch
        self.device = device
        id2label = {int(k): v for k, v in (self.model.config.id2label or {}).items()}
        self.labels = [id2label[i] for i in sorted(id2label)]
        if sorted(id2label) != list(range(len(id2label))):
            raise InputError("model id2label indices are not contiguous from 0")
        self.info = {
            "model_id": model_id,
            "revision": revision,
            "device": device,
            "num_labels": len(self.labels),
            "trust_remote_code": False,
            "framework": f"transformers {transformers.__version__}, torch {torch.__version__}",
        }

    def predict_proba(self, images: Sequence) -> list[list[float]]:
        if not images:
            return []
        rgb = [img.convert("RGB") for img in images]
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.inference_mode():
            logits = self.model(**inputs).logits
            probs = self._torch.softmax(logits.float(), dim=-1).cpu().tolist()
        return probs


def load_classifier(args) -> HFImageClassifier:
    """Build the classifier from CLI arguments (lazy heavy imports)."""
    return HFImageClassifier(
        args.model_id,
        args.model_revision or None,
        device=args.device,
        local_files_only=args.local_files_only,
        num_threads=getattr(args, "num_threads", None),
    )
