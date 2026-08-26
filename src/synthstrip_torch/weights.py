"""Weights resolution + model loading for synthstrip-torch.

The published SynthStrip weights (``synthstrip.1.pt``, MGH) are fetched at first
use (not redistributed here) and cached. Resolution order for the path:
``path`` arg -> ``$SYNTHSTRIP_TORCH_MODEL`` -> a per-user cache
(``$XDG_CACHE_HOME/synthstrip-torch/synthstrip.1.pt``), downloaded if absent.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

WEIGHTS_VERSION = "v1"                        # synthstrip.1.pt (MGH, 2022-04-28)
MODEL_URL = ("https://ftp.nmr.mgh.harvard.edu/pub/dist/freesurfer/synthstrip/"
             "models/synthstrip.1.pt")


def default_weights_path() -> Path:
    """Where weights live absent an explicit path: ``$SYNTHSTRIP_TORCH_MODEL`` or
    the per-user cache dir."""
    env = os.environ.get("SYNTHSTRIP_TORCH_MODEL")
    if env:
        return Path(env)
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return cache / "synthstrip-torch" / "synthstrip.1.pt"


def fetch_weights(dest: str | os.PathLike | None = None) -> Path:
    """Return a local path to ``synthstrip.1.pt``, downloading it from MGH once if
    it is not already present. Idempotent; safe to call every run."""
    path = Path(dest) if dest else default_weights_path()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)   # fixed MGH URL, not user input
    tmp.replace(path)
    return path


_MODELS: dict = {}                            # (path, device) -> StripModel, cached


def load_model(path: str | os.PathLike | None = None, device: str = "cpu"):
    """Build :class:`~synthstrip_torch.model.StripModel`, load the SynthStrip
    weights (fetched if needed), and cache per (path, device). Weights load with
    ``weights_only=True`` (no pickle execution)."""
    import torch

    from .model import StripModel

    wpath = fetch_weights(path)
    key = (str(wpath), str(device))
    m = _MODELS.get(key)
    if m is not None:
        return m
    ckpt = torch.load(str(wpath), map_location="cpu", weights_only=True)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    m = StripModel()
    m.load_state_dict(sd)
    m.eval()
    m = m.to(device)
    _MODELS[key] = m
    return m
