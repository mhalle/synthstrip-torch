"""Conform + inference: an input volume -> SynthStrip's signed distance transform (SDT).

Mirrors ``mri_synthstrip`` exactly: conform to 1 mm / LIA / crop-to-bbox / reshape
to a multiple of 64 in [192, 320] / normalize, then feed ``conformed.data`` to the
net. surfa's LIA array order IS the trained contract - a different axis order
under-segments the inferior brain (cerebellum), so this uses surfa's own conform,
not a reimplementation.

Geometry is handed back as a SimpleITK image via a surfa-save / SimpleITK-read, so
downstream (physical-space) resampling needs no manual surfa->SITK axis conversion:
surfa writes the correct geometry, SimpleITK reads it.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np


def _to_surfa(volume):
    """Coerce a path / SimpleITK image / surfa Volume to a surfa Volume (loaded
    eagerly, so any temp file can be released immediately)."""
    import surfa as sf

    if hasattr(volume, "conform"):                # already a surfa Volume
        return volume
    try:
        import SimpleITK as sitk

        if isinstance(volume, sitk.Image):
            with tempfile.TemporaryDirectory() as td:
                p = os.path.join(td, "in.nii.gz")
                sitk.WriteImage(volume, p)
                return sf.load_volume(p)
    except ImportError:
        pass
    return sf.load_volume(str(volume))


def conform(volume):
    """SynthStrip's trained-input conform. ``volume`` is a path, a SimpleITK image,
    or a surfa Volume; returns the conformed surfa Volume (1 mm, LIA, cropped,
    padded to a multiple of 64, intensity-normalized to [0, 1])."""
    image = _to_surfa(volume)
    conformed = image.conform(voxsize=1.0, dtype="float32", method="nearest",
                              orientation="LIA").crop_to_bbox()
    tgt = np.clip(np.ceil(np.array(conformed.shape[:3]) / 64).astype(int) * 64, 192, 320)
    conformed = conformed.reshape(tgt)
    conformed -= conformed.min()
    conformed = (conformed / conformed.percentile(99)).clip(0, 1)
    return conformed


def predict_sdt(volume, *, model=None, device: str = "cpu"):
    """Conform ``volume`` and run the net -> ``(sdt_zyx, sdt_sitk)``: the SDT as a
    ``(z, y, x)`` float32 array and the same field as a SimpleITK image carrying the
    conformed grid's geometry. ``model`` defaults to the cached SynthStrip net."""
    import SimpleITK as sitk
    import torch

    from .weights import load_model

    if model is None:
        model = load_model(device=device)
    conformed = conform(volume)
    with torch.no_grad():
        inp = torch.from_numpy(conformed.data[np.newaxis, np.newaxis]).to(device)
        sdt = model(inp).squeeze().float().cpu().numpy()
    with tempfile.TemporaryDirectory() as td:
        so = os.path.join(td, "sdt.nii.gz")
        conformed.new(sdt).save(so)               # surfa writes correct geometry
        sdt_sitk = sitk.ReadImage(so)             # SimpleITK reads it back (z,y,x + geom)
    return sitk.GetArrayFromImage(sdt_sitk).astype(np.float32), sdt_sitk
