# synthstrip-torch

A lean, standalone **PyTorch inference** package for [SynthStrip](https://synthstrip.io)
brain extraction (skull-stripping). One small 3D UNet, one call:

```python
from synthstrip_torch import predict_sdt

sdt_zyx, sdt_sitk = predict_sdt("t1.nii.gz", device="cuda")   # or "cpu" / "mps"
```

`predict_sdt` conforms the input the way the network was trained (1 mm, LIA,
crop-to-bbox, pad to a multiple of 64, intensity-normalize) and returns SynthStrip's
**signed distance transform** (negative inside brain) both as a `(z, y, x)` NumPy
array and as a SimpleITK image carrying the conformed grid's geometry. Threshold the
SDT (SynthStrip's default border is `1.0` mm) for a mask; to get a mask on the input
grid, resample the *graded* SDT to that grid first, then threshold — thresholding
after resampling gives a sub-voxel boundary.

Weights (`synthstrip.1.pt`, ~29 MB) are **fetched from MGH at first use** and cached
under `$XDG_CACHE_HOME/synthstrip-torch/` (override with `$SYNTHSTRIP_TORCH_MODEL` or
the `path=` argument). They are not redistributed in this repo.

## Install

```bash
pip install "synthstrip-torch @ git+https://github.com/mhalle/synthstrip-torch.git"
```

Depends only on `torch`, `numpy<2` (surfa's reorient requires it), `surfa`, and
`SimpleITK` — no FreeSurfer binaries, no monai/torchio.

## API

- `predict_sdt(volume, *, model=None, device="cpu") -> (sdt_zyx, sdt_sitk)` — the main entry point (`volume` is a path, a SimpleITK image, or a surfa Volume).
- `conform(volume) -> surfa.Volume` — the trained-input conform, if you want it alone.
- `load_model(path=None, device="cpu") -> StripModel` — build + load weights (fetched if absent), cached per `(path, device)`.
- `fetch_weights(dest=None) -> Path` — download `synthstrip.1.pt` once; idempotent.
- `StripModel`, `ConvBlock` — the network itself.

## Attribution and license

This is a **modified/derivative** redistribution of code from FreeSurfer's
[`mri_synthstrip`](https://github.com/freesurfer/freesurfer/tree/dev/mri_synthstrip)
(Hoopes et al., *SynthStrip: skull-stripping for any brain image*, NeuroImage 2022).
It is **not** the original SynthStrip and is not affiliated with or endorsed by MGH
or the FreeSurfer developers.

> All or portions of this licensed product have been obtained under license from
> The General Hospital Corporation.

Distributed under the **FreeSurfer Software License Agreement** — see
[`LICENSE`](LICENSE). This is not a permissive OSI license; do not relicense.

If you use SynthStrip, please cite:

> A. Hoopes, J. S. Mora, A. V. Dalca, B. Fischl, M. Hoffmann.
> *SynthStrip: skull-stripping for any brain image.* NeuroImage 260 (2022) 119474.
