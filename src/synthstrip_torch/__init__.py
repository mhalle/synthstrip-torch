"""synthstrip-torch: a lean, standalone PyTorch inference package for SynthStrip
brain extraction.

Model and conform adapted from FreeSurfer's ``mri_synthstrip`` (Hoopes et al.,
2022); this is a derivative, not the original. See the bundled FreeSurfer LICENSE.

Public API::

    from synthstrip_torch import predict_sdt, load_model
    sdt_zyx, sdt_sitk = predict_sdt("t1.nii.gz", device="cuda")

``predict_sdt`` returns the signed distance transform (negative inside brain);
threshold it (SynthStrip's default border is 1 mm) to get a mask. Callers that need
the mask on the *input* grid should resample the graded SDT first, then threshold.
"""
from .infer import conform, predict_sdt
from .model import ConvBlock, StripModel
from .weights import (
    MODEL_URL,
    WEIGHTS_VERSION,
    default_weights_path,
    fetch_weights,
    load_model,
)

__all__ = [
    "conform",
    "predict_sdt",
    "StripModel",
    "ConvBlock",
    "load_model",
    "fetch_weights",
    "default_weights_path",
    "WEIGHTS_VERSION",
    "MODEL_URL",
]

__version__ = "0.1.0"
