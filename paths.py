"""Path resolution for the CNN zoo scripts.

Nothing that is large or third-party is committed to this repo. Those pieces
are resolved at runtime, relative to the repo root by default:

    <parent>/                     <- holds the SDK snapshot (sibling layout)
      RyzenAI-SW-1.7.1/           AMD SDK: official models + xclbins
      hf_models/                  third-party ONNX (MobileNetV2, RetinaFace)
      realsrgan256/               Real-ESRGAN ONNX
      models_src/                 RCAN ONNX
      cnn_zoo/                    <- THIS REPO
        fp32/  int8/              weights fetched by tools/fetch_models.py

You can also drop the SDK directory *inside* this repo instead of beside it;
both layouts are detected. Override entirely with ``AMD_NPU_HOME``:

    set AMD_NPU_HOME=D:\\sdks\\RyzenAI

The runtime install path is separate and defaults to the standard Windows
location; override with ``RYZEN_AI_INSTALL``.
"""
import os
from pathlib import Path

#: this repo (the cnn_zoo directory)
REPO = Path(__file__).resolve().parent


def _ext_root():
    """Where the AMD SDK snapshot and third-party models live.

    Resolution order:
      1. $AMD_NPU_HOME if set
      2. a sibling ``RyzenAI-SW-<ver>/`` next to this repo
      3. a ``RyzenAI-SW-<ver>/`` nested inside this repo
      4. sibling (best guess when nothing is downloaded yet)
    """
    env = os.environ.get("AMD_NPU_HOME")
    if env:
        return Path(env).expanduser().resolve()
    ver = os.environ.get("RYZEN_AI_SW_VER", "1.7.1")
    for cand in (REPO.parent, REPO):
        if (cand / ("RyzenAI-SW-%s" % ver)).is_dir():
            return cand.resolve()
    return REPO.parent


#: root holding RyzenAI-SW-<ver>/, hf_models/, realsrgan256/, models_src/
EXT = _ext_root()


#: RyzenAI runtime install
INSTALL = Path(os.environ.get("RYZEN_AI_INSTALL") or r"C:\Program Files\RyzenAI\1.7.1")

VOE = INSTALL / "voe-4.0-win_amd64"
XCLBIN_DIR = VOE / "xclbins" / "phoenix"

#: SDK snapshot version directory to use under EXT
SW_VER = os.environ.get("RYZEN_AI_SW_VER", "1.7.1")


def xclbin(tile="4x4"):
    """Full path to a Phoenix xclbin: xclbin('4x4') / xclbin('1x4')."""
    return str(XCLBIN_DIR / ("%s.xclbin" % tile))


def sdk(*rel):
    """Path inside the RyzenAI-SW snapshot, e.g. sdk('CNN-examples', 'models')."""
    return str(EXT / ("RyzenAI-SW-%s" % SW_VER) / Path(*rel))


def ext(*rel):
    """Path inside the third-party model root, e.g. ext('hf_models', 'x.onnx')."""
    return str(EXT / Path(*rel))


def have(p):
    """True if the path exists (models are optional, never hard-fail on them)."""
    return Path(p).exists()


def missing(*paths):
    """Subset of paths that do not exist, for startup warnings."""
    return [p for p in paths if not Path(p).exists()]
