"""Universal U-Net napari plugin."""

import sys
from pathlib import Path

# Make src/ importable (model, train, inference live there)
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from napari_gui import UNetWidget  # noqa: E402
from napari.viewer import Viewer


def make_unet_widget(viewer: Viewer):
    return UNetWidget(viewer)
