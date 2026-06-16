"""Universal U-Net napari plugin."""

from .napari_gui import UNetWidget


def make_unet_widget(viewer: "napari.viewer.Viewer"):
    return UNetWidget(viewer)
