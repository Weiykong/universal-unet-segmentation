"""
Napari GUI for Universal U-Net — training and prediction.
Run with:  python src/napari_gui.py
"""

import sys
import os
import re
import glob
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import tifffile
import napari
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QTabWidget, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QPushButton, QTextEdit, QFileDialog,
    QGroupBox, QScrollArea, QSlider, QProgressBar, QSizePolicy,
)
from qtpy.QtCore import Qt, QThread, Signal, QObject, QTimer
from qtpy.QtGui import QFont


SUPPORTED_EXTENSIONS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")


def _device_name():
    try:
        import torch
        if torch.cuda.is_available():
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        if torch.backends.mps.is_available():
            return "Apple MPS (GPU)"
        return "CPU"
    except Exception:
        return "unknown"


# ── stdout redirector ────────────────────────────────────────────────────────

class _LogStream(QObject):
    text = Signal(str)

    def write(self, msg):
        if msg:
            self.text.emit(str(msg))

    def flush(self):
        pass


# ── worker base ──────────────────────────────────────────────────────────────

class _Worker(QThread):
    log      = Signal(str)
    finished = Signal(bool)

    def _redirect(self):
        stream = _LogStream()
        stream.text.connect(self.log)
        self._old_out, self._old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = stream

    def _restore(self):
        sys.stdout, sys.stderr = self._old_out, self._old_err


class TrainWorker(_Worker):
    progress = Signal(int, int)   # current_epoch, total_epochs

    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs

    def run(self):
        stream = _LogStream()
        stream.text.connect(self.log)
        stream.text.connect(self._scan_progress)
        self._old_out, self._old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = stream
        try:
            from train import train
            train(**self.kwargs)
            self.finished.emit(True)
        except Exception:
            import traceback
            self.log.emit(f"\n[ERROR]\n{traceback.format_exc()}\n")
            self.finished.emit(False)
        finally:
            self._restore()

    def _scan_progress(self, text):
        for m in re.finditer(r"Epoch\s+(\d+)/(\d+)", text):
            cur, tot = int(m.group(1)), int(m.group(2))
            self.progress.emit(cur, tot)


class PredictWorker(_Worker):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs

    def run(self):
        self._redirect()
        try:
            from inference import predict_folder
            predict_folder(**self.kwargs)
            self.finished.emit(True)
        except Exception:
            import traceback
            self.log.emit(f"\n[ERROR]\n{traceback.format_exc()}\n")
            self.finished.emit(False)
        finally:
            self._restore()


# ── shared image loader ───────────────────────────────────────────────────────

def _load_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        return tifffile.imread(path).astype(np.float32)
    from PIL import Image
    img = np.array(Image.open(path)).astype(np.float32)
    if img.ndim == 3 and img.shape[2] == 3:
        img = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    return img


def _glob_images(directory):
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)


def _add_or_replace(viewer, arr, name, kind="image", **kwargs):
    """Add a layer to napari, replacing any existing layer with the same name."""
    for l in list(viewer.layers):
        if l.name == name:
            viewer.layers.remove(l)
    if kind == "image":
        viewer.add_image(arr, name=name, **kwargs)
    elif kind == "labels":
        viewer.add_labels(arr.astype(int), name=name, **kwargs)


def _maybe_stack(paths, loader):
    """If all images share a shape, stack into one (N, H, W) array for scrubbing.
    Returns (array, is_stack). Otherwise returns (None, False)."""
    if len(paths) < 2:
        return None, False
    arrs = []
    shape = None
    for p in paths:
        a = loader(p)
        if a.ndim != 2:
            return None, False
        if shape is None:
            shape = a.shape
        elif a.shape != shape:
            return None, False
        arrs.append(a)
    return np.stack(arrs, axis=0), True


# ── UI helpers ───────────────────────────────────────────────────────────────

def _section(title):
    grp  = QGroupBox(title)
    form = QFormLayout(grp)
    form.setLabelAlignment(Qt.AlignRight)
    return grp, form


def _folder_row(parent, placeholder=""):
    row  = QWidget(parent)
    hl   = QHBoxLayout(row); hl.setContentsMargins(0, 0, 0, 0)
    edit = QLineEdit(placeholder)
    btn  = QPushButton("Browse…")
    btn.setMaximumWidth(80)
    hl.addWidget(edit); hl.addWidget(btn)

    def browse():
        d = QFileDialog.getExistingDirectory(row, "Select folder", edit.text() or ".")
        if d:
            edit.setText(d)
            edit.editingFinished.emit()

    btn.clicked.connect(browse)
    return row, edit


def _file_row(parent, placeholder="", filt="All files (*)"):
    row  = QWidget(parent)
    hl   = QHBoxLayout(row); hl.setContentsMargins(0, 0, 0, 0)
    edit = QLineEdit(placeholder)
    btn  = QPushButton("Browse…")
    btn.setMaximumWidth(80)
    hl.addWidget(edit); hl.addWidget(btn)

    def browse():
        f, _ = QFileDialog.getOpenFileName(row, "Select file", edit.text() or ".", filt)
        if f:
            edit.setText(f)
            edit.editingFinished.emit()

    btn.clicked.connect(browse)
    return row, edit


def _log_box(height=150):
    box = QTextEdit()
    box.setReadOnly(True)
    box.setFont(QFont("Menlo", 11))
    box.setMinimumHeight(height)
    return box


class IntSlider(QWidget):
    """A QSlider synced with a QSpinBox, shown side by side."""
    def __init__(self, lo, hi, val, step=1, tooltip=""):
        super().__init__()
        hl = QHBoxLayout(self); hl.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(lo, hi); self.slider.setSingleStep(step); self.slider.setValue(val)
        self.spin = QSpinBox()
        self.spin.setRange(lo, hi); self.spin.setSingleStep(step); self.spin.setValue(val)
        self.spin.setMaximumWidth(80)
        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        hl.addWidget(self.slider); hl.addWidget(self.spin)
        if tooltip:
            self.setToolTip(tooltip); self.slider.setToolTip(tooltip); self.spin.setToolTip(tooltip)

    def value(self):
        return self.spin.value()


class FloatSlider(QWidget):
    """A QSlider (integer-backed) synced with a QDoubleSpinBox."""
    valueChanged = Signal(float)

    def __init__(self, lo, hi, val, decimals=2, step=0.01, tooltip=""):
        super().__init__()
        self._lo, self._hi, self._dec = lo, hi, decimals
        self._scale = 10 ** decimals
        hl = QHBoxLayout(self); hl.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(lo * self._scale), int(hi * self._scale))
        self.slider.setValue(int(val * self._scale))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi); self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step); self.spin.setValue(val)
        self.spin.setMaximumWidth(90)
        self.slider.valueChanged.connect(lambda v: self._sync_spin(v))
        self.spin.valueChanged.connect(lambda v: self._sync_slider(v))
        hl.addWidget(self.slider); hl.addWidget(self.spin)
        if tooltip:
            self.setToolTip(tooltip); self.slider.setToolTip(tooltip); self.spin.setToolTip(tooltip)

    def _sync_spin(self, slider_val):
        v = slider_val / self._scale
        if abs(self.spin.value() - v) > 1e-9:
            self.spin.blockSignals(True); self.spin.setValue(v); self.spin.blockSignals(False)
        self.valueChanged.emit(v)

    def _sync_slider(self, spin_val):
        iv = int(round(spin_val * self._scale))
        if self.slider.value() != iv:
            self.slider.blockSignals(True); self.slider.setValue(iv); self.slider.blockSignals(False)
        self.valueChanged.emit(spin_val)

    def value(self):
        return self.spin.value()


# ── Train tab ────────────────────────────────────────────────────────────────

class TrainTab(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self._viewer = viewer

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.NoFrame)
        inner = QWidget()
        root  = QVBoxLayout(inner)
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # ── Data paths ──────────────────────────────────────────────────────
        grp, form = _section("Data Paths")
        _, self._img  = _folder_row(self, "data/images");  self._img.setText("data/images")
        _, self._mask = _folder_row(self, "data/masks");   self._mask.setText("data/masks")
        _, self._save = _folder_row(self, "models");       self._save.setText("models")
        _, self._resume = _file_row(self, "", "Checkpoint (*.pth)")

        self._img.editingFinished.connect(self._preview_data)
        self._mask.editingFinished.connect(self._preview_data)

        form.addRow("Images dir:",    self._img.parent())
        form.addRow("Masks dir:",     self._mask.parent())
        form.addRow("Save dir:",      self._save.parent())
        form.addRow("Resume (opt.):", self._resume.parent())

        self._stack_cb = QCheckBox("Stack same-size images into one scrubbable layer")
        self._stack_cb.setChecked(True)
        form.addRow("", self._stack_cb)

        preview_btn = QPushButton("👁  Preview Data in Napari")
        preview_btn.clicked.connect(self._preview_data)
        form.addRow("", preview_btn)
        root.addWidget(grp)

        # ── Model architecture ───────────────────────────────────────────────
        grp, form = _section("Model Architecture")
        self._depth = IntSlider(2, 6, 4, tooltip="Number of encoder/decoder levels. Deeper = larger receptive field, more memory.")
        self._base  = IntSlider(8, 128, 64, step=8, tooltip="Feature channels at the first level; doubles each level down.")
        self._norm  = QComboBox(); self._norm.addItems(["batch", "group", "instance"])
        self._norm.setToolTip("Normalization. 'group' generalizes best for small batches / cross-domain data.")
        self._outch = IntSlider(1, 16, 1, tooltip="Output classes. 1 = binary (sigmoid). N>1 = multi-class (softmax).")
        self._attention = QCheckBox("Enable Attention Gates in skip connections")
        self._se_block = QCheckBox("Enable Squeeze-and-Excitation in residual units")
        form.addRow("Depth:", self._depth)
        form.addRow("Base features:", self._base)
        form.addRow("Norm layer:", self._norm)
        form.addRow("Out channels:", self._outch)
        form.addRow("Attention Gates:", self._attention)
        form.addRow("SE Blocks:", self._se_block)
        root.addWidget(grp)

        # ── Training hyperparameters ─────────────────────────────────────────
        grp, form = _section("Training")
        self._epochs = IntSlider(1, 1000, 50, tooltip="Number of full passes over the dataset.")
        self._bs     = IntSlider(1, 32, 4, tooltip="Images per gradient step. Lower if you run out of memory.")
        self._lr     = QComboBox()
        self._lr.addItems(["1e-3", "5e-4", "2e-4", "1e-4", "5e-5", "1e-5"])
        self._lr.setCurrentText("1e-4")
        self._lr.setEditable(True)
        self._lr.setToolTip("Adam learning rate.")
        self._val    = FloatSlider(0.0, 0.5, 0.2, decimals=2, step=0.05,
                                   tooltip="Fraction of data held out for validation.")
        self._crop   = IntSlider(64, 1024, 512, step=64,
                                 tooltip="Random crop size during training. Larger = more context, more memory.")
        self._augment = QCheckBox("Enable augmentation (flips, rotations, elastic, intensity)")
        form.addRow("Epochs:", self._epochs)
        form.addRow("Batch size:", self._bs)
        form.addRow("Learning rate:", self._lr)
        form.addRow("Validation split:", self._val)
        form.addRow("Crop size (px):", self._crop)
        form.addRow("", self._augment)
        root.addWidget(grp)

        # ── Run ──────────────────────────────────────────────────────────────
        self._run_btn = QPushButton("▶  Start Training")
        self._run_btn.setMinimumHeight(38)
        self._run_btn.clicked.connect(self._start)
        root.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.setFormat("Epoch %v / %m")
        self._progress.hide()
        root.addWidget(self._progress)

        self._log = _log_box()
        root.addWidget(self._log)

        self._worker = None

    def _preview_data(self):
        img_dir  = self._img.text()
        mask_dir = self._mask.text()
        img_files  = _glob_images(img_dir)  if os.path.isdir(img_dir)  else []
        mask_files = _glob_images(mask_dir) if os.path.isdir(mask_dir) else []

        if not img_files and not mask_files:
            self._log.append(f"No images found in {img_dir} or {mask_dir}\n")
            return

        loaded = self._load_set(img_files, "img", is_mask=False)
        loaded += self._load_set(mask_files, "mask", is_mask=True)
        self._log.append(f"Previewing {loaded} file(s) from data dirs.\n")

    def _load_set(self, files, tag, is_mask):
        if not files:
            return 0
        if self._stack_cb.isChecked():
            stack, ok = _maybe_stack(files, _load_image)
            if ok:
                if is_mask:
                    stack = (stack > (127 if stack.max() > 1 else 0.5)).astype(np.uint8)
                    _add_or_replace(self._viewer, stack, f"[{tag} stack] {len(files)} files", kind="labels")
                else:
                    _add_or_replace(self._viewer, stack, f"[{tag} stack] {len(files)} files",
                                    kind="image", colormap="gray")
                return len(files)
        # fall back to individual layers
        n = 0
        for path in files:
            try:
                arr = _load_image(path)
                name = f"[{tag}] {os.path.basename(path)}"
                if is_mask:
                    arr = (arr > (127 if arr.max() > 1 else 0.5)).astype(np.uint8)
                    _add_or_replace(self._viewer, arr, name, kind="labels")
                else:
                    _add_or_replace(self._viewer, arr, name, kind="image", colormap="gray")
                n += 1
            except Exception as e:
                self._log.append(f"Could not load {path}: {e}\n")
        return n

    def _start(self):
        if self._worker and self._worker.isRunning():
            self._log.append("[already running — wait for it to finish]\n")
            return

        try:
            lr = float(self._lr.currentText())
        except ValueError:
            self._log.append("Invalid learning rate.\n")
            return

        kwargs = dict(
            epochs        = self._epochs.value(),
            batch_size    = self._bs.value(),
            lr            = lr,
            augment       = self._augment.isChecked(),
            val_split     = self._val.value(),
            depth         = self._depth.value(),
            base_features = self._base.value(),
            crop_size     = self._crop.value(),
            resume        = self._resume.text() or None,
            norm          = self._norm.currentText(),
            out_channels  = self._outch.value(),
            attention     = self._attention.isChecked(),
            se_block      = self._se_block.isChecked(),
            image_dir     = self._img.text(),
            mask_dir      = self._mask.text(),
            save_dir      = self._save.text(),
        )

        self._log.clear()
        self._log.append(
            f"Device : {_device_name()}\n"
            f"Images : {kwargs['image_dir']}\n"
            f"Masks  : {kwargs['mask_dir']}\n"
            f"Save   : {kwargs['save_dir']}\n"
            f"Epochs : {kwargs['epochs']}  |  BS: {kwargs['batch_size']}  |  LR: {kwargs['lr']}\n"
            f"Depth  : {kwargs['depth']}  |  Base: {kwargs['base_features']}  |  Norm: {kwargs['norm']}\n"
            f"Attn   : {kwargs['attention']}  |  SE: {kwargs['se_block']}  |  Aug: {kwargs['augment']}\n"
            f"Classes: {kwargs['out_channels']}  |  Crop: {kwargs['crop_size']}\n\n"
        )
        self._progress.setRange(0, kwargs['epochs'])
        self._progress.setValue(0)
        self._progress.show()
        self._run_btn.setEnabled(False)

        self._worker = TrainWorker(kwargs)
        self._worker.log.connect(self._append)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._done)
        self._worker.start()

    def _on_progress(self, cur, tot):
        self._progress.setRange(0, tot)
        self._progress.setValue(cur)

    def _append(self, text):
        self._log.moveCursor(self._log.textCursor().End)
        self._log.insertPlainText(text)
        self._log.ensureCursorVisible()

    def _done(self, ok):
        self._run_btn.setEnabled(True)
        if ok:
            self._progress.setValue(self._progress.maximum())
            self._log.append("\n✓ Training complete.")
        else:
            self._log.append("\n✗ Training failed — see error above.")


# ── Predict tab ──────────────────────────────────────────────────────────────

class PredictTab(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self._viewer = viewer
        self._prob_cache = {}   # base filename -> prob array (0..255 float)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.NoFrame)
        inner = QWidget()
        root  = QVBoxLayout(inner)
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # ── Paths ────────────────────────────────────────────────────────────
        grp, form = _section("Paths")
        _, self._inp   = _folder_row(self, "data/inference_input"); self._inp.setText("data/inference_input")
        _, self._out   = _folder_row(self, "output");               self._out.setText("output")
        _, self._model = _file_row(self, "models/best_model.pth", "Checkpoint (*.pth)")
        self._model.setText("models/best_model.pth")

        self._inp.editingFinished.connect(self._preview_inputs)

        form.addRow("Input dir:",  self._inp.parent())
        form.addRow("Output dir:", self._out.parent())
        form.addRow("Model:",      self._model.parent())

        preview_btn = QPushButton("👁  Preview Input Images")
        preview_btn.clicked.connect(self._preview_inputs)
        form.addRow("", preview_btn)
        root.addWidget(grp)

        # ── Options ──────────────────────────────────────────────────────────
        grp, form = _section("Options")
        self._thresh = FloatSlider(0.01, 0.99, 0.5, decimals=2, step=0.05,
                                   tooltip="Probability cutoff for the binary mask. Drag to re-threshold loaded results live.")
        self._thresh.valueChanged.connect(self._live_rethreshold)
        self._show_inputs_cb = QCheckBox("Show original images alongside results")
        self._show_inputs_cb.setChecked(True)
        form.addRow("Threshold (binary):", self._thresh)
        form.addRow("", self._show_inputs_cb)
        root.addWidget(grp)

        # ── Run ──────────────────────────────────────────────────────────────
        self._run_btn = QPushButton("▶  Run Prediction")
        self._run_btn.setMinimumHeight(38)
        self._run_btn.clicked.connect(self._start)
        root.addWidget(self._run_btn)

        self._log = _log_box()
        root.addWidget(self._log)

        self._worker = None

    def _preview_inputs(self):
        inp_dir = self._inp.text()
        if not os.path.isdir(inp_dir):
            return
        files = _glob_images(inp_dir)
        if not files:
            self._log.append(f"No images found in {inp_dir}\n")
            return
        loaded = 0
        for path in files:
            try:
                arr  = _load_image(path)
                name = f"[input] {os.path.basename(path)}"
                _add_or_replace(self._viewer, arr, name, kind="image", colormap="gray")
                loaded += 1
            except Exception as e:
                self._log.append(f"Could not load {path}: {e}\n")
        self._log.append(f"Loaded {loaded} input image(s).\n")

    def _start(self):
        if self._worker and self._worker.isRunning():
            self._log.append("[already running]\n")
            return

        inp_dir = self._inp.text()
        kwargs  = dict(
            input_dir  = inp_dir,
            output_dir = self._out.text(),
            model_path = self._model.text(),
            threshold  = self._thresh.value(),
        )

        self._log.clear()
        self._log.append(
            f"Device : {_device_name()}\n"
            f"Input  : {kwargs['input_dir']}\n"
            f"Output : {kwargs['output_dir']}\n"
            f"Model  : {kwargs['model_path']}\n"
            f"Thresh : {kwargs['threshold']}\n\n"
        )

        if self._show_inputs_cb.isChecked():
            self._preview_inputs()

        self._run_btn.setEnabled(False)

        self._worker = PredictWorker(kwargs)
        self._worker.log.connect(self._append)
        self._worker.finished.connect(
            lambda ok: self._done(ok, kwargs['input_dir'], kwargs['output_dir'])
        )
        self._worker.start()

    def _append(self, text):
        self._log.moveCursor(self._log.textCursor().End)
        self._log.insertPlainText(text)
        self._log.ensureCursorVisible()

    def _done(self, ok, inp_dir, out_dir):
        self._run_btn.setEnabled(True)
        if ok:
            self._log.append("\n✓ Prediction complete.")
            self._load_results(inp_dir, out_dir)
        else:
            self._log.append("\n✗ Prediction failed — see error above.")

    def _load_results(self, inp_dir, out_dir):
        loaded = 0
        self._prob_cache.clear()

        if self._show_inputs_cb.isChecked():
            for path in _glob_images(inp_dir):
                try:
                    arr  = _load_image(path)
                    name = f"[input] {os.path.basename(path)}"
                    _add_or_replace(self._viewer, arr, name, kind="image", colormap="gray")
                    loaded += 1
                except Exception:
                    pass

        # Probability maps — cache for live re-thresholding
        for ext in ("*.tif", "*.tiff"):
            for f in sorted(glob.glob(os.path.join(out_dir, f"prob_{ext[1:]}"))):
                try:
                    arr  = tifffile.imread(f).astype(np.float32)
                    base = os.path.basename(f)[len("prob_"):]
                    self._prob_cache[base] = arr
                    _add_or_replace(self._viewer, arr, f"[prob] {base}", kind="image",
                                    colormap="magma", opacity=0.7)
                    loaded += 1
                except Exception as e:
                    self._log.append(f"Could not load {f}: {e}\n")

        # Multi-class label maps
        for ext in ("*.tif", "*.tiff"):
            for f in sorted(glob.glob(os.path.join(out_dir, f"labels_{ext[1:]}"))):
                try:
                    arr  = tifffile.imread(f)
                    _add_or_replace(self._viewer, arr.astype(int),
                                    f"[labels] {os.path.basename(f)[len('labels_'):]}", kind="labels")
                    loaded += 1
                except Exception as e:
                    self._log.append(f"Could not load {f}: {e}\n")

        # Build mask layers from cached probabilities at the current threshold
        self._live_rethreshold(self._thresh.value())

        self._log.append(f"Loaded {loaded} layer(s) into napari.")
        if self._prob_cache:
            self._log.append("\nDrag the threshold slider to re-threshold masks live.")

    def _live_rethreshold(self, thresh):
        """Recompute binary mask layers from cached probability maps without re-running."""
        if not self._prob_cache:
            return
        cut = thresh * 255.0
        for base, prob in self._prob_cache.items():
            mask = (prob > cut).astype(np.uint8)
            _add_or_replace(self._viewer, mask, f"[mask @{thresh:.2f}] {base}", kind="labels")
        # remove stale mask layers from other thresholds
        keep = {f"[mask @{thresh:.2f}] {b}" for b in self._prob_cache}
        for l in list(self._viewer.layers):
            if l.name.startswith("[mask @") and l.name not in keep:
                self._viewer.layers.remove(l)


# ── Main widget ──────────────────────────────────────────────────────────────

class UNetWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        viewer = napari_viewer
        self.setMinimumWidth(460)
        root = QVBoxLayout(self)

        title = QLabel("Universal U-Net")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 15, QFont.Bold))
        root.addWidget(title)

        dev = QLabel(f"Compute device: {_device_name()}")
        dev.setAlignment(Qt.AlignCenter)
        dev.setStyleSheet("color: gray;")
        root.addWidget(dev)

        tabs = QTabWidget()
        tabs.addTab(TrainTab(viewer),   "🏋  Train")
        tabs.addTab(PredictTab(viewer), "🔍  Predict")
        root.addWidget(tabs)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    viewer = napari.Viewer(title="Universal U-Net")
    widget = UNetWidget(viewer)
    viewer.window.add_dock_widget(widget, name="U-Net", area="right")
    napari.run()


if __name__ == "__main__":
    main()
