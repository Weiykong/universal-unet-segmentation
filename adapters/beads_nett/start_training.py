import runpy
import sys
import types
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    train_script = repo_root / "src" / "train.py"
    src_root = str(train_script.parent)

    # Force the optional TensorBoard import in src/train.py to fail fast.
    # The local TensorFlow build hangs when SummaryWriter is instantiated.
    shim = types.ModuleType("torch.utils.tensorboard")
    sys.modules["torch.utils.tensorboard"] = shim

    if src_root not in sys.path:
        sys.path.insert(0, src_root)

    sys.argv = [str(train_script)] + sys.argv[1:]
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
