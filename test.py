"""Check that the local Edge AI Python dependencies are installed."""

import torch
from PIL import __version__ as pillow_version
import torchvision


def main() -> None:
    print("Environment is ready:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  TorchVision: {torchvision.__version__}")
    print(f"  Pillow: {pillow_version}")


if __name__ == "__main__":
    main()
