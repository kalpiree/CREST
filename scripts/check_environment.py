import argparse
import importlib.metadata
import json
import sys

PROFILES = {
    "core": {"networkx": None},
    "main": {"torch": "2.4.1+cu121", "transformers": "4.44.2", "tokenizers": "0.19.1", "networkx": None},
    "qwen35": {"torch": "2.6.0+cu124", "transformers": "5.3.0", "tokenizers": "0.22.2", "networkx": None},
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check installed dependencies without loading model weights.")
    parser.add_argument("--profile", choices=list(PROFILES), default="core")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)
    errors, packages = [], {}
    for name, expected in PROFILES[args.profile].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        packages[name] = actual
        if actual is None or expected is not None and actual != expected:
            errors.append(f"{name}: expected {expected or 'installed'}, found {actual}")
    devices = []
    if args.require_cuda:
        try:
            import torch
            devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if not devices:
                errors.append("No CUDA device is available")
            elif not torch.cuda.is_bf16_supported():
                errors.append("The selected CUDA device does not support BF16")
        except ImportError:
            errors.append("PyTorch is unavailable")
    print(json.dumps({"python": sys.version.split()[0], "profile": args.profile,
                      "packages": packages, "devices": devices, "errors": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
