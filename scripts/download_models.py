import argparse
import hashlib
import json
import os
from pathlib import Path

MODELS = {
    "qwen25": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    "llama31": ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659"),
    "qwen35": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    "guard": ("meta-llama/Llama-Prompt-Guard-2-86M", "a8ded8e697ce7c355e395a0df51f94adb4a2fd27"),
}


def hashes(path):
    sha256 = hashlib.sha256()
    git = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(block)
            git.update(block)
    return sha256.hexdigest(), git.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download and verify the fixed model checkpoints.")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=["qwen25", "guard"])
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    from huggingface_hub import HfApi, snapshot_download
    for name in args.models:
        repo, revision = MODELS[name]
        root = (Path(args.output_dir) / repo).resolve()
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError("Model revision differs from the requested checkpoint")
        files = [entry for entry in info.siblings if entry.rfilename != ".gitattributes"]
        for entry in files:
            path = root / entry.rfilename
            if not path.resolve().is_relative_to(root):
                raise ValueError("Invalid model repository path")
        if not args.verify_only:
            snapshot_download(repo, revision=revision, local_dir=root,
                              allow_patterns=[entry.rfilename for entry in files], max_workers=4)
        checksums = {}
        for entry in files:
            path = root / entry.rfilename
            if not path.is_file() or path.stat().st_size != entry.size:
                raise ValueError("Missing or wrong-sized model file: " + entry.rfilename)
            sha256, git = hashes(path)
            if entry.lfs is not None:
                expected = entry.lfs.sha256 if hasattr(entry.lfs, "sha256") else entry.lfs["sha256"]
                if sha256 != expected:
                    raise ValueError("Model SHA256 mismatch: " + entry.rfilename)
            elif getattr(entry, "blob_id", None) and git != entry.blob_id:
                raise ValueError("Model Git blob mismatch: " + entry.rfilename)
            checksums[entry.rfilename] = sha256
        marker = {"repo_id": repo, "revision": revision, "files_sha256": checksums, "status": "verified"}
        destination = root / "model_identity.json"
        encoded = json.dumps(marker, indent=2) + "\n"
        if destination.exists():
            if json.loads(destination.read_text()) != marker:
                raise ValueError("Existing model identity differs; use a separate model directory")
        else:
            temporary = root / "model_identity.json.tmp"
            temporary.write_text(encoded)
            os.replace(temporary, destination)
        print(json.dumps({"model": name, "path": str(root), "revision": revision, "verified_files": len(checksums)}))


if __name__ == "__main__":
    main()
