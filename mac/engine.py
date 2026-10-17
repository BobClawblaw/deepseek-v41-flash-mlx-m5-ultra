"""The engine: a worktree of the TensorFold checkout at the pinned commit, built under tf_cache (never in the
checkout), and libjpeg-turbo 3.1.4.1 built from its release beside it (the engine's JPEG decoder pins that release)."""

import shutil
import ssl
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from .config import JPEG_VERSION, Settings

JPEG_URL = f"https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/{JPEG_VERSION}/libjpeg-turbo-{JPEG_VERSION}.tar.gz"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def build_engine(s: Settings, rebuild: bool = False) -> Path:
    if s.binary:
        return s.run_binary
    if not s.tensorfold_sha:
        raise SystemExit("no tensorfold_sha pinned (local.toml)")
    src = s.path(s.tensorfold_src)
    if not s.worktree.exists():
        s.worktree.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "-C", str(src), "worktree", "add", "--detach", str(s.worktree), s.tensorfold_sha])
    if rebuild or not s.built_binary.exists():
        _run([s.path(s.zig).as_posix(), "build", "native", "-Doptimize=ReleaseFast", f"-Dcpu={s.zig_cpu}"], cwd=s.worktree)
    return s.built_binary


def build_jpeg(s: Settings) -> Path:
    if s.jpeg_lib.exists():
        return s.jpeg_lib
    work = s.cache / "jpeg-src"
    work.mkdir(parents=True, exist_ok=True)
    tgz = work / f"libjpeg-turbo-{JPEG_VERSION}.tar.gz"
    if not tgz.exists():
        print("downloading", JPEG_URL, flush=True)
        # python.org's Python ships no CA store: the system's bundle
        ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
        with urllib.request.urlopen(JPEG_URL, context=ctx) as r, open(tgz, "wb") as f:
            shutil.copyfileobj(r, f)
    with tarfile.open(tgz) as t:
        t.extractall(work, filter="data")
    srcdir = work / f"libjpeg-turbo-{JPEG_VERSION}"
    build = work / "build"
    prefix = s.jpeg_lib.parent.parent
    _run(["cmake", "-S", str(srcdir), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={prefix}"])
    _run(["cmake", "--build", str(build), "-j", "8"])
    _run(["cmake", "--install", str(build)])
    return s.jpeg_lib


def build(s: Settings, rebuild: bool = False) -> None:
    build_engine(s, rebuild)
    if s.vision:
        build_jpeg(s)
