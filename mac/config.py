"""Settings: recipe.toml's defaults, local.toml's overrides for this Mac, then environment variables in upper case."""

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JPEG_VERSION = "3.1.4.1"  # the libjpeg-turbo release the engine's JPEG decoder pins (Pillow's, bit for bit)


@dataclass
class Settings:
    source_repo: str = ""
    source_revision: str = ""
    served_name: str = ""
    model_dir: str = ""
    expert_bits: int = 2
    layer_bits: str = ""
    dense_bits: int = 8
    block_dense_bits: int = 8
    tensorfold_src: str = ""
    tensorfold_sha: str = ""
    zig: str = ""
    zig_cpu: str = "native"
    binary: str = ""
    host: str = "127.0.0.1"
    port: int = 8890
    tf_cache: str = ""
    context: int = 524288
    drafts: bool = True
    thinking: bool = False
    max_tokens: int = 32768
    thinking_reserve: int = 0
    temperature: str = "0"
    parallel: int = 4
    prompt_cache_gib: float = 6
    prompt_cache_over_cap: bool = True
    vision: bool = True
    vision_image_tokens: int = 4096
    vision_max_images: int = 4
    vision_urls: bool = False
    keep_warm: int = 86400
    api_key: str = ""
    extra_args: str = ""
    memory_margin_gib: float = 6

    @staticmethod
    def path(p: str) -> Path:
        return Path(os.path.expanduser(p))

    @property
    def model(self) -> Path:
        return self.path(self.model_dir)

    @property
    def cache(self) -> Path:
        return self.path(self.tf_cache)

    @property
    def worktree(self) -> Path:
        return self.cache / "engine" / self.tensorfold_sha[:12]

    @property
    def built_binary(self) -> Path:
        return self.worktree / "zig-out" / "native" / "bin" / "tensorfold-native"

    @property
    def built_grammar(self) -> Path:
        """libtfgrammar (xgrammar), which the server loads beside its binary (../lib) at the first structured request."""
        return self.worktree / "zig-out" / "native" / "lib" / "libtfgrammar.dylib"

    @property
    def run_binary(self) -> Path:
        return self.path(self.binary) if self.binary else self.built_binary

    @property
    def jpeg_lib(self) -> Path:
        return self.cache / f"libjpeg-turbo-{JPEG_VERSION}" / "lib" / "libturbojpeg.0.dylib"

    @property
    def pid_file(self) -> Path:
        return self.cache / "server.pid"

    @property
    def log_file(self) -> Path:
        return self.cache / "server.log"

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}/v1"

    def serve_args(self) -> list[str]:
        a = [str(self.run_binary), "serve", "--host", self.host, "--port", str(self.port), "--name", self.served_name,
             "--context", str(self.context), "--max-tokens", str(self.max_tokens), "--temperature", self.temperature,
             "--parallel", str(self.parallel), "--keep-warm", str(self.keep_warm), "--no-update-check"]
        a.append("--thinking" if self.thinking else "--no-thinking")
        if self.thinking_reserve > 0:
            a += ["--thinking-reserve", str(self.thinking_reserve)]
        if not self.drafts:
            a.append("--no-drafts")
        if self.prompt_cache_gib >= 0:
            a += ["--prompt-cache-gib", f"{self.prompt_cache_gib:g}"]
            if self.prompt_cache_over_cap and self.prompt_cache_gib > 0:
                a.append("--prompt-cache-over-cap")
        if self.vision:
            a += ["--vision", "--vision-image-tokens", str(self.vision_image_tokens),
                  "--vision-max-images", str(self.vision_max_images)]
            if self.vision_urls:
                a.append("--vision-urls")
        if self.api_key:
            a += ["--api-key", self.api_key]
        a += self.extra_args.split()
        a.append(str(self.model))
        return a

    def server_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["TENSORFOLD_LIBTURBOJPEG"] = str(self.jpeg_lib)
        return env


def _coerce(kind: type, raw):
    if kind is bool:
        return raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")
    return kind(raw)


def load(local_file: Path | None = None, env: dict[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    merged: dict = {}
    for f in (ROOT / "recipe.toml", local_file or ROOT / "local.toml"):
        if f.exists():
            for section in tomllib.loads(f.read_text()).values():
                merged.update(section)
    s = Settings()
    for fld in fields(Settings):
        kind = type(getattr(s, fld.name))
        if fld.name in merged:
            setattr(s, fld.name, _coerce(kind, merged[fld.name]))
        if fld.name.upper() in env:
            setattr(s, fld.name, _coerce(kind, env[fld.name.upper()]))
    return s


def validate(s: Settings) -> list[str]:
    """The settings' problems, worded for the user (empty: none)."""
    out = []
    if not s.binary and not s.tensorfold_sha:
        out.append("no engine: set tensorfold_sha (local.toml) or binary")
    if not (s.model / "config.json").exists():
        out.append(f"no converted checkpoint at {s.model}: python3 -m mac convert")
    if not (s.model / "engram" / "hash.safetensors").exists():
        out.append(f"{s.model} has no engram tables (engram/): convert again without --only")
    if s.parallel < 1 or s.parallel > 4:
        out.append("parallel must be 1 to 4 (a shared forward takes four streams at most)")
    if s.vision_image_tokens < 1 or s.vision_max_images < 1:
        out.append("vision_image_tokens and vision_max_images must be at least 1")
    return out
