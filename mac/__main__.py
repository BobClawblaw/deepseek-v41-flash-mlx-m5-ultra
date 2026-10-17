"""python3 -m mac up|down|status|logs|build|convert|command|chat|validate"""

import argparse
import shlex
import subprocess
import sys

from . import config, engine, serve


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m mac", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("up", help="start the server (in the background; --foreground for the launchd unit)")
    up.add_argument("--foreground", action="store_true")
    sub.add_parser("down", help="stop the server `up` started")
    sub.add_parser("status", help="pid, url and the server's health")
    logs = sub.add_parser("logs", help="the server log's last lines")
    logs.add_argument("-n", type=int, default=40)
    b = sub.add_parser("build", help="the pinned engine and libjpeg-turbo, under tf_cache")
    b.add_argument("--rebuild", action="store_true")
    sub.add_parser("convert", help="DeepSeek's release (in the Hugging Face cache) to the served checkpoint")
    sub.add_parser("command", help="print the serve command")
    c = sub.add_parser("chat", help="one prompt to the running server")
    c.add_argument("prompt")
    c.add_argument("--max-tokens", type=int, default=128)
    sub.add_parser("validate", help="check the settings, the checkpoint and the build without starting anything")
    a = p.parse_args(argv)
    s = config.load()
    if a.cmd == "up":
        problems = config.validate(s)
        if problems:
            raise SystemExit("\n".join(problems))
        engine.build(s)
        serve.up(s, foreground=a.foreground)
    elif a.cmd == "down":
        serve.down(s)
    elif a.cmd == "status":
        print(serve.status(s))
    elif a.cmd == "logs":
        print("\n".join(s.log_file.read_text(errors="replace").splitlines()[-a.n:]) if s.log_file.exists() else "no log")
    elif a.cmd == "build":
        engine.build(s, a.rebuild)
    elif a.cmd == "convert":
        hf = config.Settings.path(f"~/.cache/huggingface/hub/models--{s.source_repo.replace('/', '--')}/snapshots/{s.source_revision}")
        if not (hf / "model.safetensors.index.json").exists():
            raise SystemExit(f"{s.source_repo}@{s.source_revision[:8]} is not in the Hugging Face cache: "
                             f"hf download {s.source_repo} --revision {s.source_revision}")
        cmd = [sys.executable, str(config.ROOT / "tools" / "convert.py"), str(hf), str(s.model), "--expert-bits",
               str(s.expert_bits), "--layer-bits", s.layer_bits, "--dense-bits", str(s.dense_bits)]
        print("+", shlex.join(cmd), flush=True)
        return subprocess.run(cmd).returncode
    elif a.cmd == "command":
        print(f"TENSORFOLD_LIBTURBOJPEG={s.jpeg_lib} " + shlex.join(s.serve_args()))
    elif a.cmd == "chat":
        print(serve.chat(s, a.prompt, a.max_tokens))
    elif a.cmd == "validate":
        problems = config.validate(s)
        print("\n".join(problems) if problems else "ok")
        print(shlex.join(s.serve_args()))
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
