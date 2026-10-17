"""The launchd unit: `up --foreground` at login, kept alive, serving the LAN on a port. Written with this Mac's paths
(nothing machine-specific is committed); loading it is left to the user:
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tensorfold.deepseek-v41.plist"""

import shutil
from pathlib import Path
from xml.sax.saxutils import escape

from .config import ROOT, Settings

LABEL = "com.tensorfold.deepseek-v41"


def plist(s: Settings, port: int) -> str:
    python = shutil.which("python3") or "/usr/bin/python3"
    env = {"PORT": str(port), "HOST": "0.0.0.0", "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"}
    env_xml = "".join(f"<key>{k}</key><string>{escape(v)}</string>" for k, v in env.items())
    log = escape(str(s.cache / "launchd.log"))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key>
  <array><string>{escape(python)}</string><string>-m</string><string>mac</string><string>up</string><string>--foreground</string></array>
  <key>WorkingDirectory</key><string>{escape(str(ROOT))}</string>
  <key>EnvironmentVariables</key><dict>{env_xml}</dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>{log}</string>
  <key>StandardErrorPath</key><string>{log}</string>
</dict>
</plist>
"""


def write(s: Settings, port: int) -> str:
    path = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    path.write_text(plist(s, port))
    return f"wrote {path} (not loaded): launchctl bootstrap gui/$(id -u) {path}"
