"""Split a checkpoint's Engram tables into parts under the Hugging Face per-file limit, consecutive rows a part.

  python tools/split_engram.py CHECKPOINT_DIR [--parts 3]

engram/layerL.safetensors (one tensor, "rows": U8 [N, 264]) -> engram/layerL-part0.safetensors ... -partK, each the
same one-tensor layout over its rows; every part is compared with its source bytes before the next is written. The
engine reads either form; remove the whole files after checking the engine on the parts.
"""

import argparse
import json
import os
import struct

import numpy as np

ROW = 264


def header(rows: int) -> bytes:
    h = json.dumps({"__metadata__": {"format": "mlx"},
                    "rows": {"dtype": "U8", "shape": [rows, ROW], "data_offsets": [0, rows * ROW]}}).encode()
    return h + b" " * (-len(h) % 8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--parts", type=int, default=3)
    a = p.parse_args()
    cfg = json.load(open(os.path.join(a.checkpoint, "config.json")))
    for layer in cfg["engram_layer_ids"]:
        src = os.path.join(a.checkpoint, "engram", f"layer{layer}.safetensors")
        with open(src, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            info = json.loads(f.read(n))["rows"]
        rows, width = info["shape"]
        assert width == ROW and info["data_offsets"][0] == 0, info
        data = np.memmap(src, dtype=np.uint8, mode="r", offset=8 + n, shape=(rows, ROW))
        per = -(-rows // a.parts)
        files = {}
        for k in range(a.parts):
            lo, hi = k * per, min(rows, (k + 1) * per)
            out = os.path.join(a.checkpoint, "engram", f"layer{layer}-part{k}.safetensors")
            h = header(hi - lo)
            with open(out, "wb") as f:
                f.write(struct.pack("<Q", len(h)) + h)
                for r0 in range(lo, hi, 1 << 20):
                    f.write(data[r0 : min(hi, r0 + (1 << 20))].tobytes())
            got = np.memmap(out, dtype=np.uint8, mode="r", offset=8 + len(h), shape=(hi - lo, ROW))
            for r0 in range(0, hi - lo, 1 << 20):
                assert np.array_equal(got[r0 : r0 + (1 << 20)], data[lo + r0 : min(hi, lo + r0 + (1 << 20))]), (out, r0)
            files[out] = os.path.getsize(out)
            print(f"layer {layer} part {k}: rows {lo}..{hi}, {files[out] / 1e9:.1f} GB, equal to the source", flush=True)
        cfg.setdefault("engram_files", {})[str(layer)] = [os.path.relpath(f, a.checkpoint) for f in files]
    json.dump(cfg, open(os.path.join(a.checkpoint, "config.json"), "w"), indent=1)
    print("config.json: engram_files lists the parts")


if __name__ == "__main__":
    main()
