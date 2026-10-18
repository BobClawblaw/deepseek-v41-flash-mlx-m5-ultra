"""The gate's greedy control: the engine's greedy reply (tf-dsv41-forward DS_DRAFT's greedy.bin) replayed through
ref/gate.py's chains; every reply token must be the chain's argmax at the position before it.

  python gate_greedy.py ids PROMPT.json GREEDY.bin OUT.json      (the prompt and reply, for gate.py)
  python gate_greedy.py check OUT.json REF.npy [REF.npy ...]
"""

import json
import sys

import numpy as np


def main():
    if sys.argv[1] == "ids":
        prompt = json.load(open(sys.argv[2]))
        reply = np.fromfile(sys.argv[3], dtype="<u4").tolist()
        json.dump({"prompt": len(prompt), "ids": prompt + reply}, open(sys.argv[4], "w"))
        json.dump(prompt + reply, open(sys.argv[4] + ".ids", "w"))
        return
    case = json.load(open(sys.argv[2]))
    p, ids = case["prompt"], case["ids"]
    reply = np.array(ids[p:])
    for path in sys.argv[3:]:
        logits = np.load(path)
        pred = logits[p - 1 : len(ids) - 1]
        top = pred.argmax(-1)
        miss = np.nonzero(top != reply)[0]
        print(f"{path}: {len(reply) - len(miss)}/{len(reply)} reply tokens are the reference's argmax")
        for i in miss:
            row = pred[i]
            s = np.sort(row)
            print(f"  reply token {i} (position {p + i}): engine {reply[i]}, reference {top[i]}; reference margin "
                  f"top1-top2 {s[-1] - s[-2]:.4f}, engine's token {row[top[i]] - row[reply[i]]:.4f} below the top")


if __name__ == "__main__":
    main()
