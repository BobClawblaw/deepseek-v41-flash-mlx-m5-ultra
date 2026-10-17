"""DeepSeek's own image preprocessing and tower on the CPU for check images: patches, the patch embedding, block 0,
the tower's normed output and the aligner's rows, for the native tower to compare (tf-dsv41-vision).

  python vision_dump.py CKPT OUT_DIR image...
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from image_processor import load_image  # noqa: E402
from load import load  # noqa: E402


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(16)
    model, _, args = load(ckpt, 4096, vision=True)
    if os.environ.get("AFFINE8"):  # the converter's 8-bit codes for the tower's and aligner's projections
        import run
        names = ("wqkv", "wo", "w1", "w2")
        for mod_name, mod in list(model.vision.named_modules()) + list(model.aligner.named_modules()):
            if isinstance(mod, torch.nn.Linear) and mod_name.rsplit(".", 1)[-1] in names:
                mod.weight.data = run.mlx_affine(mod.weight.data.float(), 8, 64).to(mod.weight.dtype)
        out = out + "-affine8"
        os.makedirs(out, exist_ok=True)
    for path in sys.argv[3:]:
        name = os.path.splitext(os.path.basename(path))[0]
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image({"url": path}, args)
        print(name, "vit grid", n_vit_h, n_vit_w, "llm grid", n_llm_h, n_llm_w, file=sys.stderr)
        with torch.inference_mode():
            vit = model.vision
            x = vit.patch_embed(patches)
            emb = x.clone()
            from vision import get_vision_cos_sin
            cos, sin = get_vision_cos_sin(n_vit_h, n_vit_w, vit.rope_dim, vit.rope_theta)
            block0 = None
            for i, block in enumerate(vit.blocks):
                x = block(x, cos, sin)
                if i == 0:
                    block0 = x.clone()
            tower = vit.norm(x)
            rows = model.aligner(tower, n_vit_h, n_vit_w)
        f = lambda t: t.float().numpy()
        np.savez(os.path.join(out, name + ".npz"), patches=f(patches.reshape(len(patches), -1)), embed=f(emb),
                 block0=f(block0), tower=f(tower), rows=f(rows), grid=np.array([n_vit_h, n_vit_w, n_llm_h, n_llm_w]))


if __name__ == "__main__":
    main()
