"""
resave_checkpoints.py
----------------------
Converts old DQN checkpoint files (*.pth) into safe state_dict-only
checkpoints that PyTorch >=2.6 can load.

✅ Input folder (fixed):  
    D:/Desktop Material/DAD_Computing/TaskOffloadingOptimization/results/CyberShake_Models_Colab

✅ Output folder (auto-created):  
    D:/Desktop Material/DAD_Computing/TaskOffloadingOptimization/results/CyberShake_Models_Colab_safe
"""

import torch
import numpy as np
from pathlib import Path

# Fixed paths
INPUT_FOLDER = Path(r"D:\Desktop Material\DAD_Computing\TaskOffloadingOptimization\results\CyberShake_Models_Colab")
OUTPUT_FOLDER = Path(r"D:\Desktop Material\DAD_Computing\TaskOffloadingOptimization\results\CyberShake_Models_Colab_safe")

def resave_checkpoints(input_folder: Path, output_folder: Path):
    output_folder.mkdir(parents=True, exist_ok=True)

    # Allow legacy numpy objects
    torch.serialization.add_safe_globals([
        np.dtype,
        np._core.multiarray.scalar,
    ])

    for p in sorted(input_folder.glob("*.pth")):
        print(f"Processing {p.name} ...")
        try:
            # Force old-style load (safe because you trust the files)
            ckpt = torch.load(p, map_location="cpu", weights_only=False)

            # Extract state_dict
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                sd = ckpt["model_state_dict"]
                input_dim = ckpt.get("input_dim", None)
                output_dim = ckpt.get("output_dim", None)
            elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                # already just a state_dict
                sd = ckpt
                input_dim, output_dim = None, None
            else:
                # fallback: find nested state_dict
                sd, input_dim, output_dim = None, None, None
                for k, v in ckpt.items():
                    if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                        sd = v
                        break
                if sd is None:
                    print(f"  ❌ Could not extract state_dict from {p.name}")
                    continue

            # Save safe checkpoint
            safe_ckpt = {
                "model_state_dict": sd,
                "input_dim": input_dim,
                "output_dim": output_dim,
            }
            out_path = output_folder / f"{p.stem}.state_dict.pth"
            torch.save(safe_ckpt, out_path)
            print(f"  ✅ Saved {out_path.name}")

        except Exception as e:
            print(f"  ❌ Error on {p.name}: {e}")

if __name__ == "__main__":
    resave_checkpoints(INPUT_FOLDER, OUTPUT_FOLDER)
