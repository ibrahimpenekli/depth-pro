# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
import numpy as np
import PIL.Image
import torch
from matplotlib import pyplot as plt
from cog import BasePredictor, Input, Path, BaseModel
from typing import Optional

import math

from src.depth_pro import create_model_and_transforms, load_rgb
from src.depth_pro.depth_pro import DepthProConfig


MODEL_CACHE = "checkpoints"
MODEL_URL = (
    f"https://weights.replicate.delivery/default/apple/ml-depth-pro/{MODEL_CACHE}.tar"
)

os.environ.update(
    {
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": MODEL_CACHE,
        "TORCH_HOME": MODEL_CACHE,
        "HF_DATASETS_CACHE": MODEL_CACHE,
        "TRANSFORMERS_CACHE": MODEL_CACHE,
        "HUGGINGFACE_HUB_CACHE": MODEL_CACHE,
    }
)

class ModelOutput(BaseModel):
    npz: Path
    color_map: Path
    focal_length: float

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        self.model, self.transform = create_model_and_transforms(
            config=DepthProConfig(
                patch_encoder_preset="dinov2l16_384",
                image_encoder_preset="dinov2l16_384",
                checkpoint_uri=f"./{MODEL_CACHE}/depth_pro.pt",
                decoder_features=256,
                use_fov_head=True,
                fov_encoder_preset="dinov2l16_384",
            ),
            device=torch.device("cuda:0"),
            precision=torch.half,
        )
        self.model.eval()

    def predict(
        self,
        image_path: Path = Input(description="Input image"),
        focal_length: Optional[float] = Input(description="Focal length in pixels (optional)"),
    ) -> ModelOutput:
        image, _, f_px_exif = load_rgb(image_path)

        # Choose which focal length to use for inference
        f_px_used = focal_length if focal_length is not None else f_px_exif

        # Run inference (model will estimate f if None)
        prediction = self.model.infer(self.transform(image), f_px=f_px_used)

        # Decide what to *return* as focal length
        f_px_out = f_px_used
        if f_px_out is None:
            f_est = prediction.get("focallength_px", None)
            if f_est is not None:
                # f_est can be a tensor; convert and sanitize
                try:
                    f_est = float(getattr(f_est.detach().cpu(), "item", lambda: f_est)())
                    print("Estimated focal length:", f_est)
                except Exception:
                    f_est = float(f_est)
                if math.isfinite(f_est):
                    f_px_out = f_est
        else:
            print("Used focal length:", f_px_used)

        # Extract the depth
        depth = prediction["depth"].detach().cpu().numpy().squeeze()
        inverse_depth = 1 / depth

        # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth_vizu = max(1 / 250, inverse_depth.min())
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu
        )

        # Save Depth as npz file.
        out_npz = "/tmp/out.npz"
        np.savez_compressed(out_npz, depth=depth)
        np.savez_compressed("out.npz", depth=depth)

        # Save as color-mapped "turbo" jpg image.
        cmap = plt.get_cmap("turbo")
        color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
        out_color_map = "/tmp/out.jpg"

        PIL.Image.fromarray(color_depth).save(out_color_map, format="JPEG", quality=90)
        PIL.Image.fromarray(color_depth).save("out.jpg", format="JPEG", quality=90)

        return ModelOutput(
            npz=Path(out_npz),
            color_map=Path(out_color_map),
            focal_length=float(f_px_out)
        )
