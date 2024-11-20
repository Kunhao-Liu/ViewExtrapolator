import argparse
import math
import os
import time
from typing import Tuple

import json
from nerfview import CameraState, Viewer
import torch
import torch.nn.functional as F
import viser

from gsplat.rendering import rasterization
from datasets.colmap import Parser

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, help="path to the ckpt file", required=True)
parser.add_argument("--cfg", type=str, help="path to the config file")
parser.add_argument("--port", type=int, default=8080, help="port for the viewer server")
parser.add_argument("--backend", type=str, default="gsplat", help="gsplat, gsplat_legacy, inria")
args = parser.parse_args()
torch.manual_seed(42)
device = "cuda" 

config_path = args.cfg
ckpt_path = args.ckpt

# load config
with open(config_path, "r") as f:
    cfg = json.load(f)

try:
    with open(cfg['partition'], "r") as f:
        partition = json.load(f)
    train_ids = partition['train']
except:
    train_ids = None


# load cameras in dataset
parser = Parser(
    data_dir=cfg['data_dir'],
    factor=cfg['data_factor'],
    normalize=True,
    test_every=cfg['test_every'],
)
c2ws = parser.camtoworlds
Ks = [parser.Ks_dict[camera_id].copy() for camera_id in parser.camera_ids]
img_whs = [parser.imsize_dict[camera_id] for camera_id in parser.camera_ids]
scene_scale = parser.scene_scale * 1.1 * cfg['global_scale']
image_paths = parser.image_paths
result_dir = cfg['result_dir']


# load ckpt
ckpt = torch.load(ckpt_path, map_location=device)["splats"]
means = ckpt["means"]
quats = F.normalize(ckpt["quats"], p=2, dim=-1)
scales = torch.exp(ckpt["scales"])
opacities = torch.sigmoid(ckpt["opacities"])
sh0 = ckpt["sh0"]
shN = ckpt["shN"]
colors = torch.cat([sh0, shN], dim=-2)
sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
print("Number of Gaussians:", len(means))


# register and open viewer
@torch.no_grad()
def viewer_render_fn(camera_state: CameraState, img_wh: Tuple[int, int]):
    width, height = img_wh
    c2w = camera_state.c2w
    K = camera_state.get_K(img_wh)
    c2w = torch.from_numpy(c2w).float().to(device)
    K = torch.from_numpy(K).float().to(device)
    viewmat = c2w.inverse()

    if args.backend == "gsplat":
        rasterization_fn = rasterization
    elif args.backend == "gsplat_legacy":
        from gsplat import rasterization_legacy_wrapper

        rasterization_fn = rasterization_legacy_wrapper
    elif args.backend == "inria":
        from gsplat import rasterization_inria_wrapper

        rasterization_fn = rasterization_inria_wrapper
    else:
        raise ValueError

    render_colors, render_alphas, meta = rasterization_fn(
        means,  # [N, 3]
        quats,  # [N, 4]
        scales,  # [N, 3]
        opacities,  # [N]
        colors,  # [N, 3]
        viewmat[None],  # [1, 4, 4]
        K[None],  # [1, 3, 3]
        width,
        height,
        sh_degree=sh_degree,
        render_mode="RGB",
        # this is to speedup large-scale rendering by skipping far-away Gaussians.
        radius_clip=3,
    )
    render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
    return render_rgbs, render_alphas.squeeze().cpu().numpy()


server = viser.ViserServer(port=args.port, verbose=False)
_ = Viewer(
    server=server,
    render_fn=viewer_render_fn,
    mode="rendering",
    c2ws=c2ws,
    train_ids=train_ids,
    Ks=Ks,
    img_whs=img_whs,
    scene_scale=scene_scale,
    image_paths=image_paths,
    result_dir=result_dir
)
print("Viewer running... Ctrl+C to exit.")
time.sleep(100000)
