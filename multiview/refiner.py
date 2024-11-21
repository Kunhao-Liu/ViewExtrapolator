import json
import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch.nn.functional as F
import nerfview
import viser
import tqdm
import numpy as np
import imageio

from trainer import Config
from datasets.colmap import Dataset, Parser
from datasets.refine_dataset import Refine_Dataset
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

class Refiner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = "cuda"

        # Load the original dataset 
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=True,
            test_every=cfg.test_every,
        )
        self.train_dataset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
            partition_file=cfg.partition
        )
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale

        # Load the refine dataset
        self.refine_dataset = Refine_Dataset(os.path.join(cfg.result_dir, "to_refine"))

        # Load the pre-optimized gaussian splats
        ckpt_path = os.path.join(cfg.result_dir, "ckpts", "ckpt_29999.pt")
        self.splats = torch.nn.ParameterDict(torch.load(ckpt_path, map_location=self.device)["splats"])

        # init the optimizers
        self._init_optimizer()

        # Densification Strategy
        self.strategy = DefaultStrategy(
            verbose=True,
            # scene_scale=self.scene_scale,
            prune_opa=cfg.prune_opa,
            grow_grad2d=cfg.grow_grad2d,
            grow_scale3d=cfg.grow_scale3d,
            prune_scale3d=cfg.prune_scale3d,
            refine_start_iter=200,
            refine_stop_iter=5000,
            reset_every=1500,
            refine_every=200,
            absgrad=cfg.absgrad,
            revised_opacity=cfg.revised_opacity,
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()


        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            c2ws = np.concatenate([self.parser.camtoworlds, self.refine_dataset.interp_c2ws], axis=0)
            Ks = [self.parser.Ks_dict[camera_id].copy() for camera_id in self.parser.camera_ids] + \
                 [self.refine_dataset.closest_K]*len(self.refine_dataset)
            img_whs = [self.parser.imsize_dict[camera_id] for camera_id in self.parser.camera_ids] + \
                      [self.refine_dataset.img_wh]*len(self.refine_dataset)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="refining",
                c2ws=c2ws,
                Ks=Ks,
                img_whs=img_whs,
                scene_scale=self.scene_scale,
                result_dir=cfg.result_dir,
            )

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb_refine")

    def _init_optimizer(self):
        for param in self.splats.values():
            param.requires_grad = True

        self.optimizers = {
            'means': torch.optim.Adam([{"params": self.splats['means'], "lr": 1e-4 * self.scene_scale}]),
            'scales': torch.optim.Adam([{"params": self.splats['scales'], "lr": 5e-3}]),
            'quats': torch.optim.Adam([{"params": self.splats['quats'], "lr": 1e-3}]),
            'opacities': torch.optim.Adam([{"params": self.splats['opacities'], "lr": 5e-2}]),
            'sh0': torch.optim.Adam([{"params": self.splats['sh0'], "lr": 2.5e-3}]),
            'shN': torch.optim.Adam([{"params": self.splats['shN'], "lr": 2.5e-3/20}]),
        }

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=self.cfg.absgrad,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            **kwargs,
        )
        return render_colors, render_alphas, info
    
    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, render_alphas, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy(), render_alphas.squeeze().cpu().numpy()

    def train(self):
        cfg = self.cfg
        device = self.device

        max_steps = cfg.max_steps//3
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]

        trainloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        refineloader = torch.utils.data.DataLoader(
            self.refine_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        refineloader_iter = iter(refineloader)

        # Training loop.
        densification = True
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:

            if step<=max_steps*1/3:
                is_refine_step = step % 3 == 1
                sh_degree_to_use = 0
            elif step<=max_steps*2/3:
                is_refine_step = step % 5 == 1
                sh_degree_to_use = 1
            else:
                is_refine_step = step % 10 == 1
                sh_degree_to_use = cfg.sh_degree

            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            # get data
            if is_refine_step:
                try:
                    data = next(refineloader_iter)
                except StopIteration:
                    refineloader_iter = iter(refineloader)
                    data = next(refineloader_iter)
            else:
                try:
                    data = next(trainloader_iter)
                except StopIteration:
                    trainloader_iter = iter(trainloader)
                    data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)

            height, width = pixels.shape[1:3]
            
            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB",
            )
            colors = renders

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            if densification:
                self.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            loss.backward()

            if densification:
                self.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

            # logging
            desc = f"loss={loss.item():.3f}"
            pbar.set_description(desc)
            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()


            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()


            # update viewer
            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

        # Save checkpoint
        torch.save(
            {
                "step": max_steps,
                "splats": self.splats.state_dict(),
            },
            f"{cfg.result_dir}/ckpts/refine_ckpt_{step}.pt",
        )

        self.render_refined_video()
        print("Refinement done.")

    @ torch.no_grad()
    def render_refined_video(self):
        
        frames = []
        for idx in range(len(self.refine_dataset.all_image_paths)):
            camtoworlds = torch.from_numpy(self.refine_dataset.all_interp_c2ws[idx]).float()[None,...].to(self.device)
            Ks = torch.from_numpy(self.refine_dataset.closest_K).float()[None,...].to(self.device)
            img_wh = self.refine_dataset.img_wh

            colors, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=img_wh[0],
                height=img_wh[1],
                sh_degree=self.cfg.sh_degree,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                image_ids=idx,
                render_mode="RGB",
            )
            frames.append(np.clip(colors[0].cpu().numpy(),0,1))

        save_dir = f"{self.cfg.result_dir}/to_refine"
        writer = imageio.get_writer(f"{save_dir}/refined_gs_render.mp4", fps=6)
        for frame in frames:
            writer.append_data((frame*255).astype(np.uint8))
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="the path to the config file"
                        ,default='results/bike/cfg.json')
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        config = Config(**json.load(f))
    
    refiner = Refiner(config)
    refiner.train()

    if not config.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

    