import dataclasses
import time
from threading import Lock
from typing import Callable, Literal, Optional, Tuple, Union
import os
import shutil

import numpy as np
import viser
import viser.transforms as vt
from jaxtyping import Float32, UInt8
import matplotlib.pyplot as plt
import imageio

from ._renderer import Renderer, RenderTask


@dataclasses.dataclass
class CameraState(object):
    fov: float
    aspect: float
    c2w: Float32[np.ndarray, "4 4"]

    def get_K(self, img_wh: Tuple[int, int]) -> Float32[np.ndarray, "3 3"]:
        W, H = img_wh
        focal_length = H / 2.0 / np.tan(self.fov / 2.0)
        K = np.array(
            [
                [focal_length, 0.0, W / 2.0],
                [0.0, focal_length, H / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return K


@dataclasses.dataclass
class ViewerState(object):
    num_train_rays_per_sec: Optional[float] = None
    num_view_rays_per_sec: float = 100000.0
    status: Literal[
        "rendering", "preparing", "training", "paused", "completed"
    ] = "training"


VIEWER_LOCK = Lock()


def with_viewer_lock(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        with VIEWER_LOCK:
            return fn(*args, **kwargs)

    return wrapper


class Viewer(object):
    """This is the main class for working with nerfview viewer.

    On instantiation, it (a) binds to a viser server and (b) creates a set of
    GUIs depending on its mode. After user connecting to the server, viewer
    renders and servers images in the background based on the camera movement.

    Args:
        server (viser.ViserServer): The viser server object to bind to.
        render_fn (Callable): A function that takes a camera state and image
            resolution as input and returns an image as a uint8 numpy array.
            Optionally, it can return a tuple of two images, where the second image
            is a float32 numpy depth map.
        mode (Literal["training", "rendering"]): The mode of the viewer.
            Support rendering and training. Defaults to "rendering".
    """

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable[
            [CameraState, Tuple[int, int]],
            Union[
                UInt8[np.ndarray, "H W 3"],
                Tuple[UInt8[np.ndarray, "H W 3"], Optional[Float32[np.ndarray, "H W"]]],
            ],
        ],
        mode: Literal["rendering", "training", "refining"] = "rendering",
        c2ws: Optional[Float32[np.ndarray, "4 4"]] = None,
        Ks: Optional[Float32[np.ndarray, "3 3"]] = None,
        img_whs: Optional[Tuple[int, int]] = None,
        scene_scale: Optional[float] = None,
        image_paths: Optional[str] = None,
        result_dir: Optional[str] = None,
        train_ids = None,
    ):
        self.c2ws = c2ws
        self.Ks = Ks
        self.img_whs = img_whs
        self.scene_scale = scene_scale
        self.image_paths = image_paths
        self.result_dir = result_dir
        self.train_ids = train_ids

        # Public states.
        self.server = server
        self.render_fn = render_fn
        self.mode = mode
        self.lock = VIEWER_LOCK
        self.state = ViewerState()
        if self.mode == "rendering":
            self.state.status = "rendering"

        # Private states.
        self._renderers: dict[int, Renderer] = {}
        self._step: int = 0
        self._last_update_step: int = 0
        self._last_move_time: float = 0.0

        server.on_client_disconnect(self._disconnect_client)
        server.on_client_connect(self._connect_client)

        self._define_guis()

    def _define_guis(self):
        with self.server.gui.add_folder("Stats", visible=(self.mode=="training" or self.mode=="refining")) as self._stats_folder:
            self._stats_text_fn = (
                lambda: f"""<sub>
                Step: {self._step}\\
                Last Update: {self._last_update_step}
                </sub>"""
            )
            self._stats_text = self.server.gui.add_markdown(self._stats_text_fn())

        with self.server.gui.add_folder("Training", visible=(self.mode=="training" or self.mode=="refining")) as self._training_folder:
            self._pause_train_button = self.server.gui.add_button("Pause")
            self._pause_train_button.on_click(self._toggle_train_buttons)
            self._pause_train_button.on_click(self._toggle_train_s)
            self._resume_train_button = self.server.gui.add_button("Resume")
            self._resume_train_button.visible = False
            self._resume_train_button.on_click(self._toggle_train_buttons)
            self._resume_train_button.on_click(self._toggle_train_s)

            self._train_util_slider = self.server.gui.add_slider(
                "Train Util", min=0.0, max=1.0, step=0.05, initial_value=0.9
            )
            self._train_util_slider.on_update(self.rerender)

        with self.server.gui.add_folder("Rendering") as self._rendering_folder:
            self._set_up_direction_button = self.server.gui.add_button("Set Up Direction")
            self._max_img_res_slider = self.server.gui.add_slider(
                "Max Img Res", min=64, max=2048, step=1, initial_value=2048
            )
            self._max_img_res_slider.on_update(self.rerender)
            self._set_up_direction_button.on_click(self._set_up_direction)
            if self.c2ws is not None:
                self._training_view_slider = self.server.gui.add_slider(
                    "Training View",
                    min=0,
                    max=len(self.c2ws) - 1,
                    step=1,
                    initial_value=0,
                )
                self._training_view_slider.on_update(self._set_training_view)

                rainbow_cmap = plt.cm.get_cmap('rainbow', len(self.c2ws))
                self._camera_frustums = []
                # add camera frustum
                for i, c2w in enumerate(self.c2ws):
                    K = self.Ks[i]
                    W, H = self.img_whs[i]
                    fy = K[1, 1]
                    fov = 2 * np.arctan(H / (2 * fy)) 
                    self._camera_frustums.append(
                        self.server.scene.add_camera_frustum(
                        f"training_view{i}",
                        fov=fov,
                        aspect=W/H,
                        scale=self.scene_scale * 0.02,
                        wxyz=vt.SO3.from_matrix(c2w[:3,:3]).wxyz,
                        position=c2w[:3, 3],
                        color=rainbow_cmap(i)[:3],
                    ))

                self._show_camera_frustum_button = self.server.gui.add_checkbox("Show Camera", initial_value=True)
                self._show_camera_frustum_button.on_update(self._toggle_camera_frustum)

                self._render_video_from_closest_view_button = self.server.gui.add_button("Render Video From Closest View", visible=(self.mode!="refining"))
                self._render_video_from_closest_view_button.on_click(self._render_video_from_closest_view)

    def _render_video_from_closest_view(self, _):
        # get the current camera position
        for client in self.server.get_clients().values():
            camera = client.camera
        target_position = camera.position

        if self.train_ids is not None:
            all_c2ws = self.c2ws[self.train_ids]
        else:
            all_c2ws = self.c2ws[:25] if self.mode == "refining" else self.c2ws

        all_positions = [c2w[:3, 3] for c2w in all_c2ws]
        # get the closest view
        closest_view_idx = np.argmin(np.linalg.norm(np.array(all_positions) - target_position, axis=1))
        print(f"Rendering video from closest view: {closest_view_idx}")
        closest_c2w = all_c2ws[closest_view_idx]
        closest_K = self.Ks[closest_view_idx]
        closest_img_wh = self.img_whs[closest_view_idx]
        closest_fov = 2 * np.arctan(closest_img_wh[1] / (2 * closest_K[1, 1]))
        # render video
        target_so3 = vt.SO3(camera.wxyz)
        target_R = target_so3.as_matrix()
        target_c2w = np.concatenate([target_R, camera.position[:, None]], 1)
        target_c2w = np.concatenate([target_c2w, [[0, 0, 0, 1]]], 0)
        interp_c2ws = [closest_c2w + (target_c2w - closest_c2w) * t for t in np.linspace(0, 1, 25)]
        
        # save video
        save_dir = f"{self.result_dir}/to_refine"
        if self.mode == "refining":
            writer = imageio.get_writer(f"{save_dir}/refined_gs_render.mp4", fps=6)
            for i, interp_c2w in enumerate(interp_c2ws):
                cs = CameraState(fov=closest_fov, aspect=closest_img_wh[0] / closest_img_wh[1], c2w=interp_c2w)
                renders = self.render_fn(cs, closest_img_wh)
                rendered_img = (np.clip(renders[0],0,1) * 255).astype(np.uint8)
                writer.append_data(rendered_img)
            writer.close()
            print(f"Video saved to {save_dir}/refined_gs_render.mp4")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        shutil.copy(self.image_paths[closest_view_idx], f'{save_dir}/start_frame.png')
        print(f"Copying the training view {self.image_paths[closest_view_idx]} to {save_dir}")
        writer = imageio.get_writer(f"{save_dir}/video.mp4", fps=6)
        mask_writer = imageio.get_writer(f"{save_dir}/mask.mp4", fps=6)
        for i, interp_c2w in enumerate(interp_c2ws):
            cs = CameraState(fov=closest_fov, aspect=closest_img_wh[0] / closest_img_wh[1], c2w=interp_c2w)
            renders = self.render_fn(cs, closest_img_wh)
            rendered_img = (np.clip(renders[0],0,1) * 255).astype(np.uint8)
            rendered_mask = (renders[1] * 255).astype(np.uint8)

            writer.append_data(rendered_img)
            mask_writer.append_data(rendered_mask)
        writer.close()
        mask_writer.close()
        print(f"Video saved to {save_dir}/video.mp4 and {save_dir}/mask.mp4")

        # save the camera parameters
        np.save(f"{save_dir}/interp_c2ws.npy", np.array(interp_c2ws))
        np.save(f"{save_dir}/closest_K.npy", closest_K)
        np.save(f"{save_dir}/closest_img_wh.npy", closest_img_wh)

    def _toggle_camera_frustum(self, _):
        for camera_frustum in self._camera_frustums:
            camera_frustum.visible = self._show_camera_frustum_button.value

    def _set_training_view(self, _):
        c2w = self.c2ws[self._training_view_slider.value]
        K = self.Ks[self._training_view_slider.value]
        img_wh = self.img_whs[self._training_view_slider.value]

        for client in self.server.get_clients().values():
            client.camera.wxyz = vt.SO3.from_matrix(c2w[:3,:3]).wxyz
            client.camera.position = c2w[:3, 3]
            H = img_wh[1]
            # fx = K[0, 0]
            fy = K[1, 1]
            # horizontal_fov = 2 * np.arctan(W / (2 * fx))
            vertical_fov = 2 * np.arctan(H / (2 * fy)) 
            client.camera.fov = vertical_fov
            

    def _set_up_direction(self, _):
        for client in self.server.get_clients().values():
            client.camera.up_direction = vt.SO3(client.camera.wxyz) @ np.array(
                [0.0, -1.0, 0.0]
            )

    def _toggle_train_buttons(self, _):
        self._pause_train_button.visible = not self._pause_train_button.visible
        self._resume_train_button.visible = not self._resume_train_button.visible

    def _toggle_train_s(self, _):
        if self.state.status == "completed":
            return
        self.state.status = "paused" if self.state.status == "training" else "training"

    def rerender(self, _):
        clients = self.server.get_clients()
        for client_id in clients:
            camera_state = self.get_camera_state(clients[client_id])
            assert camera_state is not None
            self._renderers[client_id].submit(RenderTask("rerender", camera_state))

    def _disconnect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._renderers[client_id].running = False
        self._renderers.pop(client_id)

    def _connect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._renderers[client_id] = Renderer(
            viewer=self, client=client, lock=self.lock
        )
        self._renderers[client_id].start()

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            self._last_move_time = time.time()
            with self.server.atomic():
                camera_state = self.get_camera_state(client)
                self._renderers[client_id].submit(RenderTask("move", camera_state))

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        camera = client.camera
        c2w = np.concatenate(
            [
                np.concatenate(
                    [vt.SO3(camera.wxyz).as_matrix(), camera.position[:, None]], 1
                ),
                [[0, 0, 0, 1]],
            ],
            0,
        )
        return CameraState(
            fov=camera.fov,
            aspect=camera.aspect,
            c2w=c2w,
        )

    def update(self, step: int, num_train_rays_per_step: int):
        if self.mode == "rendering":
            raise ValueError("`update` method is only available in training mode.")
        # Skip updating the viewer for the first few steps to allow
        # `num_train_rays_per_sec` and `num_view_rays_per_sec` to stabilize.
        if step < 5:
            return
        self._step = step
        with self.server.atomic(), self._stats_folder:
            self._stats_text.content = self._stats_text_fn()
        if len(self._renderers) == 0:
            return
        # Stop training while user moves camera to make viewing smoother.
        while time.time() - self._last_move_time < 0.1:
            time.sleep(0.05)
        if self.state.status == "training" and self._train_util_slider.value != 1:
            assert (
                self.state.num_train_rays_per_sec is not None
            ), "User must keep track of `num_train_rays_per_sec` to use `update`."
            train_s = self.state.num_train_rays_per_sec
            view_s = self.state.num_view_rays_per_sec
            train_util = self._train_util_slider.value
            view_n = self._max_img_res_slider.value**2
            train_n = num_train_rays_per_step
            train_time = train_n / train_s
            view_time = view_n / view_s
            update_every = (
                train_util * view_time / (train_time - train_util * train_time)
            )
            if step > self._last_update_step + update_every:
                self._last_update_step = step
                clients = self.server.get_clients()
                for client_id in clients:
                    camera_state = self.get_camera_state(clients[client_id])
                    assert camera_state is not None
                    self._renderers[client_id].submit(
                        RenderTask("update", camera_state)
                    )
                with self.server.atomic(), self._stats_folder:
                    self._stats_text.content = self._stats_text_fn()

    def complete(self):
        self.state.status = "completed"
        self._pause_train_button.disabled = True
        self._resume_train_button.disabled = True
        self._train_util_slider.disabled = True
        with self.server.atomic(), self._stats_folder:
            self._stats_text.content = f"""<sub>
                Step: {self._step}\\
                Training Completed!
                </sub>"""
