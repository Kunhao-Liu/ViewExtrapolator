import torch
# from UniDepth.unidepth.models import UniDepthV2
from my_svd_pipeline import StableVideoDiffusionPipeline
from monocular.utils import video_to_numpy, mask_to_numpy, export_to_video
from my_scheduler import EulerDiscreteScheduler
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, required=True, help='video path')
parser.add_argument('--mask', type=str, help='mask video path')
parser.add_argument('--mode', type=str, choices=['static', 'dynamic'], default='static', help='static or dynamic mask')
args = parser.parse_args()


pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt-1-1", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

# Load the guide_video
video = video_to_numpy(args.video)[None,...] # 1, frames, H, W, 3, normalized to [0, 1]
mask = mask_to_numpy(args.mask) if args.mask else None # frames, H, W, in [0, 1]
H,W = video.shape[2], video.shape[3]

generator = torch.manual_seed(42)
    
frames = pipe(video, 
              mask,
            decode_chunk_size=8, 
            fps=6,
            generator=generator,
            # visualize_each_step_path=f"logs/vids",
            num_inference_steps=25,
            resample_steps=3,
            guide_util=15 if args.mode == 'static' else 16,
            ).frames[0]

print('Finish generating video. Exporting...')
output_path = f"{Path(args.video).parent}/refined"
export_to_video(frames, output_path, fps=6, H=H, W=W)