
# Novel View Extrapolation with Single Image or Monocular Video

## 1 Installation
Install environment:
```bash
cd monocular
git submodule update --init --recursive
conda env create -f environment.yml 
conda activate ViewExtra_Mono
pip install -e UniDepth
```

## 2 Single Image

### 2.1 Render the artifact-prone video
Given a single image, you can render the artifact-prone video with the projected point cloud by:
```bash
python warp_unidepth.py --image_path <image_path> --direction <'up','down','left','right'> --degree <degree_of_moving> --look_at_depth <look_at_depth>
# for example:
python warp_unidepth.py --image_path examples/train.jpg --direction right --degree 15 --look_at_depth 1
```
The rendered artifact-prone video as well as the mask will be saved in `logs`.

### 2.2 Refine the artifact-prone video
The artifact-prone video can be refined by:
```bash
python ../refine_video.py --video logs/video.mp4 --mask logs/mask.mp4
```
The refined video and frames will be saved in `logs/refined`.

## 3 Monocular Video

### 3.1 Render the artifact-prone video
Given a monocular video, you need to first estimate its depth by:
```bash
python DepthCrafter/run.py --video-path <video_path>
# for example
python DepthCrafter/run.py --video-path examples/hike.mp4
```
The estimated depth will be saved in `demo_output`. With the estimated depth, you can render the artifact-prone video with the projected point cloud by:
```bash
python warp_depthcrafter.py --video_path <video_path> --depth_folder demo_output --direction <'up','down','left','right'> --degree <degree_of_moving> --look_at_depth <look_at_depth>
# for example:
python warp_depthcrafter.py --video_path examples/hike.mp4 --depth_folder demo_output --direction left --degree 10 --look_at_depth 1
```
The rendered artifact-prone video as well as the mask will be saved in `logs`.

### 3.2 Refine the artifact-prone video
The artifact-prone video can be refined by:
```bash
python ../refine_video.py --video logs/video.mp4 --mask logs/mask.mp4 --mode dynamic
```
The refined video and frames will be saved in `logs/refined`.

## 4 Acknowledgements
We use the depth estimators offered by [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) and [DepthCrafter](https://github.com/Tencent/DepthCrafter). We thank the authors for their great work and open-sourcing the code.