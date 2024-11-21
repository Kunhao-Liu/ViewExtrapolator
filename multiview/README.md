

# Novel View Extrapolation with Multiview Images


## 1 Installation
Install environment:
```bash
cd multiview
conda env create -f environment.yml 
conda activate ViewExtra_Multi
```


## 2 Datasets
Please download the [LLFF-Extra dataset](https://drive.google.com/drive/folders/1-5noohYSJExO8thRjeeFe65hhhlWpF15?usp=sharing) for novel view extrapolation evaluation and place it in the `data` directory.  

**Custom Data**: For custom-captured scenes, place your captured multiview images in `data/<scene_name>/input`, and use COLMAP to recover camera poses by running `python convert.py -s data/<scene_name>`. Optionally, you can also define a train/test split by adding a `partition.json` file, similar to those in the [LLFF-Extra dataset](https://drive.google.com/drive/folders/1-5noohYSJExO8thRjeeFe65hhhlWpF15?usp=sharing).


## 3 Usage

### 3.1 Train 3D Gaussian Splatting
Train a 3D Gaussian Splatting by:
```bash
python trainer.py --data_dir <data_directory> --result_dir <result_directory>
# for example:
python trainer.py --data_dir data/hugetrike --result_dir results/hugetrike
```
The remote viewer will be automatically launched, where you can view the training process at `http://localhost:8080`.

### 3.2 Render the artifact-prone video

If you use the LLFF-Extra dataset or include the `partition.json` file to specify the test novel view for your custom data, the artifact-prone video and the corresponding opacity mask will be automatically rendered and saved in `<result_directory>/to_refine`. This directory contains all the required data for 3D Gaussian Splatting refinement.

If you do not include the `partition.json` file or wish to define a new test novel view, you can launch the remote viewer using the following command:
```bash
python viewer.py --ckpt <result_directory>/ckpts/ckpt_29999.pt --cfg <result_directory>/cfg.json
# for example:
python viewer.py --ckpt results/hugetrike/ckpts/ckpt_29999.pt --cfg results/hugetrike/cfg.json
```

<img src='https://github.com/user-attachments/assets/6f487069-c6dd-4220-a5f3-d92653dcbcc2' width='50%' />

You can adjust the camera to your desired position and then click `Render Video From Closest View` to render the artifact-prone video.

### 3.3 Refine the artifact-prone video

The artifact-prone video can be refined by:
```bash
python ../refine_video.py --video <result_directory>/to_refine/video.mp4 --mask <result_directory>/to_refine/mask.mp4
# for example:
python ../refine_video.py --video results/hugetrike/to_refine/video.mp4 --mask results/hugetrike/to_refine/mask.mp4
```
The refined video and frames will be saved in `<result_directory>/to_refine/refined`.

### (Optional) 3.4 Refine 3D Gaussian Splatting
With the refined video, you can refine the 3D Gaussian Splatting by:
```bash
python refiner.py --cfg <result_directory>/cfg.json
# for example:
python refiner.py --cfg results/hugetrike/cfg.json
```
The refined 3DGS checkpoint will be saved at `<result_directory>/ckpts/refine_ckpt_9999.pt` and the video rendered by the refined 3DGS will be saved at `<result_directory>/to_refine/refined_gs_render.mp4`.


## 4 Acknowledgements
Our interactive viewer is based on [Viser](https://github.com/nerfstudio-project/viser) and [nerfview](https://github.com/hangg7/nerfview). We thank the authors for their great work and open-sourcing the code.
