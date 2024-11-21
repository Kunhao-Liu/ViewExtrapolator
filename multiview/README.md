

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



python trainer.py --data_dir data/stove_quant/ --result_dir results/stove_quant
python refiner.py --cfg results/stove_quant/cfg.json


Custom dataset