

Install environment:
```bash
cd monocular
conda env create -f environment.yml 
conda activate ViewExtra_Mono
pip install -e UniDepth
```


python warp_unidepth.py --image_path .. --direction .. --degree

python DepthCrafter/run.py --video-path examples/car.mp4
python warp_depthcrafter.py --video_path .. --depth_folder ..