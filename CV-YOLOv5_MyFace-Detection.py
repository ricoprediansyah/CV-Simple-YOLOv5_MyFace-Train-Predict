## 1. Install Requirements YOLOv5
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
%pip install -qr requirements.txt # install dependencies
%pip install -q roboflow

import torch
import os
from IPython.display import Image, clear_output  # to display images

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

## 2. Import Labeling Roboflow
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="PZMfN1gExhcqfGkkkd43")
project = rf.workspace("rico-prediansyah").project("myface-v1gul")
version = project.version(1)
dataset = version.download("yolov5")

## 3. set up environment
os.environ["DATASET_DIRECTORY"] = "/content/datasets"

## 4. Training
!python train.py --img 416 --batch 16 --epochs 100 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache
# img: define input image size
# batch: determine batch size
# epochs: define the number of training epochs. (Note: often, 3000+ are common here!)
# data: Our dataset locaiton is saved in the dataset.location
# weights: specify a path to weights to start transfer learning from. Here we choose the generic COCO pretrained checkpoint.
# cache: cache images for faster training

## 5. Testing
!python detect.py --weights runs/train/exp2/weights/best.pt --img 416 --conf 0.1 --source /content/FEED-Terlaksana-Lokakarya_02.png
# weights : location result after training
# source : location file for testing