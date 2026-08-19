---
title: Persian Plate Recognition
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
---

# Persian Plate Recognition
Recognize Persian plate with YOLOv8

This Repo created for detect persian cars and plates and then recognize every persian characters on the plate.

## Prerequisite
YOLOv8 Ultralytics and all of Requirements for YOLOv8

use python 3.10


## Installation
```
pip install ultralytics==8.0.104
```

## Run
3 options for run:

1.use the main python script
```
python main.py
```

2.use my streamlit link on your browser(this option no need any installation):

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://persian-plate-fa.streamlit.app/)


3.use streamlit app.py and run it locally on your pc:

```
pip install streamlit
cd PersianPlateRecog
streamlit run app.py
```
## Models
for simplicity of computational using yolov8s for cars and plates detection and using yolov8n for character detection
## Training Results
1. yolov8s model for cars and plates detection

2. yolov8n model for characters detection


