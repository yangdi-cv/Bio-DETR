# Bio-DETR: A Transformer-based Network for Pest and Seed Detection with Hyperspectral Images

<img src="https://github.com/yangdi-cv/Bio-DETR/blob/master/network/bio-detr.png?raw=true" height="180"/>

## Dependencies
```sh
pip install .
```

## Inference
```sh
python tools/predict.py
```

## Training
```sh
yolo task=detect mode=train model=cfg/network/rtdetr_bio-detr.yaml data=cfg/dataset/hsi-bio.yaml batch=16 epochs=300
```

Dataset: https://drive.google.com/drive/folders/1QiMLZopLLVuNav_py7TyrwpI6h3WVJ25?usp=sharing
