# Bio-DETR: A Transformer-based Network for Pest and Seed Detection with Hyperspectral Images

By Y. Di, S. L. Phung, J. Berg, J. Clissold, L. Bui, H. T. Le, and A. Bouzerdoum.

<img src="https://github.com/yangdi-cv/Bio-DETR/blob/master/network/bio-detr.png?raw=true" height="200"/>

This repository is an official PyTorch implementation of [Bio-DETR](https://ieeexplore.ieee.org/document/10650195), published by IJCNN 2024.

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

## Citation
```
@inproceedings{biodetr,
    title={Bio-DETR: A Transformer-based Network for Pest and Seed Detection with Hyperspectral Images},
    author={Yang Di, Son Lam Phung, Julian van den Berg, Jason Clissold, Ly Bui, Hoang Thanh Le, and Abdesselam Bouzerdoum},
    booktitle={International Joint Conference on Neural Networks (IJCNN)},
    pages={1-8},
    year={2024}
}
```
