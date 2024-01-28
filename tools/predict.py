from ultralytics import RTDETR

model = RTDETR('./runs/detect/bio-detr/bio-detr.pt')

if __name__ == '__main__':
    results = model.predict('./images', save=True)