from ultralytics import YOLO
import torch 

if __name__ == '__main__':
    model = YOLO("runs/detect/train3/weights/last.pt") 

    #using cuda if available, otherwise cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    #train
    # results = model.train(
    #     data='C:/Users/neeld/SoccerNet/split/data.yaml',
    #     epochs=100,
    #     imgsz=416,
    #     workers=2, 
    #     batch=6,
    #     device=device, 
    #     amp = True
    # )
    results = model.train(
        resume = True,
        device = device
    )

