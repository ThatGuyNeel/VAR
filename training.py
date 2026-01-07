from ultralytics import YOLO
import torch # Import torch to check if GPU is available later

if __name__ == '__main__':
    model = YOLO('yolov8n.pt') 

    # Determine device: Use CUDA if available, otherwise CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Train the model
    results = model.train(
        data='C:/Users/neeld/SoccerNet/model/data.yaml',
        epochs=50,
        imgsz=512,
        workers=2, 
        batch=6, 
        device=device, 
        amp = True
    )