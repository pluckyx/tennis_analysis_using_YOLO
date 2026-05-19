from ultralytics import YOLO


model = YOLO('yolo26m-pose.pt')
result = model.track('input_video/pic.jpeg', save=True)

