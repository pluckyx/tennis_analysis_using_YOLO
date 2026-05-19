import cv2

def read_video(path):
    cap = cv2.VideoCapture(path)
    frames=[]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(video_frames, output_path):
    fourcc=cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, 24, (video_frames[0].shape[1], video_frames[0].shape[0]))
    for frame in video_frames:
        out.write(frame)
    out.release()
    
    