import cv2

def extract_frames(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    rate = int(cap.get(cv2.CAP_PROP_FPS) / fps)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % rate == 0:
            frames.append(frame)
        i += 1
    cap.release()
    return frames
