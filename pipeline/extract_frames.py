import cv2

def extract_frames(video_path, fps=1):
    """
    Extract frames from a video at given fps
    Returns list of frames (numpy arrays)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    rate = max(int(cap.get(cv2.CAP_PROP_FPS) / fps), 1)
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
