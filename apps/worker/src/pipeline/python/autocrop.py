import sys
import json
import cv2
import os

def detect_face_opencv(video_path, limit_seconds=0):
    print(f"Analyzing {video_path} for crop...", file=sys.stderr)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Analyze 1 frame per 2 seconds
    step = int(fps * 2) 
    
    limit_frames = int(limit_seconds * fps) if limit_seconds > 0 else total_frames
    
    centers_x = []
    
    # Targeting a smaller height for faster detection (e.g. 360p)
    target_detection_height = 360
    scale = target_detection_height / height
    
    frame_idx = 0
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            break

        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # detectMultiScale on smaller image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            # Pick the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            (xf, yf, wf, hf) = largest_face
            # Normalized center X (relative to small_frame width)
            small_width = small_frame.shape[1]
            center_x = (xf + wf / 2) / small_width
            centers_x.append(center_x)
        
        frame_idx += step
        if frame_idx >= total_frames or frame_idx >= limit_frames:
            break
            
    cap.release()
    
    if not centers_x:
        return 0.5
        
    centers_x.sort()
    # Use median
    return centers_x[len(centers_x) // 2]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_file")
    parser.add_argument("--limit_seconds", type=int, default=0)
    args = parser.parse_args()
    
    video_file = args.video_file
    limit_seconds = args.limit_seconds
    
    try:
        # Pass limit to detect_face_opencv (not implemented yet in the function but I'll add it)
        center_x = detect_face_opencv(video_file, limit_seconds)
        print(json.dumps({"center_x": center_x}))
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)
