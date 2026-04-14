import cv2

def print_high_risk_moments(valid_preds, threshold):
    """Prints the timestamps where the risk exceeds the defined threshold."""
    print(f"\n--- Moments exceeding {threshold}% risk threshold ---")
    threshold_float = threshold / 100.0
    high_risk_found = False

    for i, prob in valid_preds:
        if prob > threshold_float:
            timestamp_seconds = i * 0.125
            minutes = int(timestamp_seconds // 60)
            seconds = int(timestamp_seconds % 60)
            print(f"Risk: {prob*100:.1f}% detected at {minutes:02d}:{seconds:02d} ({timestamp_seconds:.1f}s)")
            high_risk_found = True

    if not high_risk_found:
        print("No moments exceeded the threshold.")

def save_top_k_frames(video_path, sorted_preds, k=10):
    """Extracts and saves the top k frames with the highest risk scores."""
    top_k = sorted_preds[:k]
    if not top_k:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\nError: Unable to open video {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n--- Saving Top {k} risks ---")

    for i, prob in top_k:
        timestamp_seconds = i * 0.125
        minutes = int(timestamp_seconds // 60)
        seconds = int(timestamp_seconds % 60)

        frame_number = int(timestamp_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()

        if ret:
            filename = f"risk_{prob:.2f}_{minutes:02d}-{seconds:02d}.png"
            cv2.imwrite(filename, frame)
            print(f"{filename} saved ({timestamp_seconds:.1f}s)")

    cap.release()