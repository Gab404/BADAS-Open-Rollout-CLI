import cv2
import argparse
import torch
import numpy as np

from badas import BADASModel
from export import export_video_with_gauge
from _utils import filter_and_sort_predictions

def parse_arguments():
    """Parses and validates command line arguments."""
    parser = argparse.ArgumentParser(description="BADAS inference with flexible options")
    parser.add_argument('--video-path', type=str, required=True,
                        help="Path to the video file (e.g., test.mp4)")
    parser.add_argument('--real-time', action='store_true',
                        help="Enables video and gauge display during inference (requires GUI)")
    parser.add_argument('--time', action='store_true',
                        help="Prints inference time for each 16-frame window")
    parser.add_argument('--threshold', type=int, default=90,
                        help="Risk threshold percentage to print timestamps (0-99). Default: 90")
    parser.add_argument('--export-video', type=str, metavar='OUTPUT_PATH',
                        help="Saves a new video with the risk gauge overlay (e.g., output.mp4)")
    
    heatmap_group = parser.add_mutually_exclusive_group()
    heatmap_group.add_argument('--grad-rollout', action='store_true',
                               help="Display the Grad-Rollout (Attention x Gradient) heatmap on the output video.")
    heatmap_group.add_argument('--attn-rollout', action='store_true',
                               help="Display the pure Attention Rollout heatmap on the output video.")
    
    args = parser.parse_args()

    # Validate threshold
    if args.threshold < 0 or args.threshold > 99:
        parser.error("The --threshold argument must be between 0 and 99.")
        
    return args

def main():
    """Main execution function."""
    args = parse_arguments()

    # Initialize model
    model = BADASModel(device="cuda")
    video_path = args.video_path

    print("\n--- Starting analysis ---")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Device: {gpu_name}")
    else:
        print("Device: CPU")
        
    print(f"Video: {video_path}")
    print(f"Real-time mode: {'ENABLED' if args.real_time else 'DISABLED'}")
    
    # Affichage clair du mode sélectionné
    if args.grad_rollout:
        print("Heatmap mode: GRAD-ROLLOUT")
    elif args.attn_rollout:
        print("Heatmap mode: ATTENTION ROLLOUT")
    else:
        print("Heatmap mode: DISABLED")
        
    # print(f"Alert threshold: {args.threshold}%")

    print("\n")

    # Run Prediction (on passe les deux nouveaux arguments)
    try:
        predictions = model.predict(
            video_path, 
            real_time=args.real_time, 
            show_time=args.time, 
            grad_rollout=args.grad_rollout, 
            attn_rollout=args.attn_rollout
        )
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # Clean and Sort Data
    valid_preds, sorted_preds = filter_and_sort_predictions(predictions)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not valid_preds:
        print("Error: The model returned only 'nan' values.")
        return

    # Execute outputs based on arguments
    vjepa_instance = getattr(model, 'model', None)

    if args.export_video:
        attn_history = getattr(vjepa_instance, 'attention_history', [])
        # On consolide les deux flags en un seul booléen pour l'export OpenCV
        heatmap_is_on = args.grad_rollout or args.attn_rollout
        export_video_with_gauge(video_path, args.export_video, predictions, attn_history, heatmap_is_on)

    if args.time:
        times = getattr(vjepa_instance, 'times', [])
        np_times = np.array(times)
        if args.grad_rollout or args.attn_rollout:
            print("\nWarning: Heatmap generation increases the computation time.")
        print(f"\nAverage runtime: {np_times.mean():.4f}s / 16 frames")

    print("\n--- Analysis complete ---")

if __name__ == "__main__":
    main()