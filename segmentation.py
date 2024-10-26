from fastsam import FastSAM, FastSAMPrompt  # Change from ultralytics.FastSAM to fastsam
import torch
import numpy as np
import cv2
import time

# Load the FastSAM model
model = FastSAM('FastSAM-x.pt')  # Make sure this path points to where you saved the model

# Set the device to CUDA if available, otherwise CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", DEVICE)

# Initialize the default camera (0 = default camera)
cap = cv2.VideoCapture(0)

# Main loop for live segmentation
while cap.isOpened():
    
    # Capture frame-by-frame
    success, frame = cap.read()
    if not success:
        print("Failed to capture image")
        break
    
    # Start time for FPS calculation
    start_time = time.perf_counter()

    # Run the segmentation model on the frame
    everything_results = model(
        source=frame,
        device=DEVICE,
        retina_masks=True,
        imgsz=1024,       # Resolution for model input
        conf=0.4,         # Confidence threshold
        iou=0.9           # IoU threshold
    )
    
    # Get the masks
    masks = everything_results[0].masks.data
    
    # Create a blank overlay for all masks
    overlay = frame.copy()
    
    # Draw each mask on the frame
    if masks is not None:
        for mask in masks:
            # Convert mask to numpy array
            mask = mask.cpu().numpy()
            # Resize mask to frame size
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            # Create colored overlay for this mask
            color_mask = np.zeros_like(frame)
            color_mask[mask > 0.5] = [0, 255, 0]  # Green color
            overlay = cv2.addWeighted(overlay, 1, color_mask, 0.5, 0)
    
    # Add FPS counter
    fps = 1 / (time.perf_counter() - start_time)
    cv2.putText(overlay, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show the frame with masks
    cv2.imshow('FastSAM Segmentation', overlay)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
