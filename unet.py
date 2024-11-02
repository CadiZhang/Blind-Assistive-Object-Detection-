import cv2
import torch
import segmentation_models_pytorch as smp

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the UNet model with pretrained weights
model = smp.Unet(
    encoder_name="resnet34",  # Choose encoder, e.g. "resnet34", "resnet50", etc.
    encoder_weights="imagenet",  # Load weights pre-trained on ImageNet
    in_channels=3,  # Input channels (RGB)
    classes=2  # Output channels (number of classes)
).to(device)

model.eval()

# Webcam setup
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Store original frame dimensions
    original_height, original_width = frame.shape[:2]

    # Preprocess the frame
    input_frame = cv2.resize(frame, (256, 256))
    input_tensor = torch.from_numpy(input_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.argmax(output, dim=1).squeeze().numpy()

    # Post-process to display segmentation mask on the frame
    mask = (output * 255).astype('uint8')
    # Resize mask back to original frame size
    mask = cv2.resize(mask, (original_width, original_height))
    color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)

    # Display the output
    cv2.imshow("Segmentation", overlay)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()