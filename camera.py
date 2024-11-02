import cv2


# Replace with your ESP32-CAM's IP address
url = "http://10.230.150.104:81/stream"  # Stream URL from the ESP32


# Open a connection to the stream
cap = cv2.VideoCapture(url)


if not cap.isOpened():
   print("Error: Could not open video stream.")
else:
   print("Connected to video stream. Press 'q' to quit.")


# Display the video stream frame by frame
while True:
   ret, frame = cap.read()
   if not ret:
       print("Error: Could not retrieve frame.")
       break
  
   # Show the frame in a window
   cv2.imshow("ESP32 Camera Stream", frame)


   # Press 'q' to quit the video stream display
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break


# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
