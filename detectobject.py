from ultralytics import YOLO
import torch
import cv2

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load the YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', force_reload=True)

# List of class names
classNames = model.names

while True:
    # Read a frame from the camera
    success, img = cap.read()
    if not success:
        break

    # Perform object detection using YOLO model
    results = model(img)

    # Iterate through the result of object detection
    for r in results.xyxy[0]:
        # Extract coordinates of the bounding box
        x1, y1, x2, y2, conf, cls = r

        # Draw rectangle around detected object
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Prepare text label
        label = classNames[int(cls)]
        org = (int(x1), int(y1) - 10 if int(y1) - 10 > 10 else int(y1) + 10)

        # Draw class name
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 2
        cv2.putText(img, label, org, font, fontScale, color, thickness)

    # Display the frame with detected objects
    cv2.imshow('Cam', img)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()