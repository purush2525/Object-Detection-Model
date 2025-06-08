Object Detection using YOLOv5

Project Overview

This project uses the YOLOv5 object detection model to detect objects in real-time webcam feed. The model is loaded using the Ultralytics library, and the OpenCV library is used for webcam feed capture and display.

Requirements

- Python 3.11.5
- Ultralytics (YOLOv5)
- OpenCV
- PyTorch

Installation

To install the required libraries, run the following command:


bash
pip install ultralytics opencv-python torch


Code Explanation

Webcam Feed Capture
The code starts by capturing the webcam feed using OpenCV's VideoCapture function. The feed is set to a resolution of 640x480 pixels.


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


Loading YOLOv5 Model
The YOLOv5 model is loaded using the Ultralytics library. The torch.hub.load function is used to load the model, and the force_reload=True argument ensures that the model is reloaded if it has been modified.


model = torch.hub.load('ultralytics/yolov5', 'yolov5m', force_reload=True)


Object Detection
The code then enters a loop where it reads frames from the webcam feed and performs object detection using the YOLOv5 model. The model(img) function returns the detection results, which include the bounding box coordinates, confidence scores, and class labels.


while True:
    success, img = cap.read()
    if not success:
        break
    results = model(img)


Drawing Bounding Boxes and Class Labels
The code then iterates through the detection results and draws bounding boxes around the detected objects. The class labels are also displayed above the bounding boxes.


for r in results.xyxy[0]:
    x1, y1, x2, y2, conf, cls = r
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    label = classNames[int(cls)]
    org = (int(x1), int(y1) - 10 if int(y1) - 10 > 10 else int(y1) + 10)
    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


Displaying the Output
Finally, the code displays the output using OpenCV's imshow function. The loop continues until the 'q' key is pressed.


cv2.imshow('Cam', img)
if cv2.waitKey(1) & 0xFF == ord('q'):
    break


Example Use Cases

- Real-time object detection in surveillance systems
- Autonomous vehicles
- Robotics
- Quality control in manufacturing

