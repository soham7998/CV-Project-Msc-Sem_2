import numpy as np
import argparse
import time
import cv2
import os
import classifier
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-y", "--yolo", default='yolo-coco', help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

car_color_classifier = classifier.Classifier()

# Load COCO class labels for YOLO
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Ensure the car class index is correct (COCO class index for car is 2)
CAR_CLASS_ID = LABELS.index("car") if "car" in LABELS else 2  

# Define colors for visualization
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Load YOLO model
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
print("[INFO] Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Load the input image
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# Get YOLO output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Convert image to a blob and perform forward pass
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
outputs = net.forward(output_layers)
end = time.time()
print(f"[INFO] YOLO took {end - start:.6f} seconds")

# Initialize lists
boxes, confidences, classIDs = [], [], []

# Process each detection
for output in outputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        if confidence > args["confidence"]:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# Apply non-maxima suppression
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

# Check if any objects were detected
if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y, w, h) = boxes[i]
        color = [int(c) for c in COLORS[classIDs[i]]]

        # Check if detected object is a car
        if classIDs[i] == CAR_CLASS_ID:
            print("[INFO] Car detected, running classifier...")
            start = time.time()
            result = car_color_classifier.predict(image, x, y, w, h)  # Fixed function call
            end = time.time()
            print(f"[INFO] Classifier took {end - start:.6f} seconds")

            if result and len(result) > 0:
                text = "{}: {:.4f}".format(result[0]['make'], float(result[0]['prob']))
                cv2.putText(image, text, (x + 2, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(image, result[0]['model'], (x + 2, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                print("Warning: No valid predictions from classifier.")

        # Draw bounding boxes and labels for all detected objects
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        label_text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show and save results
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', W, H)
cv2.imshow("Image", image)
cv2.imwrite("output.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
