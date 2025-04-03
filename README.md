# CV-Project-Msc-Sem_2

**Car Make and Model classification example with YOLOv3 object detector**

**Object Detection and Classification in images**

This example takes an image as input, detects the cars using YOLOv3 object detector, crops the car images, makes them square while keeping the aspect ratio, resizes them to the input size of the classifier, and recognizes the make and model of each car. The result is shown on the display and saved as output.jpg image file.

    $ python car_make_model_classifier_yolo3.py --image cars.jpg

**Object Detection and Classification in video files**

This example takes a video file as input, detects the cars in each frame using YOLOv3 object detector, crops the car images, makes them square while keeping the aspect ratio, resizes them to the input size of the classifier, and recognizes the make and model of each car. The result is saved as an output video file.

    $ python car_make_model_classifier_yolo3_video.py --input video.avi --output output.avi

**Requirements**

    python
    numpy
    tensorflow
    opencv
    yolov3.weights must be downloaded from https://pjreddie.com/media/files/yolov3.weights and saved in folder yolo-coco

**Configuration**
The settings are stored in python file named config.py:

    model_file = "model-weights-spectrico-mmr-mobilenet-64x64-344FF72B.pb"
    label_file = "labels.txt"
    input_layer = "input_1"
    output_layer = "softmax/Softmax"
    classifier_input_size = (64, 64)

**model_file is the path to the car make and model classifier classifier_input_size is the input size of the classifier label_file is the path to the text file, containing a list with the supported makes and models**

**The make and model classifier is based on MobileNetV2 mobile architecture:** https://arxiv.org/abs/1801.04381

**INPUT IMAGE**

![cars](https://github.com/user-attachments/assets/501ed75b-6272-4afc-b2bc-357f78779a8f)

**OUTPUT IMAGE**

![output](https://github.com/user-attachments/assets/d57205ef-ee47-4c2e-ae99-9a6971eb0d24)
