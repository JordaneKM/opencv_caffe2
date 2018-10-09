
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import deep_learning_caffe2


COLORS = np.random.uniform(0, 255, (1000, 3))

print ("STARTING VIDEO....")
vs = VideoStream(0)
time.sleep(2)
fps = FPS().start()

while True:

    frame = vs.read()
    # deep_learning_caffe2.IMAGE_LOCATION = "vending_machine.png"
    image_1 = deep_learning_caffe2.format_image(frame)
    (h,w) = frame.shape[:2]
    new_image = cv2.dnn.blobfromImage(cv2.resize(frame, (300,300)),
    0.007843, (300,300) 127.5)

    net.setInput(new_image)
    detections = net.forward()
    detect_conf,detect_pred,detect_five = deep_learning_caffe2.make_pred(image_1)
