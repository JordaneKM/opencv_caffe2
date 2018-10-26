import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import time


ap = argparse.ArgumentParser()
ap.add_argument("-v","--video", type=str,help="path to input video file")

ap.add_argument("-t","--tracker", type=str, default="csrt",
                help="OpenCV object tracker type")

args = vars(ap.parse_args())

(major, minor) = cv2.__version__.split(".")[:2]

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf" : cv2.TrackerKCF_create,
    "boosting" : cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse" : cv2.TrackerMOSSE_create
}


tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

initBB = None

video_capture = cv2.VideoCapture(0)

#cv2.namedWindow("Frame")

while True:
    ret, frame = video_capture.read()

    frame = imutils.resize(frame,width=500)
    (H,W) = frame.shape[:2]

    if initBB is not None:
        (success,box) = tracker.update(frame)

        if success:
            (x,y,w,h) = [int(v) for v in box]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        fps.update()
        fps.stop()


        info = [
                ("Tracker", args ["tracker"]),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
               ]

        for (i, (k,v)) in enumerate(info):
            text = "{}: {}".format(k,v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                 showCrosshair=True)
                 
        tracker.init(frame, initBB)
        fps = FPS().start()


    #This breaks on 'q' key
    elif key == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
