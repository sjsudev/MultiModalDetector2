import argparse
import glob
import os
import time
import cv2
from MyDetector import Yolov5Detector
from Myutils import detectimage
import sys
sys.path.append('./MyDetector/ultralyticsyolov5')

class Yolov5detectorargs:
    modelname = 'yolov5'#not used here
    modelbasefolder = './ModelOutput/yolov5/'
    modelfilename='best.pt'#'myyolov5s_resave.pt'#'yolov5l.pt' #not used
    device='cuda' #'cuda device, i.e. 0 or 0,1,2,3 or cpu'
    #showfig='True'
    threshold = 0.3

def testYolov5Detector(detectorargs):
    mydetector = Yolov5Detector.MyYolov5Detector(detectorargs)
    #mydetector = Yolov5Detector.ultralyticsYolov5Detector(detectorargs)
    imgpath=os.path.join('testdata/', "traffic1.jpg")
    print(imgpath)
    bbox_xyxy, pred_labels, cls_conf=detectimage.detectoneimage_novis(imgpath, mydetector)
    print(pred_labels)

if __name__ == "__main__":
    #Test TF2
    #testTF2Detector(TF2detectorargs)

    #Test Detectron2
    #testDetectron2Detector(Detectron2detectorargs)

    #Test TorchVision
    #testTorchVisionDetector(TorchVisiondetectorargs)

    #Test Yolov5
    testYolov5Detector(Yolov5detectorargs)

    #detect.testdetector()
    #testYolov3Detector(Yolov3detectorargs)
