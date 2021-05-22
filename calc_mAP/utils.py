import os
import cv2
import numpy as np

# YOLO coordinate -> VOC coordinate
def convertToAbsoluteValues(size, box):

    xIn = round(((2 * float(box[0]) - float(box[2])) * size[0] / 2))
    yIn = round(((2 * float(box[1]) - float(box[3])) * size[1] / 2))
    xEnd = xIn + round(float(box[2]) * size[0])
    yEnd = yIn + round(float(box[3]) * size[1])

    if xIn < 0:
        xIn = 0
    if yIn < 0:
        yIn = 0
    if xEnd >= size[0]:
        xEnd = size[0] - 1
    if yEnd >= size[1]:
        yEnd = size[1] - 1
    
    return (xIn, yIn, xEnd, yEnd)

# Add Bounding box information
def boundingBoxes(labelPath, imagePath):

    detections, groundtruths, classes = [], [], []

    for boxtype in os.listdir(labelPath):

        boxtypeDir = os.path.join(labelPath, boxtype)

        for labelfile in os.listdir(boxtypeDir):
            filename = os.path.splitext(labelfile)[0]
            with open(os.path.join(boxtypeDir, labelfile)) as f:
                labelinfos = f.readlines()
            
            imgfilepath = os.path.join(imagePath, filename + ".jpg")
            img = cv2.imread(imgfilepath)
            h, w, _ = img.shape

            for labelinfo in labelinfos:
                label, conf, rx1, ry1, rx2, ry2 = map(float, labelinfo.strip().split())
                x1, y1, x2, y2 = convertToAbsoluteValues((w, h), (rx1, ry1, rx2, ry2))
                boxinfo = [filename, label, conf, (x1, y1, x2, y2)]

                if label not in classes:
                    classes.append(label)
                
                if boxtype == "detection":
                    detections.append(boxinfo)
                else:
                    groundtruths.append(boxinfo)
        
    classes = sorted(classes)

    return detections, groundtruths, classes

# Calculate IoU(intersection over Union)
def getArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

def getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])

    return (xB - xA + 1) * (yB - yA + 1)

def getUnionAreas(boxA, boxB, interArea=None):
    area_A = getArea(boxA)
    area_B = getArea(boxB)

    if interArea is None:
        interArea = getIntersectionArea(boxA, boxB)
    
    return float(area_A + area_B - interArea)

def boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False
    if boxB[0] > boxA[2]:
        return False
    if boxA[3] < boxB[1]:
        return False
    if boxB[3] < boxA[1]:
        return False
    
    return True

def iou(boxA, boxB):

    if boxesIntersect(boxA, boxB):
        return 0
    
    interArea = getIntersectionArea(boxA, boxB)
    union = getUnionAreas(boxA, boxB, interArea=interArea)

    result = interArea / union
    assert result >= 0 , \
        "iou must be positive!"
    return result