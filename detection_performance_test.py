import cv2
import numpy as np
import os

from .calc_mAP import utils

model_path = "trained_model/"
val_imgs_dir = "data/val_imgs/"

weight_path = os.path.join(model_path + 'yolov3_last.weights')
cfg_path = os.path.join(model_path + 'yolov3.cfg')
names_path = os.path.join(model_path + 'obj.names')

net = cv2.dnn.readNet(weight_path, cfg_path)
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


file_list = os.listdir(val_imgs_dir)

for file_ in file_list:

    file_name = os.path.splitext(file_)[0]

    # Loading image
    img = cv2.imread(val_imgs_dir + file_)

    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)

    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.nanargmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                

                # 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
    
    cv2.imshow("image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
