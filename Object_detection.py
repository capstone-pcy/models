import cv2
import numpy as np
import json

net = cv2.dnn.readNet("./trained_model/yolov3_final.weights", "./trained_model/yolov3.cfg")
with open('./trained_model/obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# Loading Web-cam
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Loading Video
cap = cv2.VideoCapture('./data/video/ice_shot.mp4')


detect_log = {}
prev_time = 0
FPS = cap.get(cv2.CAP_PROP_FPS)
delay = round(1000/FPS)
frame_second = 0

# 코덱 지정, *는 문자를 풀어쓰는 방식 *'DIVX' == 'D', 'I', 'V', 'X'
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter('./data/output_video/output.avi', fourcc, FPS, (width, height), True)

while True:
    ret, frame = cap.read()

    frame_second += (1/FPS)

    if not ret:
        break

    # 이 구문 없으면 output.avi 재생 가능!
    # frame = cv2.resize(frame, None, fx=0.4, fy=0.4)

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

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
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)
            detect_log[frame_second] = (label, confidences[i])

    cv2.imshow('frame', frame)
    writer.write(frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'): break

with open('./data/detect_log/detect_log.json', 'w') as outfile:
    json.dump(detect_log, outfile)

cap.release()
writer.release()
cv2.destroyAllWindows()
