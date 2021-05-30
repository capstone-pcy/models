import mediapipe as mp
import cv2
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import json

from .utils import FACE_MODEL_DIR, FACE_OUTPUT_DIR, FACE_LOG_DIR

def liveFaceEstimator(user_name : str):

    model_dir = FACE_MODEL_DIR
    model_path = os.path.join(model_dir, 'face_estimator.pkl')

    output_dir = FACE_OUTPUT_DIR
    log_file_dir = FACE_LOG_DIR

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_face_detection = mp.solutions.face_detection

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    cap = cv2.VideoCapture(0)

    # get video's properties (FPS, delay)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    delay = round(1000 / FPS)
    frame_second = 0

    # make directory for log files
    os.makedirs(log_file_dir + user_name, exist_ok=True)
    log_file_path = os.path.join(log_file_dir, user_name + '/')
    face_log_path = os.path.join(log_file_path, user_name + '-face_log.json')
    multiFace_log_path = os.path.join(log_file_path, user_name + '-multiFace_log.json')

    # make directory for output videos
    os.makedirs(output_dir + user_name, exist_ok=True)
    out_video_path = os.path.join(output_dir, user_name + '/')
    out_video_path = os.path.join(out_video_path, user_name + '.avi')

    # define video writer
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_video_path, fourcc, FPS, (width, height), True)

    face_logs = dict()
    multiFace_logs = dict()

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret : break

                frame_second += (1 / FPS)

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                detection_results = face_detection.process(image)
                holistic_results = holistic.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw face detection results
                if detection_results.detections:
                    if len(detection_results.detections) == 1:
                        multiFace_logs[frame_second] = (0)
                    else : 
                        multiFace_logs[frame_second] = (1)
                    
                    for detection in detection_results.detections:
                        mp_drawing.draw_detection(image, detection)
                else:
                    multiFace_logs[frame_second] = (1)
                
                # Draw face landmarks
                mp_drawing.draw_landmarks(image, holistic_results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(thickness=1, circle_radius=1))

                # Draw pose landmarks
                mp_drawing.draw_landmarks(image, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=1),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = holistic_results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    
                    # Extract Face landmarks
                    face = holistic_results.face_landmarks.landmark
                    face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                    # Concate rows
                    row = pose_row + face_row

                    # Make Detectoins
                    X = pd.DataFrame([row])
                    face_estimate_class = model.predict(X)[0]
                    face_estimate_prob = model.predict_proba(X)[0]
                    
                    face_logs[frame_second] = (face_estimate_prob[0])
                    
                    # Get status bos
                    cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
                    
                    # Display Class
                    cv2.putText(image, 'CLASS', \
                        (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, face_estimate_class.split(' ')[0], \
                        (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Display Probability
                    cv2.putText(image, 'PROB', \
                        (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(face_estimate_prob[np.argmax(face_estimate_prob)], 2)), \
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                except : pass
                
                cv2.imshow("Results", image)
                writer.write(image)

                if cv2.waitKey(delay) & 0xFF == ord('q') : break
    
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    cur_sec = 0
    avg = 0
    cnt = 0

    multiFace_log = dict()

    for sec, val in multiFace_logs.items():
        if cur_sec < math.floor(sec):
            cur_sec = math.floor(sec)
            avg = 0
            cnt = 0
        
        avg += val
        cnt += 1
        multiFace_log[cur_sec] = avg / cnt
    
    with open(multiFace_log_path, 'w') as outfile:
        json.dump(multiFace_log, outfile)
    
    cur_sec = 0
    max_conf = 0

    face_log = dict()

    for sec, val in face_logs.items():
        if cur_sec < math.floor(sec):
            cur_sec = math.floor(sec)
            max_conf = 0
        
        if max_conf <= val:
            max_conf = val
            face_log[cur_sec] = val
    
    with open(face_log_path, 'w') as outfile:
        json.dump(face_log, outfile)