import mediapipe as mp
import cv2
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from mediapipe.python.solutions import holistic
import math
import json

model_dir = "trained_model/ml/"
model_path = os.path.join(model_dir, 'face_estimator.pkl')

input_dir = "data/video/"
output_dir = "data/output_video/face_estimate/"
log_file_dir = "data/log/face_log/"

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def main():

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    file_list = os.listdir(input_dir)

    for file_ in tqdm(file_list):

        file_name = os.path.splitext(file_)[0]

        cap = cv2.VideoCapture(os.path.join(input_dir, file_))

        # get video's propertys (FPS, delay)
        FPS = cap.get(cv2.CAP_PROP_FPS)
        delay = round(1000/FPS)
        frame_second = 0

        # make log file's directory
        os.makedirs(log_file_dir + file_name, exist_ok=True)
        log_file_path = os.path.join(log_file_dir, file_name + '/')
        face_log_path = os.path.join(log_file_path, file_name + '-face_log.json')

        # make output video's directory
        os.makedirs(output_dir + file_name, exist_ok=True)
        out_video_path = os.path.join(output_dir, file_name + '/')
        out_video_path = os.path.join(out_video_path, file_name + '.avi')

        # define video writer
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_video_path, fourcc, FPS, (width, height), True)

        # create dictoinary for json log file
        face_logs = dict()

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            while True:
                ret, frame = cap.read()

                if not ret : break
                
                frame_second += (1 / FPS)

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = holistic.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw face landmarks
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(thickness=1, circle_radius=1))

                # Draw pose landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=1),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    
                    # Extract Face landmarks
                    face = results.face_landmarks.landmark
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

                if cv2.waitKey(10) & 0xFF == ord('q') : break
            
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        cur_sec = 0
        max_conf = 0

        face_log = dict()

        for sec, val in face_logs.items():
            if cur_sec < math.floor(sec):
                cur_sec = math.floor(sec)
                max_conf = 0
            
            if max_conf < val:
                max_conf = val
                face_log[cur_sec] = val
        
        with open(face_log_path, 'w') as outfile:
            json.dump(face_log, outfile)
        

if __name__ == "__main__":
    main()