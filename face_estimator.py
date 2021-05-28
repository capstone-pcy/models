import mediapipe as mp
import cv2
import os
import pickle
import numpy as np
import pandas as pd
import tqdm
from mediapipe.python.solutions import holistic

model_dir = "trained_model/ml/"
model_path = os.path.join(model_dir, 'face_estimator.pkl')

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

with open(model_path, 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()

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

        if cv2.waitKey(10) & 0xFF == ord('q') : break
    
cap.release()
cv2.destroyAllWindows()