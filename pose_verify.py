import cv2
import os
import math
import json
import mediapipe as mp

input_dir = "data/video"
output_dir = "data/output_video/pose_estimate"
log_file_dir = "data/log/pose_log/"

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    file_list = os.listdir(input_dir)

    for file_ in file_list:
        
        file_name = os.path.splitext(file_)[0]

        cap = cv2.VideoCapture(os.path.join(input_dir, file_))

        FPS = cap.get(cv2.CAP_PROP_FPS)
        delay = round(1000/FPS)
        frame_second = 0

        os.makedirs(log_file_dir + file_name, exist_ok=True)
        log_file_path = os.path.join(log_file_dir, file_name + '/')
        face_log_name = file_name + '-face_verify_log.json'
        rhand_log_name = file_name + '-rhand_verify_log.json'
        lhand_log_name = file_name + '-lhand_verify_log.json'
        face_log_path = os.path.join(log_file_path, face_log_name)
        rhand_log_path = os.path.join(log_file_path, rhand_log_name)
        lhand_log_path = os.path.join(log_file_path, lhand_log_name)


        make_dir = os.path.join(output_dir, file_name)
        os.makedirs(make_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, file_name + '/')
        output_file_name = 'output_' + file_name + '.avi'
        output_file_path  = os.path.join(output_file_path, output_file_name)

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_file_path, fourcc, FPS, (width, height), True)

        right_hand_logs = dict()
        left_hand_logs = dict()
        face_logs = dict()

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            while True:
                ret, frame = cap.read()

                if not ret : break

                frame_second += (1 / FPS)

                # Recolor feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
                # make Detection
                results = holistic.process(image)

                # face_landmarks, pose_landmarks, left_handmarks, right_hand_landmarks

                # Recolor image bock to BGR for rendering
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw face landmarks
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
                    face_logs[frame_second] = (0)
                else:
                    face_logs[frame_second] = (1)

        
                # Draw right hand landmarks
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    right_hand_logs[frame_second] = (0)
                else:
                    right_hand_logs[frame_second] = (1)
        

                # Draw left hand landmarks
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    left_hand_logs[frame_second] = (0)
                else:
                    left_hand_logs[frame_second] = (1)
        
        
                cv2.imshow("image", image)
                writer.write(image)

                if cv2.waitKey(delay) & 0xFF == ord('q') : break

        cap.release()
        cv2.destroyAllWindows()

        sec_wise_face = dict()
        sec_wise_rhand = dict()
        sec_wise_lhand = dict()

        cur_sec = 0
        avg = 0

        for sec, value in face_logs.items():
            if cur_sec < math.floor(sec):
                sec_wise_face[cur_sec] = round(avg/FPS)
                cur_sec = math.floor(sec)
                avg = 0
            else:
                avg += value
        
        cur_sec = 0

        for sec, value in right_hand_logs.items():
            if cur_sec < math.floor(sec):
                sec_wise_rhand[cur_sec] = round(avg/FPS)
                cur_sec = math.floor(sec)
                avg = 0
            else:
                avg += value
         
        cur_sec = 0

        for sec, value in left_hand_logs.items():
            if cur_sec < math.floor(sec):
                sec_wise_lhand[cur_sec] = round(avg/FPS)
                cur_sec = math.floor(sec)
                avg = 0
            else:
                avg += value
        

        with open(face_log_path, 'w') as outfile:
            json.dump(sec_wise_face, outfile)
        
        with open(rhand_log_path, 'w') as outfile:
            json.dump(sec_wise_rhand, outfile)
        
        with open(lhand_log_path, 'w') as outfile:
            json.dump(sec_wise_lhand, outfile)
        


if __name__ == '__main__':
    main()