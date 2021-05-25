import cv2
import os
import mediapipe as mp

input_dir = "data/video"
output_dir = "data/output_video/pose_estimate"
log_file_dir = "data/log/pose_log"

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    file_list = os.listdir(input_dir)

    for file_ in file_list:
        
        file_name = os.path.splitext(file_)[0]

        cap = cv2.VideoCapture(os.path.join(input_dir, file_))

        FPS = cap.get(cv2.CAP_PROP_FPS)
        delay = round(1000/FPS)

        make_dir = os.path.join(output_dir, file_name)
        os.makedirs(make_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, file_name + '/')
        output_file_name = 'output_' + file_name + '.avi'
        output_file_path  = os.path.join(output_file_path, output_file_name)

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_file_path, fourcc, FPS, (width, height), True)

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            while True:
                ret, frame = cap.read()

                if not ret : break

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
                else:
                    print("No face!")

        
                # Draw right hand landmarks
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                else:
                    print("NO right hand!")
        

                # Draw left hand landmarks
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                else:
                    print("NO left hand!")
        
        
                cv2.imshow("image", image)
                writer.write(image)

                if cv2.waitKey(10) & 0xFF == ord('q') : break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()