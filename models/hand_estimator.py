import cv2
import mediapipe as mp
import os
import math
import json
from tqdm import tqdm

def handEstimator():

    input_dir = "data/video/"
    output_dir = "data/output_video/hand_estimate/"
    log_file_dir = "data/log/hand_log/"

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

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
        hand_log_path = os.path.join(log_file_path, file_name + '-hand_log.json')

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
        hand_logs = dict()

        while True:
            # read Video's frame
            ret, img = cap.read()

            # break if when end of Video
            if not ret : break

            frame_second += (1 / FPS)

            # detect hands
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            # draw & make logs of hand detection
            if results.multi_hand_landmarks:
                if len(results.multi_hand_landmarks) == 2:
                    hand_logs[frame_second] = (0)
                else:
                    hand_logs[frame_second] = (1)

                for handLms in results.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
            cv2.imshow("Results", img)
            writer.write(img)

            if cv2.waitKey(delay) & 0xFF == ord('q') : break
    
        cap.release()
        cv2.destroyAllWindows()

        # create dictionary for second wise json log file
        hand_log = dict()

        cur_sec = 0
        avg = 0
        cnt = 0

        for sec, val in hand_logs.items():
            if cur_sec < math.floor(sec):
                cur_sec = math.floor(sec)
                avg = 0
                cnt = 0
            
            avg += val
            cnt += 1
            hand_log[cur_sec] = avg / cnt
        
        with open(hand_log_path, 'w') as outfile:
            json.dump(hand_log, outfile)


if __name__ == "__main__":
    handEstimator()