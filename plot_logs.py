import json
import os
import matplotlib.pyplot as plt

detect_file_dir = "data/log/detect_log/"
pose_file_dir = "data/log/pose_log/"

file_list = os.listdir(pose_file_dir)

for file_name in file_list:

    json_name = file_name + '-hand_log.json'

    json_file_path = os.path.join(pose_file_dir, file_name + '/', json_name)

    with open(json_file_path) as json_file:
        json_data = json.load(json_file)

        time_list = list(json_data.keys())
        val_list = list(json_data.values())

        plt.plot(time_list, val_list)
        plt.show()