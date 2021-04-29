import os

file_list = os.listdir('./data/video')

print(file_list)

for file_ in file_list:
    print(file_)
    print(os.path.splitext(file_)[0])