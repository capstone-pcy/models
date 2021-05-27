import cv2
import os
import random
import numpy as np
from tqdm import tqdm

def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img

def vertical_flip(img, flag):
    if flag:
        return cv2.flip(img, 0)
    else:
        return img

def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

img_dir = os.path.join("./data/augmentation_data/")
out_dir = os.path.join("./data/augmentation_results/")
out_hori_dir = os.path.join(out_dir, 'horizontal_flip/')
out_verti_dir = os.path.join(out_dir, 'vertical_flip/')
out_bright_dir = os.path.join(out_dir, 'bright/')

file_list = os.listdir(img_dir)

for file_ in tqdm(file_list):

    file_name = os.path.splitext(file_)[0]

    img = cv2.imread(os.path.join(img_dir, file_))

    hori_flip = horizontal_flip(img, True)
    cv2.imwrite(os.path.join(out_hori_dir, file_name + '-hori.jpg'), hori_flip)

    verti_flip = vertical_flip(img, True)
    cv2.imwrite(os.path.join(out_verti_dir, file_name + '-verti.jpg'), verti_flip)

    bright = brightness(img, 0.5, 3)
    cv2.imwrite(os.path.join(out_bright_dir, file_name + '-bright.jpg'), bright)