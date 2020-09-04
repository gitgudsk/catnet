import numpy as np
import cv2

"""
from tensorflow.keras.models import load_model

net = load_model("catnet.h5")
"""

def main():

    cap = cv2.VideoCapture("VID.mp4")

    while True:
        _, frame = cap.read()

        

main()
