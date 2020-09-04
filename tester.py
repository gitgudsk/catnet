import numpy as np
import cv2

"""
from tensorflow.keras.models import load_model

net = load_model("catnet.h5")
"""

def create_pyramid(frame):
    """
    Createa an image pyramid out of frame
    :param frame: the original image
    :return pyramid: list of downsampled images after the original image
    """

    MIN_H = 24
    MIN_W = 24

    pyramid = []

    # create downsampled images until the minimum size requirement is met
    while True:
        pyramid.append(frame)
        if frame.shape[0] < MIN_H or frame.shape[1] < MIN_W:
            break
        frame = cv2.pyrDown(frame)
        print(frame.shape)

    return pyramid


def main():

    cap = cv2.VideoCapture("VID.mp4")

    while True:
        _, frame = cap.read()

        pyramid = create_pyramid(frame)

        for i, img in enumerate(pyramid):
            cv2.imshow("Feed" + str(i), img)

        k = cv2.waitKey(50) & 0xFF
        # ESC
        if k == 27:
            break


    cv2.destroyAllWindows()

        

main()
