import numpy as np
import cv2

"""
from tensorflow.keras.models import load_model

net = load_model("catnet.h5")
"""

def create_pyramid(frame):
    """
    Create an image pyramid out of a frame
    :param frame: the original image
    :return pyramid: list of downsampled images after the original image
    """

    MIN_H = 80
    MIN_W = 80

    pyramid = []

    # create downsampled images until the minimum size requirement is met
    while True:
        pyramid.append(frame)
        if frame.shape[0] < MIN_H or frame.shape[1] < MIN_W:
            break
        frame = cv2.pyrDown(frame)
        #print(frame.shape)

    return pyramid


def main():

    WIN_H = 300
    WIN_W = 300

    cap = cv2.VideoCapture("VID.mp4")

    while True:
        _, frame = cap.read()

        pyramid = create_pyramid(frame)

        for i, img in enumerate(pyramid):

            h = img.shape[0]
            w = img.shape[1]

            h_iterations = np.int32(np.ceil(h / WIN_H))
            w_iterations = np.int32(np.ceil(w / WIN_W))

            for hi in range(0, h_iterations):
                for wi in range(0, w_iterations):

                    h_start = hi * WIN_H
                    w_start = wi * WIN_W

                    h_end = h_start + WIN_H
                    if h_end >= h:
                        h_end = h

                    w_end = w_start + WIN_W
                    if w_end >= w:
                        w_end = w

                    cv2.rectangle(img, (w_start, h_start), (w_end, h_end), (0, 255, 0), 2)

                    cv2.imshow("Feed" + str(i), img)
                    cv2.waitKey(3) & 0xFF


        """
        k = cv2.waitKey(1) & 0xFF
        # ESC
        if k == 27:
            break

        """
    cv2.destroyAllWindows()

        

main()
