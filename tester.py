import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from math import pow



VIDEO_FILE = "catto.mp4"


def create_pyramid(frame):
    """
    Create an image pyramid out of a frame
    :param frame: the original image
    :return pyramid: list of downsampled images after the original image
    """

    MIN_H = 600
    MIN_W = 600

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

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    net = load_model("catnet.h5")


    WIN_H = 600
    WIN_W = 600

    cap = cv2.VideoCapture(VIDEO_FILE)
    #frame = cv2.imread("cat1.jpg")

    while True:
        ret, frame = cap.read()

        pyramid = create_pyramid(frame)

        detections = []

        # search matches in every layer of the pyramid
        for i, img in enumerate(pyramid):

            h = img.shape[0]
            w = img.shape[1]

            h_iterations = np.int32(np.ceil(h / WIN_H))
            w_iterations = np.int32(np.ceil(w / WIN_W))

            # sliding window
            for hi in range(0, h_iterations):
                for wi in range(0, w_iterations):

                    h_start = hi * WIN_H
                    w_start = wi * WIN_W

                    h_end = h_start + WIN_H

                    # pyramid image isn't an integer multiple of the window => the last window will be smaller in each column
                    if h_end >= h:
                        h_end = h

                    w_end = w_start + WIN_W
                    if w_end >= w:
                        w_end = w

                    #cv2.imshow("Feed" + str(i), img)

                    roi = img[h_start:h_end, w_start:w_end]

                    # transform in to the shape the CNN expects
                    #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi = cv2.resize(roi, (50, 50))
                    cv2.imshow("roi", roi)


                    roi = np.reshape(roi, (1, 50, 50, 3))

                    pred = net.predict(roi)
                    #print(pred)

                    DETECTION_THRESHOLD = 0.85

                    scale = int(pow(2, i))
                    #print(scale)

                    # add detections to drawing queue
                    if pred[0] > DETECTION_THRESHOLD:
                        detections.append(((scale * h_start, scale * w_start), (scale * h_end, scale * w_end)))


        # show detections
        for d in detections:
            cv2.rectangle(frame, d[0], d[1], (0, 255, 0), 2)
            print(d)

        cv2.imshow("Kuva", frame)
        k = cv2.waitKey(30) & 0xFF
        # ESC
        if k == 27:
            break

    cv2.destroyAllWindows()


main()
