#!/usr/bin/env python3
import cv2
from pygame.constants import NOEVENT
from display import Display2D
import numpy as np
from extractor import Extractor

W = 1920 // 2
H = 1080 // 2

F = 270
disp = Display2D(W, H)


K = np.array([
    [F, 0, W//2],
    [0, F, H//2],
    [0, 0, 1]
])
fe = Extractor(K)


def process_frame(img):
    img = cv2.resize(img, (W, H))
    # kp, des = orb.detectAndCompute(img,None)
    matches, pose = fe.extract(img)
    if pose is None:
        pass
    print("%d matches" % len(matches))
    print(pose)



    for pt1, pt2 in matches:
        # print(p)
        # u, v = map(lambda x: int(round(x)), p[0])
        # u1, v1 = map(lambda x: int(round(x)), pt1)
        # u2, v2 = map(lambda x: int(round(x)), pt2)

        u1, v1 = fe.denormalize(pt1)
        u2, v2 = fe.denormalize(pt2)

        cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
        # cv2.circle(img, (u2, v2), color=(255, 0, 0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(0, 0, 255))

    disp.paint(img)






if __name__ == "__main__":
    cap = cv2.VideoCapture("./test.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        # print(frame.shape)
        if ret == True:
            process_frame(frame)
        else:
            break
