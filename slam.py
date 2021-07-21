#!/usr/bin/env python3
import cv2
from pygame.constants import NOEVENT
from display import Display2D
import numpy as np
from frame import Frame, denormalize, match_frames, IRt
import g2o

# 设定画面大小
W, H = 1920 // 2, 1080 // 2
# 相机内参
F = 270
K = np.array([
    [F, 0, W//2],
    [0, F, H//2],
    [0, 0, 1]
])


class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []

    def display(self):
        for f in self.frames:
            print(f.id)
            print(f.pose)
            print()

disp = Display2D(W, H)
mapp = Map()

class Point(object):
    '''
    A Point is a 3-D point in the world
    Each Point is observed in multiple Frames
    '''
    def __init__(self, mapp, loc):
        self.frames = []
        self.xyz = loc
        self.idxs = []
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)


def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3,:], pose2[:3,:], pts1.T, pts2.T).T


frames = []
def process_frame(img):
    img = cv2.resize(img, (W, H))
    frame = Frame(mapp, img, K)
    # matches, pose = fe.extract(img)
    # frames.append(frame)

    if frame.id == 0:
        return
    
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)
    f1.pose = np.dot(Rt, f2.pose)
    # print(Rt[:3,:])
    
    # pts4d = cv2.triangulatePoints(IRt[:3,:], Rt[:3,:], pts[:,0].T, pts[:,1].T).T
    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
    # print(pts4d)
    # homogenous 3-D coords
    pts4d /= pts4d[:, 3:]

    # reject pts without enough "parallax" and points behind the camera : z > 0
    good_pts4d = (np.abs(pts4d[:, 3]) > .001) & (pts4d[:, 2] > 0)
    # print(sum(good_pts4d), len(good_pts4d))
    # pts4d = pts4d[good_pts4d]

    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])

    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
        # cv2.circle(img, (u2, v2), color=(255, 0, 0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(0, 0, 255))

    disp.paint(img)
    mapp.display()


if __name__ == "__main__":
    cap = cv2.VideoCapture("./test.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        # print(frame.shape)
        if ret == True:
            process_frame(frame)
        else:
            break
