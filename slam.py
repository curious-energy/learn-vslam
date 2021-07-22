#!/usr/bin/env python3
import cv2
from display import Display2D
import numpy as np
from frame import Frame, denormalize, match_frames
import os

from mapp import Map, Point

# 设定画面大小
W, H = 1920 // 2, 1080 // 2

# 相机内参
F = 700
# F = 270
K = np.array([
    [F, 0, W//2],
    [0, F, H//2],
    [0, 0, 1]
])

disp = Display2D(W, H) if os.getenv("D2D") is not None else None
mapp = Map()

# if os.getenv("D3D") is not None:
#     mapp.create_


def triangulate(pose1, pose2, pts1, pts2):
    ret = np.zeros((pts1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    for i, p  in enumerate(zip(pts1, pts2)):
        A = np.zeros((4, 4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]
    return ret

    # return cv2.triangulatePoints(pose1[:3,:], pose2[:3,:], pts1.T, pts2.T).T

def process_frame(img):
    img = cv2.resize(img, (W, H))
    frame = Frame(mapp, img, K)
    # matches, pose = fe.extract(img)
    # frames.append(frame)

    if frame.id == 0:
        return
    print("\n**** frame id %d ****" % (frame.id, ))
    
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)
    f1.pose = np.dot(Rt, f2.pose)
    # print(Rt[:3,:])
    # print(idx2)
    for i, idx in enumerate(idx2):
        if f2.pts[idx] is not None:
            f2.pts[idx].add_observation(f1, idx1[i])
    
    # pts4d = cv2.triangulatePoints(IRt[:3,:], Rt[:3,:], pts[:,0].T, pts[:,1].T).T
    pts4d = triangulate(f1.pose, f2.pose, f1.kps[idx1], f2.kps[idx2])
    # print(pts4d)
    # homogenous 3-D coords
    pts4d /= pts4d[:, 3:]

    ummatched_points = np.array([f1.pts[i] is None for i in idx1])
    print("Adding: %d points" % np.sum(ummatched_points))
    # reject pts without enough "parallax" and points behind the camera : z > 0
    good_pts4d = (np.abs(pts4d[:, 3]) > .005) & (pts4d[:, 2] > 0) & ummatched_points
    # print(sum(good_pts4d), len(good_pts4d))
    # pts4d = pts4d[good_pts4d]

    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue

        u, v = int(round(f1.kpus[idx1[i], 0])), int(round(f1.kpus[idx1[i], 1]))
        pt = Point(mapp, p, img[v, u])
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])

    for pt1, pt2 in zip(f1.kps[idx1], f2.kps[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
        # cv2.circle(img, (u2, v2), color=(255, 0, 0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(0, 0, 255))

    # display it by cmd "D2D=1 ./slam.py"
    if disp is not None:
        # 2D-display
        disp.paint(img)
    
    # optimize
    if frame.id >= 4:
        mapp.optimize()
        # exit(0)
    # 3D-display
    mapp.display()


if __name__ == "__main__":
    cap = cv2.VideoCapture("./test_road.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        # print(frame.shape)
        if ret == True:
            process_frame(frame)
        else:
            break
