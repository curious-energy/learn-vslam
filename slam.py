#!/usr/bin/env python3
import cv2
from pygame import draw
# from pygame.constants import NOEVENT
from display import Display2D
import numpy as np
from frame import Frame, denormalize, match_frames, IRt
import g2o

import OpenGL.GL as gl
import pangolin

# 设定画面大小
W, H = 1920 // 2, 1080 // 2

# 相机内参
F = 270
K = np.array([
    [F, 0, W//2],
    [0, F, H//2],
    [0, 0, 1]
])

from multiprocessing import Process, Queue

class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.state = None

        self.q = Queue()
        p = Process(target=self.viewer_thread, args=(self.q,))
        p.daemon = True
        p.start()


    def viewer_thread(self, q):
        self.viewer_init(1024, 768)
        while 1:
            self.viewer_refresh(q)

    def viewer_init(self, w, h):
        pangolin.CreateWindowAndBind('Main', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 1000),
            pangolin.ModelViewLookAt(0, -10, -20,
                                     0,  0,  0,
                                     0, -1,  0)) # 调整窗口可视方向
                                     #pangolin.AxisDirection.AxisY))
        handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -w/h)
        self.dcam.SetHandler(handler)

    def viewer_refresh(self, q):
        if self.state is None or not q.empty():
            self.state = q.get()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)

        # draw pose
        # gl.glPointSize(10)
        gl.glColor3f(0.0, 1.0, 0.0)
        # pangolin.DrawPoints(np.array([d[:3, 3] for d in self.state[0]]))
        # camera pose 
        pangolin.DrawCameras(self.state[0])
        
        # draw keypoints
        gl.glPointSize(3)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawPoints(np.array(self.state[1]))

        pangolin.FinishFrame()


    def display(self):
        poses, pts = [], []
        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.pt)
        self.q.put((poses, pts))
        

disp = Display2D(W, H)
mapp = Map()

class Point(object):
    '''
    A Point is a 3-D point in the world
    Each Point is observed in multiple Frames
    '''
    def __init__(self, mapp, loc):
        self.frames = []
        self.pt = loc
        self.idxs = []
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)


def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3,:], pose2[:3,:], pts1.T, pts2.T).T


# frames = []
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
