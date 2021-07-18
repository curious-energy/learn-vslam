import cv2
import numpy as np
from numpy.core.defchararray import add
from numpy.core.fromnumeric import shape
from numpy.core.records import array
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1) 

def extractorRt(E):
    W = np.mat([
        [0,-1,0],
        [1, 0,0],
        [0, 0,1]
    ], dtype=float)
    U, d, Vt = np.linalg.svd(E)
    # print(d)
    assert(np.linalg.det(U) > 0)
    if np.linalg.det(Vt) < 0:
        Vt *= -1
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    pose = np.concatenate([R, t.reshape(3, 1)], axis=1)
    return pose


class Extractor(object):
    # GX = 16 // 2
    # GY = 12 // 2
    def __init__(self, K):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None

        self.K = K
        self.invK = np.linalg.inv(K)

    def normalize(self, pts):
        return np.dot(self.invK, add_ones(pts).T).T[:, 0:2]

    def denormalize(self, pt):
        # return int(round(pt[0] + self.w//2)), int(round(pt[1] + self.h//2))
        ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]).T)
        # ret /= ret[2]
        # print("ret",ret)
        return int(round(ret[0])), int(round(ret[1]))

    def extract(self, img):
        # 这儿是分区域特征点，效果欠佳
        # sy = img.shape[0] // self.GY
        # sx = img.shape[1] // self.GX
        # akp = []
        # for ry in range(0, img.shape[0], sy):
        #     for rx in range(0, img.shape[1], sx):
        #         # print(ry,rx)
        #         chunk = img[ry:ry+sy, rx:rx+sx]
        #         # print(chunk)
        #         kp = self.orb.detect(chunk, None)
        #         # print(kp)
        #         for p in kp:
        #             print(p.pt)
        #             p.pt = (p.pt[0] + rx, p.pt[1] + ry)
        #             akp.append(p)
        # return akp
        # detect
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)

        # extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)
        
        # matching
        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], k=2) # 需要学习
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))

        # filter
        Rt = None
        if len(ret) > 0:
            ret = np.array(ret)
            # print(ret.shape)

            # normalize coords: subtract to 0
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])

            model, inliers = ransac((ret[:,0], ret[:,1]),
                                    EssentialMatrixTransform,
                                    # FundamentalMatrixTransform,
                                    min_samples=8,
                                    residual_threshold=0.005,
                                    max_trials=100)
            # print(sum(inliers), len(inliers))
            ret = ret[inliers]

            Rt = extractorRt(model.params)
            # print(pose)


        # return
        self.last = {'kps':kps, "des":des}
        return ret, Rt
