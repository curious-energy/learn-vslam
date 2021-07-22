import cv2
import numpy as np
from numpy.core.defchararray import add
from numpy.core.fromnumeric import shape
from numpy.core.records import array
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform


IRt = np.eye(4)

# 给向量的后加一列 [[x,y]] -> [[x,y,1]]
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1) 

# pose
def extractorRt(E):
    W = np.mat([
        [0,-1,0],
        [1, 0,0],
        [0, 0,1]
    ], dtype=float)
    # svd分解
    U, d, Vt = np.linalg.svd(E)
    # print(d)
    assert(np.linalg.det(U) > 0)
    if np.linalg.det(Vt) < 0:
        Vt *= -1
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    # pose = np.concatenate([R, t.reshape(3, 1)], axis=1)
    ret = np.eye(4)
    ret[:3,:3] = R
    ret[:3, 3] = t
    # print(ret)
    return ret

def normalize(invK, kps):
    return np.dot(invK, add_ones(kps).T).T[:, 0:2]

def denormalize(K, pt):
    ret = np.dot(K, np.array([pt[0], pt[1], 1.0]).T)
    ret /= ret[2]
    # print("ret",ret)
    return int(round(ret[0])), int(round(ret[1]))

def extractor(img):
    orb = cv2.ORB_create()
    # detection
    kps = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 1000, qualityLevel=0.01, minDistance=10)

    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in kps]
    kps, des = orb.compute(img, kps)

    # return kps and des
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des


def match_frames(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    # 查看匹配点对
    # print("%d matches" % len(matches))

    # matching
    ret = []
    idx1, idx2 = [], []

    for m, n in matches:
        if m.distance < 0.75*n.distance:
            p1 = f1.kps[m.queryIdx]
            p2 = f2.kps[m.trainIdx]

            # if np.linalg.norm((p1-p2)) < 0.1:
            if m.distance < 32:
                if m.queryIdx not in idx1 and m.trainIdx not in idx2:
                    # 保留索引
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)

                    ret.append((p1, p2))

    assert(len(set(idx1)) == len(idx1))
    assert(len(set(idx2)) == len(idx2))

    assert len(ret) >= 8
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    # fit matrix
    model, inliers = ransac((ret[:,0], ret[:,1]),
                            # EssentialMatrixTransform,
                            FundamentalMatrixTransform,
                            min_samples=8,
                            residual_threshold=0.001,
                            max_trials=100)
    # 查看ransac过滤效果
    # print(sum(inliers), len(inliers))

    # 输出匹配结果
    print("Matches: %d -> %d -> %d -> %d" % (len(f1.des), len(matches), len(inliers), sum(inliers)))

    # ignore outliers
    # ret = ret[inliers]
    Rt = extractorRt(model.params)
    # print(pose)

    # return
    return idx1[inliers], idx2[inliers], Rt


class Frame(object):
    def __init__(self, mapp, img, K):
        self.K = K
        self.invK = np.linalg.inv(K)
        self.pose = IRt
        self.h, self.w = img.shape[0:2]

        kps, self.des = extractor(img)
        self.kps = normalize(self.invK, kps)
        self.pts = [None]*len(self.kps)

        self.id = len(mapp.frames)
        mapp.frames.append(self)
