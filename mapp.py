from multiprocessing import Process, Queue
import OpenGL.GL as gl
import pangolin
import numpy as np
import g2o

class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.state = None

        self.q = Queue()
        p = Process(target=self.viewer_thread, args=(self.q,))
        p.daemon = True
        p.start()

    # add g2o optimizer
    def optimize(self):
        
        # 创建 g2o optimizer
        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        opt.set_algorithm(solver)

        robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))

        # add frames to graph
        for f in self.frames:
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_id(f.id)
            v_se3.set_estimate(g2o.SE3Quat(f.pose[0:3, 0:3], f.pose[3, 0:3]))
            v_se3.set_fixed(f.id == 0)
            opt.add_vertex(v_se3)

        # add points to frames
        for p in self.points:
            pt = g2o.VertexSBAPointXYZ()
            pt.set_id(p.id + 0x10000)
            pt.set_estimate(p.pt[0:3])
            pt.set_marginalized(True)
            opt.add_vertex(pt)


        for f in p.frames:
            edge = g2o.EdgeSE3ProjectXYZ()
            edge.set_vertex(0, pt)
            edge.set_vertex(1, opt.vertex(f.id))
            edge.set_measurement(f.kps[f.pts.index(p)])
            edge.set_information(np.eye(2))
            edge.set_robust_kernel(robust_kernel)
            opt.add_edge(edge)

        # 初始化
        opt.initialize_optimization()
        opt.set_verbose(True)
        opt.optimize(20)

    # 可视化
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


class Point(object):
    '''
    A Point is a 3-D point in the world
    Each Point is observed in multiple Frames
    '''
    def __init__(self, mapp, loc):
        self.frames = []
        self.pt = loc
        self.idxs = []

        self.id = len(mapp.points)
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)
        
        