'''
    YOLOV7 Detect 层
    author: xiaohe
'''
import numpy as np

class Detect():
    # YOLOv5 Detect head for detection models
    def __init__(self, nc=80, anchors=(), stride=[], image_size=640):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.stride = stride
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid, self.anchor_grid = [0]*self.nl, [0]*self.nl
        self.anchors = np.array(anchors, dtype=np.float32).reshape(self.nl, -1, 2)

        base_scale = image_size//8
        for i in range(self.nl):
            self.grid[i], self.anchor_grid[i] = self._make_grid(base_scale//(2**i), base_scale//(2**i), i)

    def _make_grid(self, nx=20, ny=20, i=0):
        y, x = np.arange(ny, dtype=np.float32), np.arange(nx, dtype=np.float32)
        yv, xv = np.meshgrid(y, x)
        yv, xv = yv.T, xv.T
        # add grid offset, i.e. y = 2.0 * x - 0.5
        grid = np.stack((xv, yv), 2)
        grid = grid[np.newaxis, np.newaxis, ...]
        grid = np.repeat(grid, self.na, axis=1) - 0.5
        anchor_grid = self.anchors[i].reshape((1, self.na, 1, 1, 2))
        anchor_grid = np.repeat(anchor_grid, repeats=ny, axis=2)
        anchor_grid = np.repeat(anchor_grid, repeats=nx, axis=3)
        return grid, anchor_grid

    def sigmoid(self, arr):
        return 1/(1+np.exp(-arr))

    def __call__(self, x):
        z = []  # inference output
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].reshape(bs, self.na, self.no, ny, nx).transpose(0, 1, 3, 4, 2)
            y = self.sigmoid(x[i])
            y[..., 0:2] = (y[..., 0:2] * 2. + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.reshape(bs, self.na * nx * ny, self.no))

        return np.concatenate(z, 1)

        