#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class Detect:
    """YOLOv7 detect head decoder used by the QNN detector predictor."""

    def __init__(self, nc=80, anchors=(), stride=(), image_size=640):
        self.nc = int(nc)
        self.no = self.nc + 5
        self.stride = tuple(int(v) for v in stride)
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid, self.anchor_grid = [0] * self.nl, [0] * self.nl
        self.anchors = np.array(anchors, dtype=np.float32).reshape(self.nl, -1, 2)

        base_scale = int(image_size) // int(self.stride[0] if self.stride else 8)
        for i in range(self.nl):
            scale = int(self.stride[i] // self.stride[0]) if self.stride else (2 ** i)
            self.grid[i], self.anchor_grid[i] = self._make_grid(base_scale // scale, base_scale // scale, i)

    def _make_grid(self, nx=20, ny=20, i=0):
        y, x = np.arange(ny, dtype=np.float32), np.arange(nx, dtype=np.float32)
        yv, xv = np.meshgrid(y, x)
        yv, xv = yv.T, xv.T
        grid = np.stack((xv, yv), 2)
        grid = grid[np.newaxis, np.newaxis, ...]
        grid = np.repeat(grid, self.na, axis=1) - 0.5
        anchor_grid = self.anchors[i].reshape((1, self.na, 1, 1, 2))
        anchor_grid = np.repeat(anchor_grid, repeats=ny, axis=2)
        anchor_grid = np.repeat(anchor_grid, repeats=nx, axis=3)
        return grid, anchor_grid

    @staticmethod
    def sigmoid(arr):
        return 1 / (1 + np.exp(-arr))

    def __call__(self, x):
        outputs = []
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape
            layer = x[i].reshape(bs, self.na, self.no, ny, nx).transpose(0, 1, 3, 4, 2)
            y = self.sigmoid(layer)
            y[..., 0:2] = (y[..., 0:2] * 2.0 + self.grid[i]) * self.stride[i]
            y[..., 2:4] = (y[..., 2:4] * 2.0) ** 2 * self.anchor_grid[i]
            outputs.append(y.reshape(bs, self.na * nx * ny, self.no))
        return np.concatenate(outputs, axis=1)
