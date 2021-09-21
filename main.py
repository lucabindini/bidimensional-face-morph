#!/usr/bin/python3

import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

import fitting
import landmarks

FRAMES_PATH = 'LR/stefano_berretti/rgbReg_frames'
DEF_SHAPE = 'defShape'
OUTPUT_DIR_3D_ITER = 'iter_deformation_heatmaps'
OUTPUT_DIR_3D_INTERMEDIATE = 'intermediate_deformation_heatmaps'
OUTPUT_DIR_PLOT = 'temporal_deformation_plots'
N = 8
MEDIAN_ITER = 4

for directory in {OUTPUT_DIR_3D_ITER, OUTPUT_DIR_3D_INTERMEDIATE, landmarks.OUTPUT_DIR_2D, OUTPUT_DIR_PLOT}:
    if not os.access(directory, os.F_OK):
        os.mkdir(directory)

for i in range(1, len(os.listdir(FRAMES_PATH)) + 1):
    print('i =', i)

    path = f'{FRAMES_PATH}/{i:04}'
    # adding (i - 1) to compensate for first frames coming from a different cut
    start = int(
        os.path.splitext(min(os.listdir(path), key=lambda s: int(os.path.splitext(s)[0])))[0]) + i - 1
    def_shapes = []
    def_shapes0 = []
    for k in range(start, start + MEDIAN_ITER):
        img_path = f'{path}/{k}.jpg'
        preds0 = landmarks.lm_dir(img_path, True)
        def_shape00 = fitting.fit_3dmm(preds0)[DEF_SHAPE]
        def_shape0 = def_shape00.copy()
        def_shape00 = fitting.fit_3dmm(preds0, def_shape00)[DEF_SHAPE]
        def_shapes.append(def_shape00)
        def_shapes0.append(def_shape0)
    def_shapes_a = np.array(def_shapes)
    def_shape00 = np.median(def_shapes_a, axis=0)
    def_shapes0_a = np.array(def_shapes0)
    def_shape0 = np.median(def_shapes0_a, axis=0)

    y = [0]

    for j in range(1, N + 1):
        print('j =', j)
        def_shapes = []
        end = int(os.path.splitext(max(os.listdir(path), key=lambda s: int(os.path.splitext(s)[0])))[0])
        n = start + int((end - start) * j/N)
        for k in range(n + 1 - MEDIAN_ITER, n + 1):
            img_path = f'{path}/{k}.jpg'
            preds1 = landmarks.lm_dir(img_path, True)
            def_shape01 = def_shape0.copy()
            def_shape01 = fitting.fit_3dmm(preds1, def_shape01)[DEF_SHAPE]
            def_shapes.append(def_shape01)
        def_shapes_a = np.array(def_shapes)
        def_shape_median = np.median(def_shapes_a, axis=0)

        errors = landmarks.models_error(def_shape00, def_shape_median)
        y.append(errors.mean())
        print(y[-1])

        if j == N // 2:
            def_shape_intermediate = def_shape_median.copy()
            errors_intermediate = errors.copy()

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection='3d', elev=90, azim=-90)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d', elev=90, azim=-90)
        ax1.scatter3D(*def_shape00.transpose(), s=1, c=errors, cmap='jet', norm=colors.PowerNorm(gamma=0.75))
        ax2.scatter3D(*def_shape_median.transpose(), s=1, c=errors, cmap='jet', norm=colors.PowerNorm(gamma=0.75))
        ax1.axis(False)
        ax2.axis(False)
        fig.savefig(f'{OUTPUT_DIR_3D_ITER}/{i:04}-{j}.png')
        plt.close(fig)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d', elev=90, azim=-90)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d', elev=90, azim=-90)
    vmax = max(np.max(errors_intermediate), np.max(errors))
    ax1.scatter3D(*def_shape_intermediate.transpose(), s=1, c=errors_intermediate, cmap='jet',
                  norm=colors.PowerNorm(gamma=0.75, vmax=vmax))
    ax2.scatter3D(*def_shape_median.transpose(), s=1, c=errors, cmap='jet', norm=colors.PowerNorm(gamma=0.75, vmax=vmax))
    ax1.axis(False)
    ax2.axis(False)
    ax1.set_title(errors_intermediate.mean())
    ax2.set_title(errors.mean())
    fig.savefig(f'{OUTPUT_DIR_3D_INTERMEDIATE}/{i:04}.png')
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(y)) / N, y)
    ax.set_xlabel("frame's fraction")
    ax.set_ylabel('mean error')
    fig.savefig(f'{OUTPUT_DIR_PLOT}/{i:04}.png')
    plt.close(fig)
