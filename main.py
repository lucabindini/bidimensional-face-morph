#!/usr/bin/python3

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import fitting
import landmarks

FRAMES_PATH = sys.argv[1]
DEF_SHAPE = 'defShape'
OUTPUT_DIR_3D_ITER = 'iter_deformation_heatmaps'
OUTPUT_DIR_3D_INTERMEDIATE = 'intermediate_deformation_heatmaps'
OUTPUT_DIR_PLOT = 'temporal_deformation_plots'
N = 8
MEDIAN_ITER = 3

for directory in {OUTPUT_DIR_3D_ITER, OUTPUT_DIR_3D_INTERMEDIATE, landmarks.OUTPUT_DIR_2D, OUTPUT_DIR_PLOT}:
    if not os.access(directory, os.F_OK):
        os.mkdir(directory)

fig_all, ax_all = plt.subplots()
ax_all.set_xlabel('frame fraction')
ax_all.set_ylabel('deformation index')

for i in range(1, len(os.listdir(FRAMES_PATH)) + 1):

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
        def_shapes = []
        end = int(os.path.splitext(max(os.listdir(path), key=lambda s: int(os.path.splitext(s)[0])))[0])
        n = start + int((end - start) * j / N)
        for k in range(n + 1 - MEDIAN_ITER, n + 1):
            img_path = f'{path}/{k}.jpg'
            preds1 = landmarks.lm_dir(img_path, True)
            def_shape01 = def_shape0.copy()
            def_shape01 = fitting.fit_3dmm(preds1, def_shape01)[DEF_SHAPE]
            def_shapes.append(def_shape01)
        def_shapes_a = np.array(def_shapes)
        def_shape_median = np.median(def_shapes_a, axis=0)

        def_indices = landmarks.models_deformation(def_shape00, def_shape_median)
        y.append(def_indices.mean())

        if j == N // 2:
            def_shape_intermediate = def_shape_median.copy()
            def_indices_intermediate = def_indices.copy()

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection='3d', elev=90, azim=-90)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d', elev=90, azim=-90)
        ax1.scatter3D(*def_shape00.transpose(), s=1, c=def_indices, cmap='jet')
        ax2.scatter3D(*def_shape_median.transpose(), s=1, c=def_indices, cmap='jet')
        ax1.axis(False)
        ax2.axis(False)
        fig.savefig(f'{OUTPUT_DIR_3D_ITER}/{i:04}-{j}.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d', elev=90, azim=-90)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d', elev=90, azim=-90)
    vmax = max(np.max(def_indices_intermediate), np.max(def_indices))
    ax1.scatter3D(*def_shape_intermediate.transpose(), s=1, c=def_indices_intermediate, cmap='jet', vmax=vmax)
    ax2.scatter3D(*def_shape_median.transpose(), s=1, c=def_indices, cmap='jet', vmax=vmax)
    ax1.axis(False)
    ax2.axis(False)
    ax1.set_title(f'{def_indices_intermediate.mean():.2f}')
    ax2.set_title(f'{def_indices.mean():.2f}')
    fig.savefig(f'{OUTPUT_DIR_3D_INTERMEDIATE}/{i:04}.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(y)) / N, y)
    ax_all.plot(np.arange(len(y)) / N, y, color='C0')
    ax.set_xlabel('frame fraction')
    ax.set_ylabel('deformation index')
    fig.savefig(f'{OUTPUT_DIR_PLOT}/{i:04}.png')
    plt.close(fig)

    print(f'{i} / {len(os.listdir(FRAMES_PATH))} expressions done')

fig_all.savefig(f'{OUTPUT_DIR_PLOT}/all.png')
