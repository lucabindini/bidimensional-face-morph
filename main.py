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
OUTPUT_DIR_3D_FINAL = 'final_deformation_heatmaps'
OUTPUT_DIR_3D_INTERMEDIATE = 'intermediate_deformation_heatmaps'
OUTPUT_DIR_PLOT = 'temporal_deformation_plots'
N = 4  # number of frames analysed in each sequence
MEDIAN_ITER = 1  # number of iteration to get median
VMAX = 4  # saturation value for uniform 3d model scaling
LR = True  # set to True if using the LR dataset (due to additional initial frames issue)

for directory in {OUTPUT_DIR_3D_FINAL, OUTPUT_DIR_3D_INTERMEDIATE, landmarks.OUTPUT_DIR_2D, OUTPUT_DIR_PLOT}:
    if not os.access(directory, os.F_OK):
        os.mkdir(directory)

fig_all, ax_all = plt.subplots()
ax_all.set_xlabel('frame fraction')
ax_all.set_ylabel('deformation index')

for i in range(1, len(os.listdir(FRAMES_PATH)) + 1):

    path = f'{FRAMES_PATH}/{i:04}'
    start = int(
        os.path.splitext(min(os.listdir(path), key=lambda s: int(os.path.splitext(s)[0])))[0])
    if LR:
        start += i - 1
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
    fig_intermediate = plt.figure(figsize=plt.figaspect(N))

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

        ax_intermediate = fig_intermediate.add_subplot(N, 1, j, projection='3d', elev=90, azim=-90)
        ax_intermediate.scatter3D(*def_shape_median.transpose(), s=1, c=def_indices, cmap='jet', vmin=0, vmax=VMAX)
        ax_intermediate.axis(False)
        ax_intermediate.set_title(f'{y[-1]:.2f}')

    fig_final = plt.figure()
    ax_final = fig_final.add_subplot(1, 1, 1, projection='3d', elev=90, azim=-90)
    ax_final.scatter3D(*def_shape_median.transpose(), s=1, c=def_indices, cmap='jet', vmin=0, vmax=VMAX)
    ax_final.axis(False)
    ax_final.set_title(f'{y[-1]:.2f}')
    fig_final.savefig(f'{OUTPUT_DIR_3D_FINAL}/{i:04}.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig_final)

    fig_intermediate.savefig(f'{OUTPUT_DIR_3D_INTERMEDIATE}/{i:04}.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig_intermediate)

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(y)) / N, y)
    if not LR or i > 1:
        ax_all.plot(np.arange(len(y)) / N, y, color='C0')
    ax.set_xlabel('frame fraction')
    ax.set_ylabel('deformation index')
    fig.savefig(f'{OUTPUT_DIR_PLOT}/{i:04}.png')
    plt.close(fig)

    print(f'{i} / {len(os.listdir(FRAMES_PATH))} expressions done')

fig_all.savefig(f'{OUTPUT_DIR_PLOT}/all.png')
