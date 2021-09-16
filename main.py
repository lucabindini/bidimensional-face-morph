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
OUTPUT_DIR_3D = 'deformation_heatmaps'
OUTPUT_DIR_PLOT = 'temporal_deformation_plots'
N = 8

for directory in {OUTPUT_DIR_3D, landmarks.OUTPUT_DIR_2D, OUTPUT_DIR_PLOT}:
    if not os.access(directory, os.F_OK):
        os.mkdir(directory)

for i in range(1, len(os.listdir(FRAMES_PATH)) + 1):
    print('i =', i)

    preds0 = landmarks.lm_dir(FRAMES_PATH, i, 0, True)
    def_shape00 = fitting.fit_3dmm(preds0)[DEF_SHAPE]

    def_shape0 = def_shape00.copy()
    def_shape00 = fitting.fit_3dmm(preds0, def_shape00)[DEF_SHAPE]

    y = [0]

    for j in range(1, N+1):
        print('j =', j)
        preds1 = landmarks.lm_dir(FRAMES_PATH, i, j/N, True)
        def_shape01 = def_shape0.copy()
        def_shape01 = fitting.fit_3dmm(preds1, def_shape01)[DEF_SHAPE]

        errors = landmarks.models_error(def_shape00, def_shape01)
        y.append(errors.mean())
        print(y[-1])

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1, projection='3d', elev=90, azim=-90)
        ax2 = fig.add_subplot(1, 3, 2, projection='3d', elev=90, azim=-90)
        ax3 = fig.add_subplot(1, 3, 3, projection='3d', elev=90, azim=-90)
        ax1.scatter3D(*def_shape0.transpose(), s=1, c=errors, cmap='jet', norm=colors.PowerNorm(gamma=0.75))
        ax2.scatter3D(*def_shape00.transpose(), s=1, c=errors, cmap='jet', norm=colors.PowerNorm(gamma=0.75))
        plot = ax3.scatter3D(*def_shape01.transpose(), s=1, c=errors, cmap='jet', norm=colors.PowerNorm(gamma=0.75))
        ax1.axis(False)
        ax2.axis(False)
        ax3.axis(False)
        fig.colorbar(plot)
        fig.savefig(f'{OUTPUT_DIR_3D}/{i:04}-{j}.png')
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(y))/N, y)
    fig.savefig(f'{OUTPUT_DIR_PLOT}/{i:04}.png')
    plt.close(fig)
