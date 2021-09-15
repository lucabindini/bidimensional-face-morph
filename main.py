#!/usr/bin/python3

import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import fitting
import landmarks

FRAMES_PATH = 'LR/stefano_berretti/rgbReg_frames'
DEF_SHAPE = 'defShape'
OUTPUT_DIR_3D = 'deformation_heatmaps'

if not os.access(OUTPUT_DIR_3D, os.F_OK):
    os.mkdir(OUTPUT_DIR_3D)
if not os.access(landmarks.OUTPUT_DIR_2D, os.F_OK):
    os.mkdir(landmarks.OUTPUT_DIR_2D)

for i in range(1, len(os.listdir(FRAMES_PATH)) + 1):
    print(i)

    preds0 = landmarks.lm_dir(FRAMES_PATH, i, True, True)
    def_shape00 = fitting.fit_3dmm(preds0)[DEF_SHAPE]

    def_shape0 = def_shape00.copy()
    def_shape00 = fitting.fit_3dmm(preds0, def_shape00)[DEF_SHAPE]

    preds1 = landmarks.lm_dir(FRAMES_PATH, i, False, True)
    def_shape01 = def_shape0.copy()  # def_shape0 -> def_shape00
    def_shape01 = fitting.fit_3dmm(preds1, def_shape01)[DEF_SHAPE]

    errors = landmarks.models_error(def_shape00, def_shape01)  # def_shape00 -> def_shape0
    print(errors.mean())

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1, projection='3d', elev=90, azim=-90)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d', elev=90, azim=-90)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d', elev=90, azim=-90)
    ax1.scatter3D(*def_shape0.transpose(), s=1, c=errors)
    ax2.scatter3D(*def_shape00.transpose(), s=1, c=errors)
    plot = ax3.scatter3D(*def_shape01.transpose(), s=1, c=errors)
    ax1.axis(False)
    ax2.axis(False)
    ax3.axis(False)
    fig.colorbar(plot)
    fig.savefig(f'{OUTPUT_DIR_3D}/{i:04}.png')
    plt.close(fig)
