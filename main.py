#!/usr/bin/python3

import os

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

import fitting
import landmarks


for i in range(1, len(os.listdir('LR/stefano_berretti/rgbReg_frames')) + 1):
    print(i)
    preds = landmarks.lm_dir('LR/stefano_berretti/', i, True, True)

    result = fitting.fit_3dmm(preds)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d', elev=90, azim=-90)

    def_shape = result['defShape']
    ax.scatter3D(*def_shape.transpose(), s=1, c=def_shape[:, 2])
    
    preds1 = landmarks.lm_dir('LR/stefano_berretti/', i, False, True)

    def_shape_copy = def_shape.copy()

    result1 = fitting.fit_3dmm(preds1, def_shape)

    errors = landmarks.models_error(def_shape_copy, result1['defShape'])
    
    print(errors.mean())

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d', elev=90, azim=-90)
    ax.axis(False)
    plot = ax.scatter3D(*result1['defShape'].transpose(), s=1, c=errors, cmap='jet', norm=colors.PowerNorm(gamma=0.75))

    fig.colorbar(plot)
    
    fig.savefig('figures/%04d.png' % i)

