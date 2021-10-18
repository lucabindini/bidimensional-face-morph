import os
from collections import namedtuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import face_alignment

OUTPUT_DIR_2D = 'landmarks_2d'


def lm_dir(path, plot=False):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')

    input_img = mpimg.imread(path)

    preds = fa.get_landmarks_from_image(input_img)[-1]

    if plot:
        # 2D-Plot
        pred_type = namedtuple('prediction_type', ['slice', 'color'])
        pred_types = {'face': pred_type(slice(0, 17), 'b'),
                      'eyebrow1': pred_type(slice(17, 22), 'g'),
                      'eyebrow2': pred_type(slice(22, 27), 'g'),
                      'nose': pred_type(slice(27, 31), 'r'),
                      'nostril': pred_type(slice(31, 36), 'c'),
                      'eye1': pred_type(slice(36, 42), 'm'),
                      'eye2': pred_type(slice(42, 48), 'm'),
                      'lips': pred_type(slice(48, 60), 'y'),
                      'teeth': pred_type(slice(60, 68), 'k')
                      }

        fig, ax = plt.subplots()
        ax.imshow(input_img)
        ax.axis(False)

        for pred_type in pred_types.values():
            ax.plot(preds[pred_type.slice, 0],
                    preds[pred_type.slice, 1],
                    color=pred_type.color,
                    marker='o',
                    markersize=1)

        fig.savefig(f'{OUTPUT_DIR_2D}/{os.path.basename(os.path.dirname(path))}_with_image.png', bbox_inches='tight',
                    pad_inches=0)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.axis(False)
        ax.set_aspect('equal')
        ax.invert_yaxis()

        for pred_type in pred_types.values():
            ax.plot(preds[pred_type.slice, 0],
                    preds[pred_type.slice, 1],
                    color=pred_type.color,
                    marker='o')

        fig.savefig(f'{OUTPUT_DIR_2D}/{os.path.basename(os.path.dirname(path))}.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    return preds


def models_deformation(shape1, shape2):
    return ((shape1 - shape2) ** 2).sum(axis=1) ** (1 / 2)
