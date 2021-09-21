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
        pred_types = {'face': pred_type(slice(0, 17), 'C0'),
                      'eyebrow1': pred_type(slice(17, 22), 'C1'),
                      'eyebrow2': pred_type(slice(22, 27), 'C1'),
                      'nose': pred_type(slice(27, 31), 'C2'),
                      'nostril': pred_type(slice(31, 36), 'C3'),
                      'eye1': pred_type(slice(36, 42), 'C4'),
                      'eye2': pred_type(slice(42, 48), 'C4'),
                      'lips': pred_type(slice(48, 60), 'C5'),
                      'teeth': pred_type(slice(60, 68), 'C6')
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

        fig.savefig(f'{OUTPUT_DIR_2D}/{os.path.basename(os.path.dirname(path))}.png')
        plt.close(fig)
    return preds


def models_error(shape1, shape2):
    return ((shape1 - shape2) ** 2).sum(axis=1) ** (1 / 2)
