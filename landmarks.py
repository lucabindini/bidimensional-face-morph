import os
from collections import namedtuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import face_alignment


def lm_dir(path, expression, neutral=False, plot=False):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')

    img_path = path + 'rgbReg_frames/%04d/' % expression

    if neutral:
        # adding (expression - 1) to compensate for first few frames coming from a different cut
        n = int(os.path.splitext(min(os.listdir(img_path), key=lambda s: int(os.path.splitext(s)[0])))[0]) + expression - 1
    else:
        n = int(os.path.splitext(max(os.listdir(img_path), key=lambda s: int(os.path.splitext(s)[0])))[0])

    input_img = mpimg.imread(img_path + str(n) + '.jpg')

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

        for pred_type in pred_types.values():
            ax.plot(preds[pred_type.slice, 0],
                    preds[pred_type.slice, 1],
                    color=pred_type.color,
                    marker='o')

        ax.axis(False)
    return preds


def models_error(shape1, shape2):
    return (((shape1 - shape2) ** 2).sum(axis=1) ** (1 / 2))