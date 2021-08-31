import numpy as np
import numpy.matlib as npm
import h5py

from deep_3dmm_refinement._3DMM import _3DMM
from SLC_3DMM import Matrix_operations


def fit_3dmm(lm, avg_model=None):
    # Fixed Params
    rounds = 1
    r = 3.8
    c_dist = 400
    
    lm = lm[:, :2]
    lm = np.delete(lm, 64, axis=0)
    lm = np.delete(lm, 60, axis=0)

    # Load 3D data
    avgfile = h5py.File('data3dmm/avgModel_bh_1779_NE_mediumBound.mat', 'r')
    idx_landmarks_3D = np.transpose(np.array(avgfile['idxLandmarks3D']))
    idx_landmarks_3D -= 1
    landmarks_3D = np.transpose(np.array(avgfile['landmarks3D']))
    
    if avg_model is None:
        _lambda = 0.15
        componentsfile_path = 'data3dmm/components_DL_300_1779.mat'
    else:
        _lambda = 1
        componentsfile_path = 'SLC_3DMM/data/SLC_300_1_1.mat'

    # Load 3DMM params and dictionary
    componentsfile = h5py.File(componentsfile_path, 'r')
    Components = np.transpose(np.array(componentsfile['Components']))
    Weights = np.array(componentsfile['Weights'])
    if avg_model is None:
        avg_model = np.transpose(np.array(avgfile['avgModel']))
        Weights = np.transpose(Weights)
        Components_res = np.transpose(np.array(componentsfile['Components_res']))
    else:
        aligned_models_data = None
        components_R = Matrix_operations.Matrix_op(Components, aligned_models_data)
        components_R.reshape(Components)
        Components_res = components_R.X_res
    print('Data Loaded.')
    
    # Center to zero
    baric_3dmm = np.mean(avg_model, axis=0)
    avg_model = avg_model - npm.repmat(baric_3dmm, avg_model.shape[0], 1)
    landmarks_3D = landmarks_3D - npm.repmat(baric_3dmm, landmarks_3D.shape[0], 1)

    # Create 3DMM object
    _3DMM_obj = _3DMM()

    # Fit the 3DMM
    result = _3DMM_obj.opt_3DMM_fast(Weights, Components, Components_res,
                                     landmarks_3D, idx_landmarks_3D, lm, avg_model, _lambda, rounds, r, c_dist)
    print('3DMM Fitting Completed')

    return result
