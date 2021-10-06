# Bidimensional face morph

In face rehabilitation, patients are asked to perform a face deformation and see how well they can deform their faces.
This project aims to give a way to measure this deformation objectively.

Our approach uses frames taken from a colored video to create a 3D model of the person's relaxed face.
Subsequently, a second model is created starting from the frames of the video where a certain grimace is performed and through the comparison of the models the extent and intensity with which the grimace was performed is detected.

## Requirements

The following are the Python libraries required:

- [numpy](https://numpy.org)
- [matplotlib](https://matplotlib.org)
- [h5py](https://www.h5py.org/)
- [face_alignment](https://github.com/1adrianb/face-alignment)

You also need to download the following repositories and rename their directories in the specified way:

- [deep-3dmm-refinement](https://github.com/clferrari/deep-3dmm-refinement) as `deep_3dmm_refinement`
- [SLC-3DMM](https://github.com/clferrari/SLC-3DMM) as `SLC_3DMM`

## Datasets

Download [data3dmm](https://drive.google.com/file/d/12ull7YHxsqEvF4OlllOc8kneS9h4fI7y/view) for the average model components and put it inside a directory called `data3dmm`.

You must put your frames in a directory and separate them in subdirectories called `0001`, `0002`, etc.
Each of these subdirectories contains the frames of a video of someone's face that starts from a relaxed facial expression and end up with some kind of grimace.
The frames must be jpg images called as sequential numbers (e. g. `1916.jpg`, `1917.jpg`, ...)

## Usage

Run the `main.py` file passing to it the directory in which you have all of the frames (e. g. `python3 main.py LR/claudio_ferrari/rgbReg_frames`).

This will create and populate the following output directories:
- `landmarks_2d` where the images with the 2D landmarks will be saved;
- `deformation_heatmaps` where your heatmaps for the face deformation will end up.