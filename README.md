# Bidimensional face morph

This project aims to measure the facial deformation of a person in an objective way.
It uses frames captured from a video to create a 3D model of the face which is compared against a frame of the same person in a neutral expression.

## Requirements

The following are the Python libraries required:

- [numpy](https://numpy.org)
- [matplotlib](https://matplotlib.org)
- [h5py](https://www.h5py.org/)
- [face_alignment](https://github.com/1adrianb/face-alignment) by [Adrian Bulat](https://www.adrianbulat.com/)

You also need to download the following repositories and put them in specific directories:

- [deep-3dmm-refinement](https://github.com/clferrari/deep-3dmm-refinement) inside `deep_3dmm_refinement`
- [SLC-3DMM](https://github.com/clferrari/SLC-3DMM) inside `SLC-3DMM`

## Datasets

Download [data3dmm](https://drive.google.com/file/d/12ull7YHxsqEvF4OlllOc8kneS9h4fI7y/view) for the average model components and put it inside a directory called `data3dmm`.

Create a folder structure like so: `LR/<your_name>/rgbReg_frames/` and inside that put your frames in directories called `0001`, `0002`, etc.

## Usage

Run the `main.py` file.
This will create and populate the following output directories:
- `landmarks_2d` where the images with the 2D landmarks will be saved;
- `deformation_heatmaps` where your heatmaps for the face deformation will end up.