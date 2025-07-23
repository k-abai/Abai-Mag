# Abai-Mag


Abai-Mag is a small collection of Python scripts for training a neural network to
predict magnetopause location. It learns from magnetopause crossing data and
solar wind parameters (taken from the OMNI database) and can then estimate the
radial distance or full 3‑D location of the boundary from simple input values.

Abai-Mag is a small collection of Python scripts for training a neural network to predict magnetopause location. The code trains a model using magnetopause crossing data and solar wind parameters taken from the OMNI database. The resulting model can be used to predict the radial distance or 3‑D location of the magnetopause from simple input parameters.


## Installation

1. Clone this repository.
2. Create a Python 3 environment and install the required packages:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow joblib
   ```
3. Place `omni_data.h5` (OMNI solar wind data in HDF5 format) in the project directory. The training and prediction scripts expect this file to be available when calling `makedata`.

## Dataset

Along with the OMNI file, you will need a CSV of magnetopause crossings. Its
location is configured in `model2.py` via the `mag_data_path` variable. The
`makedata` helper combines the crossing data with values from `omni_data.h5` to
produce the training set used by the neural network.


## Training

Edit `model2.py` and set `mag_data_path` to the path of your magnetopause crossing CSV file. Then run

```bash
python model2.py
```

This trains the neural network and saves the model as `my_model2.keras` together with the input and output scalers (`input_xscaler.pkl` and `input_yscaler.pkl`).

## Prediction

To obtain predictions for new input values you can use the helper in `predict.py`:

```python
from predict import abaimp

# example values (lat, lon, bz, pdyn, tilt, Bx, By, Bz, Vx, Vy, Vz)
result = abaimp(1.0, 1.0, -0.8, 1.9, -0.2, -3, -0.2, -0.8, -400, 3, -8.5)
print(result)
```

The utilities in `xyz_predict.py` and `valtest.py` offer additional ways to visualise or validate model output.


## Data Visualisation

Use `xyz_predict.py` to explore the magnetopause crossings in your dataset by plotting the predicted magnetopause points in three dimensions to check how they cluster in space.

* Simple histograms of `r_hist.py`, `Bz_hist.py` and dynamic pressure (`pdyn_hist.py`) reveal their most common values.

These plots show that most radii fall between 0 and 5 RE while `Bz` usually lies within roughly ±10 nT, and pdyn concentrates around 3 nPa. With more labelled crossings the model can be trained on a larger set and the validation loss (about 2.8 RE with the sample data) generally improves.

## Synthetic Magnetopause

The script `synth_mp.py` is the final product of this project. It loads a saved
model and generates a synthetic magnetopause surface for any set of solar wind
conditions. The routine sweeps through latitude and longitude angles, predicts
the corresponding radial distance and then plots the boundary as 3‑D and planar
scatter plots. This provides an immediate visual impression of the magnetopause
shape under the specified conditions.

```python
from synth_mp import synth_mp

# Example solar wind parameters
synth_mp(pdyn=2.5, bx=0.0, by=0.0, bz=-5.0, vx=-400, vy=0.0, vz=0.0, tilt=0.0)
print(result)
```
## Jupyter Lab Demo

A short demonstration notebook is available in `notebooks/synth_mp_demo.ipynb`.
To explore the synthetic magnetopause interactively, install JupyterLab and
launch it from the repository root:

```bash
pip install jupyterlab
jupyter lab
```

Open the `synth_mp_demo.ipynb` notebook and run the provided cell to display the
magnetopause surface under example solar wind conditions.

## Acknowledgment

I would like to extend my sincere thanks to **Connor O'Brien** ([connor-obrien888](https://github.com/connor-obrien888)) for his mentorship and support throughout the development of this project.

Connor's guidance helped shape both the structure and logic of the codebase, and his contributions were instrumental in my learning process. Special thanks for providing the `sw_loader.py` module, which played a key role in the implementation.

> His work: [https://github.com/connor-obrien888](https://github.com/connor-obrien888)

## License

This project is distributed under the terms of the [MIT License](LICENSE).
