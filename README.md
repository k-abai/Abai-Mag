# Abai-Mag

Abai-Mag is a small collection of Python scripts for training a neural network to predict magnetopause location. The code trains a model using magnetopause crossing data and solar wind parameters taken from the OMNI database. The resulting model can be used to predict the radial distance or 3‑D location of the magnetopause from simple input parameters.

## Installation

1. Clone this repository.
2. Create a Python 3 environment and install the required packages:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow joblib
   ```
3. Place `omni_data.h5` (OMNI solar wind data in HDF5 format) in the project directory. The training and prediction scripts expect this file to be available when calling `makedata`.

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

The utilities in `rplot.py` and `Valtest.py` offer additional ways to visualise or validate model output.

## Data Visualisation

Use `rplot.py` to explore the magnetopause crossings in your dataset:

* **3D scatter** &ndash; plot the predicted magnetopause points in three dimensions to check how they cluster in space.
* **Radius vs Bz** &ndash; create a scatter of radius (`r`) against the z component of the magnetic field (`Bz`) where the third axis encodes the frequency of observations. Viewed from above this appears as a heat map. Southward `Bz` allows solar wind to enter the magnetosphere and typically reduces the radius.
* Simple histograms of `r`, `Bz` and dynamic pressure (`pdyn`) reveal their most common values.

These plots show that most radii fall between 0&nbsp;and&nbsp;5&nbsp;RE while `Bz` usually lies within roughly ±10&nbsp;nT. With more labelled crossings the model can be trained on a larger set and the validation loss (about 2.8&nbsp;RE with the sample data) generally improves.

## License

This project is distributed under the terms of the [MIT License](LICENSE).
