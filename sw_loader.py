"""Utility functions for loading solar wind data."""

import pandas as pd

def load_sw(start, stop, store='omni_data.h5', key='omni_1min'):
    """Load OMNI solar wind data for a given time range.

    Parameters
    ----------
    start : str
        Inclusive UTC start time in ``YYYY-MM-DD HH:MM:SS`` format.
    stop : str
        Inclusive UTC stop time in ``YYYY-MM-DD HH:MM:SS`` format.
    store : str, optional
        Path to the HDF5 file containing OMNI data.
    key : str, optional
        Dataset key within the HDF5 file.

    Returns
    -------
    DataFrame
        Solar wind measurements spanning the requested time range.
    """
    try: #Very hacky way to cache the dataframe
        sw_cut = sw_data.copy()
    except: #Error is thrown when sw_data isn't loaded
        sw_data = pd.read_hdf('omni_data.h5', key = 'omni_1min', mode = 'r')
        sw_cut = sw_data.copy()
    sw_cut = sw_cut[(sw_cut['Epoch'] >= pd.to_datetime(start, utc = True)) & (sw_cut['Epoch'] <= pd.to_datetime(stop, utc = True))].reset_index(drop=True)
    return sw_cut