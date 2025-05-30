import pandas as pd

def load_sw(start, stop, store = 'omni_data.h5', key = 'omni_1min'):
    '''
    Loads OMNI database solar wind data from HDF file for a particular time range.

    Parameters:
    -----------
    - start (string) : Start time of solar wind data. Format 'YYYY-MM-DD HH:MM:SS'
    - stop (string) : Stop time of solar wind data. Format 'YYYY-MM-DD HH:MM:SS'
    - store (string) : HDF file with data
    - key (string) : Key of desired data within file

    Returns:
    --------
    - sw_cut (DataFrame) : Pandas DataFrame containing solar wind data in desired time range
    '''
    try: #Very hacky way to cache the dataframe
        sw_cut = sw_data.copy()
    except: #Error is thrown when sw_data isn't loaded
        sw_data = pd.read_hdf('omni_data.h5', key = 'omni_1min', mode = 'r')
        sw_cut = sw_data.copy()
    sw_cut = sw_cut[(sw_cut['Epoch'] >= pd.to_datetime(start, utc = True)) & (sw_cut['Epoch'] <= pd.to_datetime(stop, utc = True))].reset_index(drop=True)
    return sw_cut