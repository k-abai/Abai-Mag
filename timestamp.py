# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 17:15:56 2025

@author: kekea
"""
import numpy as np
import pandas as pd

#convert float to int for timestamp
def time_int(df):
    df_time = df[['year','month','day','hour','minute','second']]
    df_time_int = df_time.astype(int)
    return df_time_int
#function extraxts timestamp in string from mag data and creates
def timestamp(df):
    df = time_int(df)
    
    df2 = df #make second df for 'end'
    #df2['second'] = df2.apply(lambda row: row['second'] + 1, axis=1) #makes 'end' time (one second after start)
    
    #converts into timnestamp
    df['start'] = df.apply(lambda row: f"{row['year']:04}-{row['month']:02}-{row['day']:02} "f"{row['hour']:02}:{row['minute']:02}:{row['second']:02}", axis=1)
   # df2['stop'] = df2.apply(lambda row: f"{row['year']:04}-{row['month']:02}-{row['day']:02} "f"{row['hour']:02}:{row['minute']:02}:{row['second']:02}", axis=1)
    
    df_start = df['start'] 
    #df_end = df2['stop']
   # timestamp = pd.DataFrame({'start': df_start,'stop': df_end})
    #timestamp = timestamp.astype(str)
    timestamp = pd.to_datetime(df_start,utc=True)
    return timestamp