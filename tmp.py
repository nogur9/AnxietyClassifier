import xlsxwriter
import os
import pandas as pd
import numpy as np

df = pd.DataFrame({'a':['inf', 'y', 0], 'b':['blabla', '-inf', 999]})
df = df.replace(to_replace={'inf': np.nan, '-inf': np.nan})
print(df)