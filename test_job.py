import scipy.stats
import ironman
import pandas as pd
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
import rmfit
import corner
import lightkurve as lk

df = pd.read_csv('TOI1259_RV.csv')
plt.plot(df['rv'])
plt.savefig('RV.png')
plt.clf()

search_result = lk.search_lightcurve('TIC 288735205', mission='TESS')
data_1 = search_result[1].download()

plt.plot(data_1["time"].value, data_1["flux"].value)
plt.savefig('TESS.png')
plt.clf()