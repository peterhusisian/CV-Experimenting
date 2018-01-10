import scipy.signal as signal
import numpy as np

'''could make it so that there can only be one extrema within an inputted radius to another extrema'''
def find_maxima(arr, min_val = None):
    extrema = list(signal.argrelextrema(arr, np.greater))

    extrema = np.dstack(extrema)[0]
    if min_val is not None:
        new_extrema = []
        for i in range(0, extrema.shape[0]):

            val = arr[extrema[i][0]]
            for j in range(1, extrema.shape[1]):
                val = val[extrema[i][j]]
            if val > min_val:
                new_extrema.append(extrema[i])
        extrema = np.asarray(new_extrema)
    return extrema
