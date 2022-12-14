import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def conv1d(data, kernel, pad=(0,0), padmode="reflect"):
    if isinstance(pad, int):
        pad = (pad, pad)
    ks = len(kernel)
    ds = len(data)
    outlen = ds-ks-1+sum(pad)
    if padmode=="reflect":
        before = data[:pad[0]][::-1]
        after = data[:-pad[1]:-1][::-1]
        data = np.concatenate((before, data, after), axis=0)
    if padmode=="zeros":
        data = np.concatenate((np.zeros(pad[0]), data, np.zeros(pad[1])), axis=0)
    
    output = np.zeros(outlen)
    for i, k in enumerate(kernel):
        output += data[i:outlen+i]*k
    return output

def exp_kernel(len, sig=1):
    kernel = np.exp(-(np.arange(0,len)-(len-1)/2)**2/(2*sig**2))
    return kernel/np.sum(kernel)

def get_data():
    filename = "data.csv"
    data = pd.read_csv(filename, sep=",")
    return data["tempmax"].to_numpy()[-500:]

def plot(data, data_filtered):
    plt.figure(figsize=[10,5])
    plt.plot(data)
    plt.plot(data_filtered)
    plt.grid()
    plt.legend(["ZR","gefilterte ZR"])
    plt.show()
    
def binafy(array):
    facs = (2**np.arange(len(array))[::-1])[:,None]
    return np.sum(array*facs, axis=0)

def get_histogramm(array, k_size=32):
    histogramm = np.zeros(k_size)
    for v in array:
        histogramm[v] += 1
    return histogramm

def entropy(histogramm):
    N = np.sum(histogramm)
    histogramm_N = histogramm/N
    h = -np.sum(histogramm_N*np.log(histogramm_N+1e-10))
    return h, np.log(N)
