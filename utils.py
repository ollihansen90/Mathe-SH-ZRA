import numpy as np
import pandas as pd

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
    return data["tempmax"].to_numpy()
