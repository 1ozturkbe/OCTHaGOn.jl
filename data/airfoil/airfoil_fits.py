"TASOPT c series airfoil fits"
from __future__ import print_function
from builtins import zip
from builtins import range
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gpfit.fit import fit
plt.rcParams.update({'font.size':15})

def text_to_df(filename):
    "parse XFOIL polars and concatente data in DataFrame"
    lines = list(open(filename))
    for i, l in enumerate(lines):
        lines[i] = l.split("\n")[0]
        for j in 10-np.arange(9):
            if " "*j in lines[i]:
                lines[i] = lines[i].replace(" "*j, " ")
            if "---" in lines[i]:
                start = i
    data = {}
    titles = lines[start-1].split(" ")[1:]
    for t in titles:
        data[t] = []

    for l in lines[start+1:]:
        for i, v in enumerate(l.split(" ")[1:]):
            data[titles[i]].append(v)

    df = pd.DataFrame(data)
    df = df.astype(float)
    return df

def fit_setup(thick_range, re_range, M_range):
    "set up x and y parameters for gp fitting"
    cd = []
    tau = []
    mach = []
    re = []
    cl = []
    for m in M_range:
        for n in thick_range:
            for r in re_range:
                dataf = text_to_df("blade_data/blade.c%s.Re%dk.M%s.pol" % (n, r, m))
                for i in range(len(dataf["CD"])):
                    if dataf["CD"][i] and dataf["CL"][i] != 0:
                        cd.append(dataf["CD"][i])
                        cl.append(dataf["CL"][i])
                        re.append(r)
                        tau.append(float(n)/1000)
                        mach.append(m)

    u1 = np.hstack(re)
    print(u1)
    u2 = np.hstack(tau)
    print(u2)
    u3 = np.hstack(mach)
    u4 = np.hstack(cl)
    w = np.hstack(cd)
    u1 = u1.astype(np.float)
    u2 = u2.astype(np.float)
    u3 = u3.astype(np.float)
    w = w.astype(np.float)
    u = [u1, u2, u3, u4]
    x = np.log(u)
    y = np.log(w)
    return x, y

def make_fit(thick_range, re_range, M_range):
    #call the fit setup function
    x, y = fit_setup(thick_range, re_range, M_range)
    cstrt, rms = fit(x, y, 4, 'SMA')
    print("RMS")
    print(rms)
    return cstrt, rms

if __name__ == "__main__":
    Re = list(range(10000, 35000, 5000))
    thick = ["100", "110", "120", "130", "140", "145"]
    M = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cl = np.linspace(0.35, 0.70, 8)
    X, Y = fit_setup(thick, Re, M) 
    X = np.transpose(X)
    np.savetxt('airfoil_X.csv', X, delimiter=',')
    np.savetxt('airfoil_Y.csv', Y, delimiter=',')
    cstrt, rms = make_fit(thick, Re, M)
    
