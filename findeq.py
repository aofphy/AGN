import pandas as pd
import numpy as np
from pysr import *
import matplotlib.pyplot as plt
import os


df = pd.read_csv('Data_AGN_lags_reduce.csv')
y=df['LogM'].values.tolist()
x1=df['logL'].values.tolist()
x2=df['Fvar'].values.tolist()
x3=df['Tlag'].values.tolist()
x=df['mix'].values.tolist()
X = np.array(x)
Y = np.array(y)
#np.arange(0,len(LogM))


# ""Learn equations
equations = pysr(
    X,
    Y,
    niterations=1,
    binary_operators=["+", "*","/","-"],
    unary_operators=[
        "exp",
        "log",  
        "sqrt",
        "invs(x)=x^2",
        "invc(x)=x^3",
        "inv(x) = 1/x",  # Define your own operator! (Julia syntax)
    ],
)

...# (you can use ctl-c to exit early)

print(best(equations))

plt.plot(X,Y,'.',markersize=20)
plt.show()