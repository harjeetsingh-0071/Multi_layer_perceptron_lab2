import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as mp
np.random.seed(30)
x = np.sort(np.random.uniform(low=0, high=1, size=(30)))
w1 = np.round(np.random.rand(6), 4)
w2 = np.round(np.random.rand(6), 4)
b1 = np.round(np.random.rand(6), 4)
b2 = np.round(np.random.rand(1), 1)

def sig(x):
    fun = 1 / (1 + np.exp(-x))
    return fun

def d_sig(x):
    fun = x * (1 - x)
    return fun

def custom_func(x):
    eq = (1 + 0.6 * math.sin(2 * np.pi * x / 0.7)) + (0.3 * math.sin(2 * np.pi * x)) / 2
    return eq

def activation(W1,B1,W2,B2,X):
    out1 = W1 * X + B1
    #act1 = 1 / (1 + np.exp(-out1))
    act1 = sig(out1)
    out2 = act1*W2
    act2 = np.sum(out2) + B2
    return act2,act1,out2+B2
def backpropagation(error,act1,w2):
    d2 = error
    d1 = w2.T* d2 * d_sig(act1)     
    return d2,d1
def update(w1,b1,w2,b2,d2,d1,l_r,h,j):
    w2 = w2 + l_r * d2 * j
    b2 = b2 + l_r * d2
    w1 = w1 + l_r * d1 * h
    b1 = b1 + l_r * d1
    return w1,b1,w2,b2

b = np.zeros(0)
for o in range(len(x)):
    eq = (1 + 0.6 * math.sin(2 * np.pi * x[o] / 0.7)) + (0.3 * math.sin(2 * np.pi * x[o])) / 2
    b = np.append(b,eq)

def train(w1,b1,w2,b2):
    for i in range(len(x)):
        l_r = 0.1
        y,act1 ,out2= activation(w1,b1,w2,b2,x[i])
        a = custom_func(x[i])
        error = a - y
        d2,d1 = backpropagation(error,act1,w2)
        w1,b1,w2,b2 = update(w1,b1,w2,b2,d2,d1,l_r,x[i],act1)
    return w1,b1,w2,b2


for j in range(10000):
    print(j)
    w1, b1, w2, b2 = train(w1, b1, w2, b2)
plt = np.zeros(0)
for k in range(len(x)):
    y, act1,u = activation(w1, b1, w2, b2, x[k])
    plt = np.append(plt,y)
mp.figure("out")
mp.plot(x, b)
mp.plot(x, plt)
mp.show()
