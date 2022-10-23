import math
import numpy as np
import math
import matplotlib.pyplot as mp

x = np.linspace(0, 1, 100)
x_test = np.linspace(0, 1, 20)
y = np.zeros(0)
W_1 = np.round(np.random.rand(6), 4)
W_2 = np.round(np.random.rand(6), 4)
b_1 = np.round(np.random.rand(6), 4)
b_2 = np.round(np.random.rand(1), 4)
#y_dat = np.round(np.random.rand(30), 4)
y_dat = np.random.uniform(low=0, high=2, size=(100))
y_test = np.round(np.random.rand(20), 4)
w1_up = np.zeros(1)
b1_up = np.zeros(1)
w2_up = np.zeros(1)
b2_up = np.zeros(1)


def sig(x):
    fun = 1 / (1 + np.exp(-x))
    return fun


def d_sig(x):
    fun = sig(x) * (1 - sig(x))
    return fun


def activation(W1, X, B1, W2, B2):
    cal_1 = W1 * X + B1
    z_1 = sig(cal_1)
    Z1 = z_1
    Z2 = Z1*W2
    out = np.round(np.sum(Z2), 6) + B2
    z_2 = out
    return z_2, out, Z1


def back_prop(e, z, W2, Z):  #second layer ka outpout ayega idhar
    del_L = e * 1
    del_l = (W2.T * del_L) * d_sig(Z)
    return del_L, del_l


def upgrade_parms(w1, b1, w2, b2, e1, e2, l_r, Z1, Z2):
    #layer 2
    b2_up = b2 + l_r * e2
    w2_up = w2 + l_r * e2
    w1_up = w1 + l_r * e1
    b1_up = b1 + l_r * e1
    return w1_up, w2_up, b1_up, b2_up


def custom_func(x):
    eq = (1 + 0.6 * math.sin(2 * np.pi * x / 0.7)) + (0.3 * math.sin(2 * np.pi * x)) / 2
    #eq = math.sin(x) + pow(x,4)
    #eq = math.sin(x[o])
    return eq


def error_func(out, i):
    y = np.round(custom_func(x[i]), 6)
    error = y - out
    return error

# program starts here :(

for o in range(len(x)):
    eq = (1 + 0.6 * math.sin(2 * np.pi * x[o] / 0.7)) + (0.3 * math.sin(2 * np.pi * x[o])) / 2
    #eq = math.sin(x[o]) + pow(x[o],4)
    #eq = math.sin(x[o])
    y = np.append(y,eq)
mp.plot(x, y, 2)
print("initial values:",W_1, b_1, W_2, b_2)
l_r = 1

for j in range (3000):
    print(" weigth  parameters:", W_1, b_1, W_2, b_2)
    imp_dat = np.zeros(0)
    for i in range(len(x)):
        d_i, output_z, Z_out1 = activation(W_1, y_dat[i], b_1, W_2, b_2)
        error = error_func(d_i, i)
        imp_dat = np.append(imp_dat, d_i)
        error_layer2, error_layer1 = back_prop(error, d_i, W_2, Z_out1)
          # error values after one iteration
        W_1, W_2, b_1, b_2 =  upgrade_parms(W_1, b_1, W_2, b_2, error_layer1, error_layer2, l_r, Z_out1, d_i)
        #mp.scatter(x[i], imp_dat[i])
print(j)
mp.scatter(x, imp_dat)

print(" final  parameters:",W_1, b_1, W_2, b_2)
n = 0
y_plot = []
ww = np.linspace(0, 1,10)

#for g in range(len(y_test)):
 #   cal_1 = 0
  #  x = x_test[g]
   # cal_1 = W_1 * y_test[g] + b_1
    #z_1 = sig(cal_1)
    #Z1 = z_1
   # Z2 = Z1 * W_2
  #  out = np.round(np.sum(Z2), 6) + b_2
 #   print(out)
#    mp.scatter(x, out )
    #y_plot= np.append(y_plot,test_y)

mp.show()




