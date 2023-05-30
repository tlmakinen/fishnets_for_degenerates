import sympy
import numpy as np
import jax
import jax.numpy as jnp
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import pysr
from chainconsumer import ChainConsumer

import os,glob,sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from data_sr import *


np.set_printoptions(precision=3, suppress=True)


do_plot = bool(int(sys.argv[1]))
do_avg_plot = bool(int(sys.argv[2]))
model_selection = sys.argv[3]
full_model = bool(int(sys.argv[4]))

#modeldir = "/data80/makinen/degenerate_fishnets/sr/l1_reg/"
#modeldir = "/data80/makinen/degenerate_fishnets/sr/new_eta/"
modeldir = "/data80/makinen/degenerate_fishnets/sr/full_k_eta/"
thetadir = '/home/makinen/repositories/fishnets_for_degenerates/'


models = [load_eta_component(i, parentdir=modeldir) for i in range(6)]
params = ["eta%d"%(i) for i in range(6)]

print("--- ETA HALL OF FAME ---")
for i,m in enumerate(models):
    m.model_selection = model_selection
    print("--------------------")
    if full_model:
        print(params[i] + " =\n", m)
    else:
        print(params[i] + " =", m.sympy())

print("--------------------")

theta_ = np.load(thetadir + 'theta_true_for_sr.npy')
F_theta = np.load(thetadir + 'F_theta_for_sr.npy')

theta_test = np.load(thetadir + "theta_test.npy")
F_theta_test = np.load(thetadir + "F_theta_test.npy")

ymin,ymax = np.load(modeldir + "minmax.npy")

Feta, Jeta = get_F_eta_sr(models, F_theta_test, theta_test, minmax=(ymin,ymax))


print("average Feta\n", Feta.mean(0))

print("random Feta\n", Feta[np.random.randint(low=0, high=400)])


#print("average Jeta\n", Jeta.mean(0))

#print("variance Jeta\n", Jeta.var(0))

plotdir = thetadir + "symbolic_regression/"


if do_plot:
    # make plot of Fetas
    cs = ChainConsumer()

    indx = [3,59,67,98]

    for i in indx:
        
        name = r"$\theta$ index %d"%(i)

        cs.add_covariance(np.zeros(6), np.linalg.inv(Feta[i]), 
                        parameters=['A\'', 'B\'', 'C\'', 'D\'', 'E\'', 'F\''], name=name) #, color=corner_colors[0])

            #cs.configure(linestyles=["-", "-", "-"], linewidths=[1.0, 1.0, 1.0], usetex=False,
            #        shade=[True, True, False], shade_alpha=[0.7, 0.6, 0.], tick_font_size=8, sigma2d=True)
            

    cs.plotter.plot((7,7))

    plt.savefig(plotdir + "flat_jacobian.png", transparent=False)


if do_avg_plot:
    # make plot of Fetas
    cs = ChainConsumer()

    name = r"average $F_\eta$"

    cs.add_covariance(np.zeros(6), np.linalg.inv(Feta.mean(0)), 
                    parameters=['A\'', 'B\'', 'C\'', 'D\'', 'E\'', 'F\''], name=name) #, color=corner_colors[0])


    cs.plotter.plot((7,7))

    plt.savefig(plotdir + "flat_avg_jac.png", transparent=False)