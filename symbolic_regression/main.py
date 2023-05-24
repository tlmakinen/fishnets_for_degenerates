import sympy
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import pysr
from chainconsumer import ChainConsumer

import os,glob,sys

from data_sr import *

do_plot = bool(int(sys.argv[1]))

modeldir = "/data80/makinen/degenerate_fishnets/sr/l1_reg/"
thetadir = '/home/makinen/repositories/fishnets_for_degenerates/'


models = [load_eta_component(i, parentdir=modeldir) for i in range(6)]

for m in models:
    m.model_selection = "score"

theta_ = np.load(thetadir + 'theta_true_for_sr.npy')
F_theta = np.load(thetadir + 'F_theta_for_sr.npy')

ymin,ymax = np.load(modeldir + "minmax.npy")

Feta, Jeta = get_F_eta_sr(models, F_theta, theta_, minmax=(ymin,ymax))

np.set_printoptions(precision=3, suppress=True)

print("average Feta", Feta.mean(0))

print("random Feta", Feta[np.random.randint(low=0, high=400)])


print("average Jeta", Jeta.mean(0))

print("variance Jeta", Jeta.var(0))


if do_plot:
    plotdir = thetadir + "symbolic_regression/"
    # make plot of Fetas
    cs = ChainConsumer()

    for i in [3, 59, 300, 2000]:
        
        name = r"$\theta$ index %d"%(i)

        cs.add_covariance(np.zeros(6), np.linalg.inv(Feta[i]), 
                        parameters=['A\'', 'B\'', 'C\'', 'D\'', 'E\'', 'F\''], name=name) #, color=corner_colors[0])

            #cs.configure(linestyles=["-", "-", "-"], linewidths=[1.0, 1.0, 1.0], usetex=False,
            #        shade=[True, True, False], shade_alpha=[0.7, 0.6, 0.], tick_font_size=8, sigma2d=True)
            
    cs.plotter.plot((7,7))

    plt.savefig(plotdir + "flattened_jacobian_L1.png", transparent=False)